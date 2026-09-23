"""Sequential optical system: nominal axial prescription with local 3-D placements."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from ..math.transforms import RigidTransform
from ..optics.materials import AIR, Material
from .rows import SurfaceKind, SurfaceRow


class OpticalSystem:
    """Ordered surfaces with nominal axial vertices and independent local frames.

    Vertex positions are accumulated from the signed thickness column; the
    per-gap refractive indices are resolved from the materials. Mirrors and
    stops keep the current medium. The tracer refreshes derived state before
    propagation; mutation helpers update it when needed.

    ``object_z`` is the object-plane position; prescriptions usually omit it
    (recover it with :func:`raytracer.propagation.paraxial.differential_conjugates`).
    """

    def __init__(
        self,
        rows: Sequence[SurfaceRow],
        *,
        wavelength_um: float | None = None,
        object_space: Material = AIR,
        object_z: float | None = None,
        name: str = "",
        frame: RigidTransform | None = None,
        image_placement: RigidTransform | None = None,
    ) -> None:
        if not rows:
            raise ValueError("An optical system needs at least one surface")
        self.rows = list(rows)
        self.wavelength_um = wavelength_um
        self.object_space = object_space
        self.object_z = object_z
        self.name = name
        self.frame = frame or RigidTransform.identity()
        self.image_placement = image_placement or RigidTransform.identity()
        if self.frame.origin.shape != (3,) or self.image_placement.origin.shape != (3,):
            raise ValueError("system and image frames must be three-dimensional")
        self._rebuild()

    # -- derived state -------------------------------------------------

    def _rebuild(self) -> None:
        wl = self.wavelength_um if self.wavelength_um is not None else 0.0
        z = 0.0
        vertices = []
        n_before = []
        n_after = []
        n_current = self.object_space.index(wl)
        for row in self.rows:
            vertices.append(z)
            n_before.append(n_current)
            if row.kind in (SurfaceKind.MIRROR, SurfaceKind.STOP):
                n_next = n_current
            else:
                n_next = row.material_after.index(wl)
            n_after.append(n_next)
            n_current = n_next
            z += row.thickness
        self.vertices = np.asarray(vertices, dtype=float)
        self.n_before = np.asarray(n_before, dtype=float)
        self.n_after = np.asarray(n_after, dtype=float)
        self.image_z = float(z)

    # -- convenience ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[SurfaceRow]:
        return iter(self.rows)

    @property
    def stop_index(self) -> int | None:
        for i, row in enumerate(self.rows):
            if row.kind is SurfaceKind.STOP or row.is_stop:
                return i
        return None

    @property
    def n_image(self) -> float:
        """Refractive index of the final (image-space) medium."""

        return float(self.n_after[-1])

    @property
    def mirror_indices(self) -> list[int]:
        return [i for i, row in enumerate(self.rows) if row.reflective]

    def surface_frame(self, index: int) -> RigidTransform:
        """Local surface axes placed relative to its nominal axial vertex."""

        placement = self.rows[index].placement
        origin = placement.origin + [0.0, 0.0, self.vertices[index]]
        return self.frame.compose(RigidTransform(origin, placement.rotation))

    @property
    def image_frame(self) -> RigidTransform:
        placement = self.image_placement
        origin = placement.origin + [0.0, 0.0, self.image_z]
        return self.frame.compose(RigidTransform(origin, placement.rotation))

    @property
    def is_centered(self) -> bool:
        placements = [row.placement for row in self.rows] + [self.image_placement]
        return all(
            np.allclose(p.origin, 0.0, atol=1e-14, rtol=0)
            and np.allclose(p.rotation, np.eye(3), atol=1e-14, rtol=0)
            for p in placements
        )

    def transform_surfaces(self, indices, transform: RigidTransform) -> None:
        """Move selected surfaces together, in system coordinates, about the origin.

        For rotation about a point c, use translation c - R @ c with rotation R.
        The image plane is independent and is not moved by this operation.
        """

        for index in indices:
            row = self.rows[index]
            vertex = np.array([0.0, 0.0, self.vertices[index]])
            placed = RigidTransform(vertex + row.placement.origin, row.placement.rotation)
            moved = transform.compose(placed)
            row.placement = RigidTransform(moved.origin - vertex, moved.rotation)

    def at_wavelength(self, wavelength_um: float) -> "OpticalSystem":
        """Independent prescription copy, evaluated at another wavelength."""

        import copy

        result = copy.deepcopy(self)
        result.wavelength_um = float(wavelength_um)
        result._rebuild()
        return result

    def require_axial_coordinates(self) -> None:
        """Guard legacy scalar analyses and layouts that assume the global z axis."""

        if (
            not self.is_centered
            or np.any(self.frame.origin)
            or not np.allclose(self.frame.rotation, np.eye(3), atol=1e-14, rtol=0)
        ):
            raise ValueError(
                "this analysis requires a centered system in axial coordinates; "
                "use geometric_aberrations with the explicit image_frame for placed systems"
            )

    # -- mutation (for optimisation loops) -------------------------------

    def set_asphere(
        self,
        index: int,
        coefficients: tuple[float, ...] | list[float],
        conic: float | None = None,
    ) -> None:
        row = self.rows[index]
        row.profile = row.profile.with_coefficients(coefficients, conic)

    def set_thickness(self, index: int, thickness: float) -> None:
        self.rows[index].thickness = float(thickness)
        self._rebuild()

    # -- solid elements (for layout drawing) -----------------------------

    def solid_elements(self) -> list[tuple[int, int, str]]:
        """Return unique (entrance_index, exit_index, material_name) triples.

        Consecutive surface pairs where the first enters a solid medium form
        a physical element. Double-passed elements on folded paths are
        deduplicated by their geometric signature, mirroring the reference
        implementation.
        """

        gases = {"AIR", "VACUUM"}
        pairs: list[tuple[int, int, str]] = []
        signatures = set()
        for i in range(len(self.rows) - 1):
            first, second = self.rows[i], self.rows[i + 1]
            if first.kind is not SurfaceKind.REFRACT:
                continue
            material = first.material_after.name.upper()
            if material in gases or material == "REFL":
                continue
            if second.kind is SurfaceKind.REFRACT and (
                second.material_after.name.upper() == material
            ):
                continue
            endpoints = tuple(
                sorted(
                    [
                        (round(self.vertices[i], 6), round(first.radius, 6)),
                        (round(self.vertices[i + 1], 6), round(second.radius, 6)),
                    ]
                )
            )
            signature = (material, endpoints)
            if signature in signatures:
                continue
            signatures.add(signature)
            pairs.append((i, i + 1, material))
        return pairs

    # -- prescription I/O ------------------------------------------------

    @classmethod
    def from_prescription(cls, path, *, fmt: str = "csv", **kwargs) -> "OpticalSystem":
        from ..io.prescription_csv import FORMAT_READERS

        try:
            reader = FORMAT_READERS[fmt]
        except KeyError:
            raise ValueError(f"Unknown prescription format {fmt!r}") from None
        return reader(path, **kwargs)

    def to_prescription(self, path, *, fmt: str = "csv") -> None:
        from ..io.prescription_csv import FORMAT_WRITERS

        try:
            writer = FORMAT_WRITERS[fmt]
        except KeyError:
            raise ValueError(f"Unknown prescription format {fmt!r}") from None
        writer(self, path)


__all__ = ["OpticalSystem"]
