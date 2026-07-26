"""Sequential optical system: ordered surface rows on a common axis."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from ..physics.materials import AIR, Material
from .surfaces import SurfaceKind, SurfaceRow


class OpticalSystem:
    """An ordered stack of surfaces along the z axis.

    Vertex positions are accumulated from the signed thickness column; the
    per-gap refractive indices are resolved once from the materials (mirrors
    keep the current medium). Mutating helpers (``set_asphere``,
    ``set_thickness``) trigger a rebuild so optimisation loops can refit
    coefficients cheaply.

    ``object_z`` is the object-plane position; prescriptions usually omit it
    (recover it with :func:`raytracer.sequential.paraxial.differential_conjugates`).
    """

    def __init__(
        self,
        rows: Sequence[SurfaceRow],
        *,
        wavelength_um: float | None = None,
        object_space: Material = AIR,
        object_z: float | None = None,
        name: str = "",
    ) -> None:
        if not rows:
            raise ValueError("An optical system needs at least one surface")
        self.rows = list(rows)
        self.wavelength_um = wavelength_um
        self.object_space = object_space
        self.object_z = object_z
        self.name = name
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
            if row.kind is SurfaceKind.MIRROR:
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
            if row.kind is SurfaceKind.STOP:
                return i
        return None

    @property
    def n_image(self) -> float:
        """Refractive index of the final (image-space) medium."""

        return float(self.n_after[-1])

    @property
    def mirror_indices(self) -> list[int]:
        return [i for i, row in enumerate(self.rows) if row.reflective]

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
        from .prescription import FORMAT_READERS

        try:
            reader = FORMAT_READERS[fmt]
        except KeyError:
            raise ValueError(f"Unknown prescription format {fmt!r}") from None
        return reader(path, **kwargs)

    def to_prescription(self, path, *, fmt: str = "csv") -> None:
        from .prescription import FORMAT_WRITERS

        try:
            writer = FORMAT_WRITERS[fmt]
        except KeyError:
            raise ValueError(f"Unknown prescription format {fmt!r}") from None
        writer(self, path)


__all__ = ["OpticalSystem"]
