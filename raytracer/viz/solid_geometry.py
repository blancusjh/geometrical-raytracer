"""Closed display volumes bounded by real optical faces in world coordinates.

The circular clear aperture is used as the display rim, not as a mechanical
specification. Only coaxial circular face pairs without obscurations are
inferred. Unsupported geometry raises rather than inventing a lens body.
"""

from dataclasses import dataclass

import numpy as np

from ..design.rows import SurfaceKind
from ..surfaces.apertures import CircularAperture


@dataclass(frozen=True)
class LensMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    rims: tuple[np.ndarray, np.ndarray]
    surface_indices: tuple[int, int]
    material: str
    refractive_index: float


def glass_blue(index, *, limits=(1.45, 1.80)):
    """Fixed, scene-independent blue scale; indices outside limits saturate."""
    if not np.isfinite(index) or not limits[0] < limits[1]:
        raise ValueError("a finite index and increasing color limits are required")
    amount = np.clip((index - limits[0]) / (limits[1] - limits[0]), 0, 1)
    pale = np.array([0.53, 0.82, 0.98])
    deep = np.array([0.08, 0.29, 0.72])
    return tuple(pale * (1 - amount) + deep * amount)


def lens_meshes(system, *, radial_samples=24, angular_samples=128):
    """Triangulate positive-index dense gaps, retaining rigid placements.

    Supported faces have equal circular apertures and a common local axis.
    Edge crossing is rejected: rendering must not hide invalid prescriptions.
    Every triangle mesh has unique poles and a closed rim, with no degenerate
    triangles. The routine has no graphics-context or optional dependencies.
    """
    if radial_samples < 2 or angular_samples < 8:
        raise ValueError("mesh sampling needs at least 2 radial and 8 angular samples")
    system._rebuild()
    meshes = []
    angles = np.arange(angular_samples) * 2 * np.pi / angular_samples
    for index in range(len(system.rows) - 1):
        first, second = system.rows[index : index + 2]
        if system.n_after[index] <= 1 + 1e-9:
            continue
        if first.kind is not SurfaceKind.REFRACT or second.kind is not SurfaceKind.REFRACT:
            raise ValueError("solid display needs two adjacent refracting boundaries")
        aperture, rear_aperture = first.clear_aperture, second.clear_aperture
        if (
            not isinstance(aperture, CircularAperture)
            or not isinstance(rear_aperture, CircularAperture)
            or aperture.inner_radius
            or rear_aperture.inner_radius
            or not np.isclose(aperture.radius, rear_aperture.radius, rtol=0, atol=1e-10)
        ):
            raise ValueError("solid display needs matching unobscured circular face apertures")
        front_frame, back_frame = system.surface_frame(index), system.surface_frame(index + 1)
        separation = front_frame.to_local(back_frame.origin)
        if not np.allclose(
            front_frame.rotation, back_frame.rotation, rtol=0, atol=1e-11
        ) or not np.allclose(separation[:2], 0, rtol=0, atol=1e-10):
            raise ValueError("solid display needs coaxial faces; whole-element tilts are supported")
        radii = np.linspace(0, aperture.radius, radial_samples + 1)
        front_sag = first.profile.sag(radii)
        back_sag = second.profile.sag(radii) + separation[2]
        if not np.all(np.isfinite(front_sag + back_sag)) or np.any(back_sag <= front_sag):
            raise ValueError(
                "lens faces cross or exceed their sag domain inside the display aperture"
            )
        xy = np.vstack(
            [
                np.zeros((1, 2)),
                np.column_stack(
                    [
                        (radii[1:, None] * np.cos(angles)).ravel(),
                        (radii[1:, None] * np.sin(angles)).ravel(),
                    ]
                ),
            ]
        )
        points = []
        for row, frame in [(first, front_frame), (second, back_frame)]:
            local = np.column_stack([xy, row.profile.sag(np.linalg.norm(xy, axis=1))])
            points.append(frame.to_world(local))
        count = len(xy)
        triangles = []
        for offset, reverse in [(0, True), (count, False)]:
            face = []
            for angle in range(angular_samples):
                nxt = (angle + 1) % angular_samples
                face.append([offset, offset + 1 + angle, offset + 1 + nxt])
            for ring in range(radial_samples - 1):
                start = offset + 1 + ring * angular_samples
                for angle in range(angular_samples):
                    nxt = (angle + 1) % angular_samples
                    a, b = start + angle, start + nxt
                    c, d = a + angular_samples, b + angular_samples
                    face.extend([[a, c, d], [a, d, b]])
            triangles.extend([t[::-1] for t in face] if reverse else face)
        outer = 1 + (radial_samples - 1) * angular_samples
        for angle in range(angular_samples):
            a, b = outer + angle, outer + (angle + 1) % angular_samples
            triangles.extend([[a, b, b + count], [a, b + count, a + count]])
        rims = tuple(np.vstack([p[outer:], p[outer]]) for p in points)
        meshes.append(
            LensMesh(
                np.vstack(points),
                np.asarray(triangles),
                rims,
                (index, index + 1),
                first.material_after.name,
                float(system.n_after[index]),
            )
        )
    return meshes
