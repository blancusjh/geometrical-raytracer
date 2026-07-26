"""Sequential propagation: every ray through every surface, in order.

Rays are traced through the system's surfaces in prescription order, one
deterministic path per ray -- no branching into reflected+refracted
children (a row is either a mirror or a dielectric interface). All N rays
are advanced together with vectorized numpy operations for performance;
:func:`~raytracer.math.intersections.intersect_profile_batch` and
:func:`~raytracer.physics.refraction.reflect_batch`/``refract_batch`` do
the actual math this loop is built from. Failures (vignetting, TIR,
non-convergence) do not raise: rays carry a status and the surface index
where they died, so pupil-filling bundles can simply be filtered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from ..design.surfaces import SurfaceKind
from ..design.system import OpticalSystem
from ..math.intersections import intersect_profile_batch
from ..physics.refraction import reflect_batch, refract_batch


class TraceStatus(IntEnum):
    OK = 0
    VIGNETTED = 1
    TIR = 2
    DIVERGED = 3
    PARALLEL = 4


@dataclass
class TraceResult:
    """Outcome of tracing a single ray."""

    image_point: np.ndarray
    direction: np.ndarray
    opl: float
    status: TraceStatus
    failed_surface: int | None = None
    path: np.ndarray | None = None  # (n_surfaces + 2, 3) incl. start and image point
    aoi_deg: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.status is TraceStatus.OK


@dataclass
class BatchTraceResult:
    """Outcome of tracing N rays at once."""

    image_points: np.ndarray  # (N, 3)
    directions: np.ndarray  # (N, 3)
    opl: np.ndarray  # (N,)
    status: np.ndarray  # (N,) int
    failed_surface: np.ndarray  # (N,) int, -1 when the ray survived
    paths: np.ndarray | None = None  # (N, n_surfaces + 2, 3)
    aoi_deg: np.ndarray | None = None  # (N, n_surfaces)

    @property
    def valid(self) -> np.ndarray:
        return self.status == TraceStatus.OK

    def __len__(self) -> int:
        return self.image_points.shape[0]

    def __getitem__(self, i: int) -> TraceResult:
        return TraceResult(
            image_point=self.image_points[i],
            direction=self.directions[i],
            opl=float(self.opl[i]),
            status=TraceStatus(int(self.status[i])),
            failed_surface=None if self.failed_surface[i] < 0 else int(self.failed_surface[i]),
            path=None if self.paths is None else self.paths[i],
            aoi_deg=None if self.aoi_deg is None else self.aoi_deg[i],
        )


class SequentialTracer:
    """Exact tracer for a :class:`OpticalSystem`."""

    def __init__(
        self,
        system: OpticalSystem,
        *,
        newton_tol: float = 2e-11,
        max_newton: int = 15,
        clip_apertures: bool = True,
        aperture_slack: float = 1e-7,
        restart_offset: float = 1e-8,
    ) -> None:
        self.system = system
        self.newton_tol = newton_tol
        self.max_newton = max_newton
        self.clip_apertures = clip_apertures
        self.aperture_slack = aperture_slack
        self.restart_offset = restart_offset

    # -- public API ------------------------------------------------------

    def trace(
        self,
        origin: ArrayLike,
        direction: ArrayLike,
        *,
        keep_path: bool = True,
        keep_aoi: bool = False,
    ) -> TraceResult:
        """Trace one ray; convenience wrapper around :meth:`trace_batch`."""

        batch = self.trace_batch(
            np.asarray(origin, dtype=float)[None, :],
            np.asarray(direction, dtype=float)[None, :],
            keep_paths=keep_path,
            keep_aoi=keep_aoi,
        )
        return batch[0]

    def trace_batch(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        *,
        keep_paths: bool = False,
        keep_aoi: bool = False,
    ) -> BatchTraceResult:
        """Trace N rays through every surface with a masked, vectorized Newton loop."""

        system = self.system
        p = np.array(origins, dtype=float, copy=True)
        d = np.array(directions, dtype=float, copy=True)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("origins must have shape (N, 3)")
        if d.shape != p.shape:
            raise ValueError("directions must match origins in shape")
        d /= np.linalg.norm(d, axis=1, keepdims=True)

        n_rays = p.shape[0]
        n_surfaces = len(system.rows)
        status = np.zeros(n_rays, dtype=np.int64)
        failed = np.full(n_rays, -1, dtype=np.int64)
        opl = np.zeros(n_rays)
        alive = np.ones(n_rays, dtype=bool)

        paths = None
        if keep_paths:
            paths = np.full((n_rays, n_surfaces + 2, 3), np.nan)
            paths[:, 0, :] = p
        aoi = np.full((n_rays, n_surfaces), np.nan) if keep_aoi else None

        for i, row in enumerate(system.rows):
            if not alive.any():
                break
            vz = system.vertices[i]
            profile = row.profile

            t, h, slope, parallel, diverged = intersect_profile_batch(
                p, d, vz, profile, alive,
                max_newton=self.max_newton, newton_tol=self.newton_tol,
            )
            if parallel.any():
                status[parallel] = TraceStatus.PARALLEL
                failed[parallel] = i
                alive &= ~parallel
            if diverged.any():
                status[diverged] = TraceStatus.DIVERGED
                failed[diverged] = i
                alive &= ~diverged

            q = p + t[:, None] * d

            if self.clip_apertures and row.semidiameter is not None:
                clipped = alive & (h > row.semidiameter + self.aperture_slack)
                if clipped.any():
                    status[clipped] = TraceStatus.VIGNETTED
                    failed[clipped] = i
                    alive &= ~clipped

            segment = np.linalg.norm(q - p, axis=1)
            opl = np.where(alive, opl + system.n_before[i] * segment, opl)
            if keep_paths:
                paths[alive, i + 1, :] = q[alive]

            # Surface normal from the sag gradient, oriented against the ray.
            safe_h = np.where(h == 0.0, 1.0, h)
            normal = np.column_stack(
                [
                    np.where(h > 0.0, -slope * q[:, 0] / safe_h, 0.0),
                    np.where(h > 0.0, -slope * q[:, 1] / safe_h, 0.0),
                    np.ones(n_rays),
                ]
            )
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            facing = np.einsum("ij,ij->i", d, normal) > 0.0
            normal[facing] *= -1.0

            cos_i = -np.einsum("ij,ij->i", d, normal)
            if keep_aoi:
                aoi[alive, i] = np.degrees(np.arccos(np.clip(np.abs(cos_i[alive]), 0.0, 1.0)))

            if row.kind is SurfaceKind.MIRROR:
                d_new = reflect_batch(d, normal)
            else:
                n1 = system.n_before[i]
                n2 = system.n_after[i]
                if n1 == n2:
                    d_new = d.copy()
                else:
                    d_new, tir = refract_batch(d, normal, n1, n2)
                    tir = alive & tir
                    if tir.any():
                        status[tir] = TraceStatus.TIR
                        failed[tir] = i
                        alive &= ~tir
            d_new /= np.linalg.norm(d_new, axis=1, keepdims=True)

            d = np.where(alive[:, None], d_new, d)
            p = np.where(alive[:, None], q + d * self.restart_offset, p)

        # Final transfer to the image plane.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_img = (system.image_z - p[:, 2]) / d[:, 2]
        image = p + t_img[:, None] * d
        opl = np.where(alive, opl + system.n_after[-1] * np.abs(t_img), opl)
        image[~alive] = np.nan
        if keep_paths:
            paths[alive, -1, :] = image[alive]

        return BatchTraceResult(
            image_points=image,
            directions=d,
            opl=opl,
            status=status,
            failed_surface=failed,
            paths=paths,
            aoi_deg=aoi,
        )


__all__ = ["TraceStatus", "TraceResult", "BatchTraceResult", "SequentialTracer"]
