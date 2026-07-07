"""Exact sequential 3-D ray tracing with vectorized Newton intersection.

All rays are traced through every surface in prescription order. Failures
(vignetting, TIR, non-convergence) do not raise: rays carry a status and the
surface index where they died, so pupil-filling bundles can simply be
filtered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from .surfaces import SurfaceKind
from .system import OpticalSystem


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

            dz = d[:, 2]
            parallel = alive & (np.abs(dz) < 1e-14)
            if parallel.any():
                status[parallel] = TraceStatus.PARALLEL
                failed[parallel] = i
                alive &= ~parallel

            # Newton iteration from the vertex-plane initial guess.
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(alive, (vz - p[:, 2]) / np.where(dz == 0.0, 1.0, dz), 0.0)
                step = np.zeros(n_rays)
                residual = np.zeros(n_rays)
                for _ in range(self.max_newton):
                    q = p + t[:, None] * d
                    h = np.hypot(q[:, 0], q[:, 1])
                    sag, slope = profile.sag_and_slope(h)
                    residual = q[:, 2] - vz - sag
                    radial_dot = np.where(
                        h > 0.0,
                        (q[:, 0] * d[:, 0] + q[:, 1] * d[:, 1]) / np.where(h == 0.0, 1.0, h),
                        0.0,
                    )
                    derivative = dz - slope * radial_dot
                    bad = alive & (np.abs(derivative) < 1e-14)
                    if bad.any():
                        status[bad] = TraceStatus.DIVERGED
                        failed[bad] = i
                        alive &= ~bad
                    step = np.where(
                        alive, residual / np.where(derivative == 0.0, 1.0, derivative), 0.0
                    )
                    t = t - step
                    if not np.any(np.abs(step[alive]) >= self.newton_tol):
                        break

            diverged = alive & ~np.isfinite(t)
            diverged |= alive & (np.abs(step) > 1e-6)
            if diverged.any():
                status[diverged] = TraceStatus.DIVERGED
                failed[diverged] = i
                alive &= ~diverged

            q = p + t[:, None] * d
            h = np.hypot(q[:, 0], q[:, 1])

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
            _, slope = profile.sag_and_slope(h)
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
                d_new = d + 2.0 * cos_i[:, None] * normal
            else:
                n1 = system.n_before[i]
                n2 = system.n_after[i]
                if n1 == n2:
                    d_new = d.copy()
                else:
                    eta = n1 / n2
                    radicand = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
                    tir = alive & (radicand < 0.0)
                    if tir.any():
                        status[tir] = TraceStatus.TIR
                        failed[tir] = i
                        alive &= ~tir
                    root = np.sqrt(np.maximum(radicand, 0.0))
                    d_new = eta * d + (eta * cos_i - root)[:, None] * normal
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
