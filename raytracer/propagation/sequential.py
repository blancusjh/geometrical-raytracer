"""Sequential propagation: every ray through every surface, in order.

Rays are traced through the system's surfaces in prescription order, one
deterministic path per ray -- no branching into reflected+refracted
children (a row is either a mirror or a dielectric interface). All N rays
are advanced together with vectorized numpy operations for performance;
:func:`~raytracer.math.intersections.intersect_rays_with_profile_surface` and
:func:`~raytracer.optics.laws.reflect_batch`/``refract_batch`` do
the actual math this loop is built from. Failures (vignetting, TIR,
non-convergence) do not raise: rays carry a status and the surface index
where they died, so pupil-filling bundles can simply be filtered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from ..design.rows import SurfaceKind
from ..design.system import OpticalSystem
from ..math.intersections import intersect_rays_with_profile_surface
from ..optics.laws import reflect_batch, refract_batch


class TraceStatus(IntEnum):
    OK = 0
    VIGNETTED = 1
    TIR = 2
    DIVERGED = 3
    PARALLEL = 4
    NO_INTERSECTION = 5
    BACKWARD = 6


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
    """Vectorized sequential tracing in local surface frames.

    Origins, directions and returned paths use world coordinates. Geometric
    distances and optical paths are in mm. Virtual extensions contribute signed
    optical path; the object and internal legs require explicit opt-in.
    """

    def __init__(
        self,
        system: OpticalSystem,
        *,
        newton_tol: float = 2e-11,
        max_newton: int = 30,
        clip_apertures: bool = True,
        aperture_slack: float = 1e-7,
        allow_virtual_object: bool = False,
        allow_virtual_image: bool = True,
        allow_virtual_segments: bool = False,
    ) -> None:
        self.system = system
        self.newton_tol = newton_tol
        self.max_newton = max_newton
        self.clip_apertures = clip_apertures
        self.aperture_slack = aperture_slack
        self.allow_virtual_object = allow_virtual_object
        self.allow_virtual_image = allow_virtual_image
        self.allow_virtual_segments = allow_virtual_segments
        if newton_tol <= 0 or max_newton < 1 or aperture_slack < 0:
            raise ValueError("invalid intersection tolerance or iteration limit")

    def with_system(self, system: OpticalSystem) -> "SequentialTracer":
        """Reuse numerical and virtual-path settings with another prescription."""
        from copy import copy

        tracer = copy(self)
        tracer.system = system
        return tracer

    def trace(self, origin, direction, *, keep_path=True, keep_aoi=False) -> TraceResult:
        """Trace one ray in world coordinates."""

        return self.trace_batch(
            np.asarray(origin, dtype=float)[None, :],
            np.asarray(direction, dtype=float)[None, :],
            keep_paths=keep_path,
            keep_aoi=keep_aoi,
        )[0]

    def trace_batch(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        *,
        keep_paths: bool = False,
        keep_aoi: bool = False,
        stop_at: int | None = None,
    ) -> BatchTraceResult:
        """Trace a bundle in world coordinates; local frames place every surface.

        OPL is the signed sum of n times segment length. Backward propagation
        is allowed only for an explicitly enabled virtual object (first leg)
        or virtual image (last leg). ``allow_virtual_segments`` explicitly
        permits algebraic backward transfers inside a prescription, e.g. a
        dummy stop plane coinciding with a curved surface's vertex plane.
        No restart displacement is needed in an ordered surface sequence.
        ``stop_at`` terminates on that surface without an image-plane transfer.
        """

        system = self.system
        system._rebuild()
        points = np.array(origins, dtype=float, copy=True)
        directions = np.array(directions, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or directions.shape != points.shape:
            raise ValueError("origins and directions must have matching shape (N, 3)")
        lengths = np.linalg.norm(directions, axis=1)
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(directions)):
            raise ValueError("ray coordinates must be finite")
        if np.any(lengths == 0):
            raise ValueError("ray directions must be nonzero")
        directions /= lengths[:, None]
        if stop_at is not None and not 0 <= stop_at < len(system.rows):
            raise ValueError("stop_at must identify a surface in the system")

        count = len(points)
        surface_count = len(system.rows) if stop_at is None else stop_at + 1
        status = np.full(count, TraceStatus.OK, dtype=np.int64)
        failed = np.full(count, -1, dtype=np.int64)
        alive = np.ones(count, dtype=bool)
        opl = np.zeros(count)
        paths = np.full((count, surface_count + 2, 3), np.nan) if keep_paths else None
        aoi = np.full((count, surface_count), np.nan) if keep_aoi else None
        if paths is not None:
            paths[:, 0] = points

        def fail(mask, reason, surface):
            mask = alive & mask
            status[mask] = reason
            failed[mask] = surface
            alive[mask] = False

        for index, row in enumerate(system.rows[:surface_count]):
            if not alive.any():
                break
            frame = system.surface_frame(index)
            local_points = frame.to_local(points)
            local_directions = frame.direction_to_local(directions)
            hit = intersect_rays_with_profile_surface(
                local_points,
                local_directions,
                0.0,
                row.profile,
                alive,
                max_newton=self.max_newton,
                newton_tol=self.newton_tol,
            )
            fail(hit.parallel, TraceStatus.PARALLEL, index)
            fail(hit.outside_domain, TraceStatus.NO_INTERSECTION, index)
            fail(hit.diverged, TraceStatus.DIVERGED, index)
            allow_backward = (index == 0 and self.allow_virtual_object) or (
                index > 0 and self.allow_virtual_segments
            )
            if not allow_backward:
                fail(hit.distance < -self.newton_tol, TraceStatus.BACKWARD, index)

            # Replace failed geometry before vector arithmetic; dead rays stay masked.
            distance = np.where(alive, hit.distance, 0.0)
            local_hits = local_points + distance[:, None] * local_directions
            aperture = row.clear_aperture
            if self.clip_apertures and aperture is not None:
                fail(
                    ~aperture.contains(local_hits[:, :2], self.aperture_slack),
                    TraceStatus.VIGNETTED,
                    index,
                )
            world_hits = frame.to_world(local_hits)
            opl += np.where(alive, system.n_before[index] * distance, 0.0)
            if paths is not None:
                paths[alive, index + 1] = world_hits[alive]

            normal = frame.direction_to_world(hit.normals)
            normal[~alive] = [0.0, 0.0, 1.0]
            facing = np.einsum("ij,ij->i", directions, normal) > 0
            normal[facing] *= -1
            if aoi is not None:
                cosine = -np.einsum("ij,ij->i", directions, normal)
                aoi[alive, index] = np.rad2deg(np.arccos(np.clip(cosine[alive], -1, 1)))

            if row.kind is SurfaceKind.MIRROR:
                outgoing = reflect_batch(directions, normal)
            elif row.kind is SurfaceKind.STOP or system.n_before[index] == system.n_after[index]:
                outgoing = directions.copy()
            else:
                outgoing, tir = refract_batch(
                    directions, normal, system.n_before[index], system.n_after[index]
                )
                fail(tir, TraceStatus.TIR, index)
            points[alive] = world_hits[alive]
            directions[alive] = outgoing[alive]

        image = points.copy()
        if stop_at is None:
            image_frame = system.image_frame
            local_points = image_frame.to_local(points)
            local_directions = image_frame.direction_to_local(directions)
            fail(np.abs(local_directions[:, 2]) < 1e-14, TraceStatus.PARALLEL, surface_count)
            distance = np.divide(
                -local_points[:, 2],
                local_directions[:, 2],
                out=np.zeros(count),
                where=alive,
            )
            if not self.allow_virtual_image:
                fail(distance < -self.newton_tol, TraceStatus.BACKWARD, surface_count)
            image = points + distance[:, None] * directions
            opl += np.where(alive, system.n_image * distance, 0.0)
        image[~alive] = np.nan
        if paths is not None:
            paths[alive, -1] = image[alive]
        return BatchTraceResult(image, directions, opl, status, failed, paths, aoi)


__all__ = ["TraceStatus", "TraceResult", "BatchTraceResult", "SequentialTracer"]
