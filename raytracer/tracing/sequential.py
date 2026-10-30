"""Sequential tracing: every ray visits the boundaries of the system's path, in order.

All N rays advance together, one boundary at a time. At each step the rays
are carried into the boundary's local frame, intersected with its surface,
clipped to its clear aperture, and turned by the event the path states
there. Which medium a ray leaves and which it enters is read off the
boundary itself — the side it arrives from — so a lens crossed twice in a
catadioptric, or a boundary between two cemented glasses, needs no
special case.

Rays that fail are not raised on: each carries a :class:`TraceStatus` and
the step where it stopped, so pupil-filling bundles can simply be
filtered afterwards.

The tracer also holds the two facts of *use* a sequential trace needs and
the hardware does not: the wavelength, and the object plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from ..interfaces import reflect, refract
from ..system import DETECT, REFLECT, STOP, TRANSMIT, OpticalSystem


class TraceStatus(IntEnum):
    OK = 0
    VIGNETTED = 1
    TIR = 2
    DIVERGED = 3
    PARALLEL = 4
    MISSED = 5


@dataclass
class TraceResult:
    """Outcome of tracing a single ray."""

    image_point: np.ndarray
    direction: np.ndarray
    opl: float
    status: TraceStatus
    failed_step: int | None = None
    path: np.ndarray | None = None  # (n_steps + 1, 3): start, then the hit at every step
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
    failed_step: np.ndarray  # (N,) int, -1 when the ray survived
    paths: np.ndarray | None = None  # (N, n_steps + 1, 3)
    aoi_deg: np.ndarray | None = None  # (N, n_steps)

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
            failed_step=None if self.failed_step[i] < 0 else int(self.failed_step[i]),
            path=None if self.paths is None else self.paths[i],
            aoi_deg=None if self.aoi_deg is None else self.aoi_deg[i],
        )


class SequentialTracer:
    """Exact sequential tracer along an :class:`OpticalSystem`'s path.

    ``wavelength_um`` is where the media are evaluated (it may be omitted
    when every medium has a constant index). ``object_z`` is the axial
    position of the object plane, used by the field and pupil helpers;
    :func:`~raytracer.tracing.paraxial.solve_object_plane` finds it for a
    prescription that leaves it out.
    """

    def __init__(
        self,
        system: OpticalSystem,
        *,
        wavelength_um: float | None = None,
        object_z: float | None = None,
        newton_tol: float = 2e-11,
        clip_apertures: bool = True,
        aperture_slack: float = 1e-7,
        restart_offset: float = 1e-8,
    ) -> None:
        self.system = system
        self.wavelength_um = wavelength_um
        self.object_z = object_z
        self.newton_tol = newton_tol
        self.clip_apertures = clip_apertures
        self.aperture_slack = aperture_slack
        self.restart_offset = restart_offset
        self.steps = system.path
        if not self.steps:
            raise ValueError("the system's path is empty")
        if self.steps[-1].event != DETECT:
            raise ValueError("a sequential path ends on a detector (a Sensor)")
        wl = 0.0 if wavelength_um is None else wavelength_um
        self._n = [
            (self._index(s.face.medium_minus, wl), self._index(s.face.medium_plus, wl)) for s in self.steps
        ]

    def _index(self, medium, wl: float) -> float:
        try:
            return float(medium.index(wl))
        except ValueError as exc:
            raise ValueError(f"{exc} (pass wavelength_um= to the tracer)") from None

    def at_wavelength(self, wavelength_um: float) -> "SequentialTracer":
        """The same tracer, evaluating the media at another wavelength."""

        return SequentialTracer(
            self.system,
            wavelength_um=wavelength_um,
            object_z=self.object_z,
            newton_tol=self.newton_tol,
            clip_apertures=self.clip_apertures,
            aperture_slack=self.aperture_slack,
            restart_offset=self.restart_offset,
        )

    # -- facts of the path used by the analyses ------------------------------

    def step_z(self, i: int) -> float:
        """Axial (global z) position of the vertex visited at step *i*."""

        return float(self.steps[i].face.pose.position[2])

    @property
    def image_z(self) -> float:
        return self.step_z(len(self.steps) - 1)

    def _arrival_index(self, i: int, previous: np.ndarray) -> float:
        face = self.steps[i].face
        side = (previous - face.pose.position) @ face.pose.axis
        return self._n[i][0] if side < 0.0 else self._n[i][1]

    @property
    def n_object(self) -> float:
        """Index of the medium light leaves the object plane in."""

        z = self.object_z if self.object_z is not None else self.step_z(0) - 1.0
        return self._arrival_index(0, np.array([0.0, 0.0, z]))

    @property
    def n_image(self) -> float:
        """Index of the medium light reaches the detector through."""

        last = len(self.steps) - 1
        return self._arrival_index(last, self.steps[last - 1].face.pose.position) if last else self.n_object

    # -- tracing -------------------------------------------------------------

    def trace(self, origin: ArrayLike, direction: ArrayLike, *, keep_path: bool = True, keep_aoi: bool = False) -> TraceResult:
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
        p = np.array(origins, dtype=float, copy=True)
        d = np.array(directions, dtype=float, copy=True)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("origins must have shape (N, 3)")
        if d.shape != p.shape:
            raise ValueError("directions must match origins in shape")
        d /= np.linalg.norm(d, axis=1, keepdims=True)

        n_rays, n_steps = p.shape[0], len(self.steps)
        status = np.zeros(n_rays, dtype=np.int64)
        failed = np.full(n_rays, -1, dtype=np.int64)
        opl = np.zeros(n_rays)
        alive = np.ones(n_rays, dtype=bool)
        paths = None
        if keep_paths:
            paths = np.full((n_rays, n_steps + 1, 3), np.nan)
            paths[:, 0, :] = p
        aoi = np.full((n_rays, n_steps), np.nan) if keep_aoi else None
        image = np.full((n_rays, 3), np.nan)

        def fail(mask: np.ndarray, code: TraceStatus, step: int) -> None:
            nonlocal alive
            mask = mask & alive
            status[mask] = code
            failed[mask] = step
            alive &= ~mask

        for i, step in enumerate(self.steps):
            if not alive.any():
                break
            face = step.face
            pose = face.pose
            pl = pose.to_local(p)
            dl = pose.direction_to_local(d)
            hits = face.surface.intersect(pl, dl, alive)
            fail(hits.parallel, TraceStatus.PARALLEL, i)
            fail(hits.diverged, TraceStatus.DIVERGED, i)
            fail(hits.missed, TraceStatus.MISSED, i)
            t = np.where(alive, hits.t, 0.0)
            ql = pl + t[:, None] * dl

            if step.clear_aperture is not None:
                region = step.clear_aperture
            else:
                region = face.interface.clear_aperture if step.event == STOP else face.clear_aperture
            if self.clip_apertures and region is not None:
                fail(~region.contains(ql[:, 0], ql[:, 1], slack=self.aperture_slack), TraceStatus.VIGNETTED, i)

            # The normal on the surface's +z side tells which side each ray arrives from.
            normal_local = face.surface.normal(ql)
            going_plus = np.einsum("ij,ij->i", dl, normal_local) > 0.0
            n_minus, n_plus = self._n[i]
            n1 = np.where(going_plus, n_minus, n_plus)
            q = pose.to_world(ql)
            opl = np.where(alive, opl + n1 * t, opl)
            if keep_paths:
                paths[alive, i + 1, :] = q[alive]

            normal = pose.direction_to_world(normal_local)
            normal = np.where(going_plus[:, None], -normal, normal)  # face the incoming ray
            if keep_aoi:
                cos_i = -np.einsum("ij,ij->i", d, normal)
                aoi[alive, i] = np.degrees(np.arccos(np.clip(np.abs(cos_i[alive]), 0.0, 1.0)))

            if step.event == DETECT:
                image[alive] = q[alive]
                break
            if step.event == REFLECT:
                d_new = reflect(d, normal)
            elif step.event == TRANSMIT and n_minus != n_plus:
                n2 = np.where(going_plus, n_plus, n_minus)
                d_new, tir = refract(d, normal, n1, n2)
                fail(tir, TraceStatus.TIR, i)
            else:  # a stop, or a boundary between equal indices
                d_new = d
            d = np.where(alive[:, None], d_new, d)
            p = np.where(alive[:, None], q + d * self.restart_offset, p)

        image[~alive] = np.nan
        return BatchTraceResult(
            image_points=image,
            directions=d,
            opl=opl,
            status=status,
            failed_step=failed,
            paths=paths,
            aoi_deg=aoi,
        )


__all__ = ["TraceStatus", "TraceResult", "BatchTraceResult", "SequentialTracer"]
