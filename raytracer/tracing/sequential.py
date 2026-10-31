"""Sequential tracing: every ray visits the boundaries of the system's path, in order.

All N rays advance together, one boundary at a time. At each step the rays
are carried into the boundary's local frame, intersected with its surface,
clipped to its clear aperture, and turned by the event the path states
there. Which medium a ray leaves and which it enters is read off the
boundary itself — the side it arrives from — so a lens crossed twice in a
catadioptric, or a boundary between two cemented glasses, needs no
special case.

Rays that fail are not raised on: each ends with a
:class:`~raytracer.rays.Status` saying why, its trajectory ending where it
stopped, so pupil-filling bundles can simply be filtered afterwards. Rays
that reach the detector end ``DETECTED``.

The tracer also holds the two facts of *use* a sequential trace needs and
the hardware does not: the object plane, and the wavelength of rays that
state none.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import reflect, refract
from ..rays import Rays, Status
from ..system import DETECT, REFLECT, STOP, TRANSMIT, OpticalSystem
from .arrival import arrive, indices


class SequentialTracer:
    """Exact sequential tracer along an :class:`OpticalSystem`'s path.

    ``wavelength_um`` is where the media are evaluated for rays that state
    no wavelength of their own (it may be omitted when every medium has a
    constant index). ``object_z`` is the axial
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

    def replace(self, **changes) -> "SequentialTracer":
        """The same tracer with some settings changed — another system, wavelength or object plane."""

        settings = dict(
            system=self.system,
            wavelength_um=self.wavelength_um,
            object_z=self.object_z,
            newton_tol=self.newton_tol,
            clip_apertures=self.clip_apertures,
            aperture_slack=self.aperture_slack,
            restart_offset=self.restart_offset,
        )
        settings.update(changes)
        return SequentialTracer(settings.pop("system"), **settings)

    def at_wavelength(self, wavelength_um: float) -> "SequentialTracer":
        """The same tracer, evaluating the media at another wavelength."""

        return self.replace(wavelength_um=wavelength_um)

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

    def _media(self, wavelength_um: np.ndarray) -> list:
        """The indices on the two sides of every step, for each ray's wavelength."""

        if np.all(wavelength_um == self.wavelength_um) or (self.wavelength_um is None and np.isnan(wavelength_um).all()):
            return self._n
        return [(indices(s.face.medium_minus, wavelength_um), indices(s.face.medium_plus, wavelength_um)) for s in self.steps]

    def trace(self, rays: Rays) -> Rays:
        """Carry *rays* along the path: each ray in flight meets every step in turn, and ends detected or failed."""

        rays = rays.stated(self.wavelength_um)
        n, n_steps = len(rays), len(self.steps)
        first = rays._last + 1  # where each ray's next point goes
        record = rays._record(points=int(first.max(initial=1)) + n_steps)
        status, last, rows = record["status"], record["_last"], np.arange(n)
        alive = status == Status.IN_FLIGHT
        p, d, opl = rays.position, rays.direction, rays.opl
        media = self._media(record["wavelength_um"])

        def fail(mask: np.ndarray, code: Status) -> None:
            nonlocal alive
            mask = mask & alive
            status[mask] = code
            alive &= ~mask

        for i, step in enumerate(self.steps):
            if not alive.any():
                break
            face = step.face
            pose = face.pose
            pl = pose.to_local(p)
            dl = pose.direction_to_local(d)
            hits = face.surface.intersect(pl, dl, alive)
            fail(hits.parallel, Status.PARALLEL)
            fail(hits.diverged, Status.DIVERGED)
            fail(hits.missed, Status.MISSED)
            if step.event != STOP:
                fail(hits.t < -1e-6, Status.BEHIND)
            t = np.where(alive, hits.t, 0.0)
            ql = pl + t[:, None] * dl
            normal_local, n1, n2 = arrive(face, ql, dl, *media[i])

            region = face.interface.clear_aperture if step.event == STOP else face.clear_aperture
            if self.clip_apertures and region is not None:
                fail(~region.contains(ql[:, 0], ql[:, 1], slack=self.aperture_slack), Status.VIGNETTED)

            q = pose.to_world(ql)
            opl = np.where(alive, opl + n1 * t, opl)
            here, j = rows[alive], first[alive] + i
            record["trajectory"][here, j] = q[here]
            record["_opl"][here, j] = opl[here]
            record["_power"][here, j] = record["_power"][here, last[here]]
            record["_boundary"][here, j] = step.name
            record["_event"][here, j] = step.event
            last[here] = j

            normal = pose.direction_to_world(normal_local)
            if step.event == REFLECT:
                d_new = reflect(d, normal)
            elif step.event == TRANSMIT:
                differ = n1 != n2
                d_new, tir = refract(d, normal, n1, n2)
                d_new = np.where(differ[:, None], d_new, d)
                fail(tir & differ, Status.TIR)
            else:  # a stop, or the detector
                d_new = d
            d = np.where(alive[:, None], d_new, d)
            record["_direction"][here, j] = d[here]
            p = np.where(alive[:, None], q + d * self.restart_offset, p)

        status[alive] = Status.DETECTED
        return Rays._of(**record)


__all__ = ["SequentialTracer"]
