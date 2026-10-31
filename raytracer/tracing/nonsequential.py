"""Non-sequential tracing: light goes wherever the physics sends it.

No order is assumed. Every ray is tested against every physical boundary
of the system; at the first one it meets, what happens follows from the
boundary itself:

- at an optical surface the light divides into a reflected and a
  transmitted part, weighted by the coating (bare surfaces by Fresnel),
  with total internal reflection where Snell's law allows no transmission.
  The ray goes on along the branch a sequential trace would take
  (refraction through glass, reflection off a mirror; transmission
  through a beam splitter), and the other part is a new ray split off it;
- a lens rim or a diaphragm blade absorbs the ray, a detector detects it.

With ``split=False`` a ray keeps all its power and splits off nothing —
the clean picture for drawing, and row for row the rays a sequential trace
gives — except at a beam splitter, where physics offers no single branch.
All rays advance together, one interaction at a time.

Virtual stops are reference planes of a sequential design, not hardware:
light passes them untouched here.
"""

from __future__ import annotations

import numpy as np

from ..elements import BLADE, DETECTOR, EDGE
from ..interfaces import reflect, refract
from ..rays import Rays, Status
from ..sources import Source
from ..system import DETECT, REFLECT, TRANSMIT, OpticalSystem, PhysicalFace
from .arrival import arrive, indices

ABSORB = "absorb"


class NonSequentialTracer:
    """Trace rays through a system with no assumed order.

    ``wavelength_um`` is given to rays that state none.
    """

    def __init__(
        self,
        system: OpticalSystem,
        *,
        wavelength_um: float | None = None,
        max_interactions: int = 12,
        split: bool = True,
        min_power: float = 1e-4,
        t_min: float = 1e-7,
    ) -> None:
        self.system = system
        self.wavelength_um = wavelength_um
        self.max_interactions = int(max_interactions)
        self.split = bool(split)
        self.min_power = float(min_power)
        self.t_min = float(t_min)
        self.faces: list[PhysicalFace] = [f for f in system.faces if not f.virtual]

    # -- the trace, interaction by interaction --------------------------------

    def trace(self, sources: Source | Rays | list) -> Rays:
        """The rays of *sources* (sources, or rays), carried wherever physics sends them, with those split off them."""

        if isinstance(sources, (Source, Rays)):
            sources = [sources]
        rays = Rays.concatenate([s if isinstance(s, Rays) else s.emit() for s in sources]).stated(self.wavelength_um)
        followed = np.flatnonzero(rays.status == Status.IN_FLIGHT)
        while followed.size:
            rays, followed = self._interact(rays, followed)
        return rays

    def _first_hits(self, origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to, and index of, the first boundary each ray meets (−1: none)."""

        best = np.full(len(origin), np.inf)
        which = np.full(len(origin), -1)
        for k, face in enumerate(self.faces):
            o = face.pose.to_local(origin)
            d = face.pose.direction_to_local(direction)
            if face.role == EDGE:
                t = face.wall.wall_hits(o, d, t_min=self.t_min).min(axis=1)
            else:
                hits = face.surface.intersect(o, d, mode="forward", t_min=self.t_min)
                t = np.where(hits.ok, hits.t, np.inf)
                q = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
                on_patch = np.ones(len(origin), dtype=bool)
                if face.extent is not None:
                    on_patch &= face.extent.contains(q[:, 0], q[:, 1])
                if face.hole is not None:
                    on_patch &= ~face.hole.contains(q[:, 0], q[:, 1])
                t = np.where(on_patch, t, np.inf)
            closer = t < best
            best = np.where(closer, t, best)
            which = np.where(closer, k, which)
        return best, which

    def _interact(self, rays: Rays, followed: np.ndarray) -> tuple[Rays, np.ndarray]:
        """Carry the rays in rows *followed* to the first boundary each meets, and act there.

        Returns the rays — a point more for those that met a boundary, and
        any split off them — with the rows still to follow.
        """

        # A ray on a boundary restarts a hair past it, so as not to meet that boundary again.
        lift = np.where([b is not None for b in rays.boundary[followed]], self.t_min, 0.0)
        origin = rays.position[followed] + rays.direction[followed] * lift[:, None]
        t, which = self._first_hits(origin, rays.direction[followed])
        escaped = which < 0
        arrivals, born, follow = [], [], []
        for k in np.unique(which[~escaped]):
            m = which == k
            met, split_off = self._meet(self.faces[k], rays, followed[m], origin[m], t[m], lift[m])
            arrivals += met
            born += split_off
        reach = max((int(j.max()) + 1 for _, j, *_ in arrivals), default=0)
        record = rays._record(points=reach, rays=sum(len(b[0]) for b in born))
        record["status"][followed[escaped]] = Status.ESCAPED
        for row, j, columns, status, keep in arrivals:
            self._put(record, row, j, columns)
            record["status"][row] = status
            follow.append(row[keep])
        new = len(rays)
        for parent, j, columns in born:
            row = np.arange(new, new + len(parent))
            self._put(record, row, j, columns)
            record["parent"][row] = parent
            record["wavelength_um"][row] = record["wavelength_um"][parent]
            follow.append(row)
            new += len(parent)
        return Rays._of(**record), np.concatenate(follow) if follow else np.zeros(0, dtype=int)

    @staticmethod
    def _put(record: dict, row: np.ndarray, j: np.ndarray, columns: dict) -> None:
        """Write the point *j* of each ray in *row*."""

        for name, value in columns.items():
            record[name][row, j] = value
        record["_last"][row] = j

    def _meet(self, face: PhysicalFace, rays: Rays, row: np.ndarray, origin: np.ndarray, t: np.ndarray, lift: np.ndarray):
        """What happens to the rays in *row*, run *t* from *origin* (*lift* past their last point), meeting *face*.

        Returns their new points, ``(rows, j, columns, status, followed)``,
        and the rays split off them, ``(parent rows, j, columns)``.
        """

        direction, power, opl = rays.direction[row], rays.power[row], rays.opl[row]
        j = rays._last[row] + 1
        pose = face.pose
        point = origin + t[:, None] * direction
        local, d = pose.to_local(point), pose.direction_to_local(direction)
        n_minus = indices(face.medium_minus, rays.wavelength_um[row])
        n_plus = indices(face.medium_plus, rays.wavelength_um[row])
        absorbed = np.full(len(row), Status.ABSORBED)

        if face.role == EDGE:  # a lens rim: the ray arrives from the glass when it heads outward
            n1 = np.where(np.einsum("ij,ij->i", d[:, :2], local[:, :2]) > 0.0, n_minus, n_plus)
            met = self._columns(point, direction, power, opl + n1 * lift + n1 * t, face.name, ABSORB)
            return [(row, j, met, absorbed, np.zeros(len(row), dtype=bool))], []

        normal, n1, n2 = arrive(face, local, d, n_minus, n_plus)
        normal = pose.direction_to_world(normal)
        opl = opl + n1 * lift + n1 * t  # the lift is travelled too
        if face.role in (DETECTOR, BLADE) or face.coating.sequential_event == ABSORB:
            event, status = (DETECT, Status.DETECTED) if face.role == DETECTOR else (ABSORB, absorbed)
            met = self._columns(point, direction, power, opl, face.name, event)
            return [(row, j, met, np.full(len(row), status), np.zeros(len(row), dtype=bool))], []

        # The ray goes on along the branch a sequential trace would take — a beam splitter's
        # transmission, the reflection under TIR — and the other branch is a ray split off it.
        cos_i = -np.einsum("ij,ij->i", direction, normal)
        coeffs = face.coating.power(cos_i, n1, n2, None)
        transmitted, tir = refract(direction, normal, n1, n2)
        reflected = reflect(direction, normal)
        R, T = coeffs.R, np.where(tir, 0.0, coeffs.T)
        event = face.coating.sequential_event
        reflects = np.full(len(row), event == REFLECT) | tir
        if self.split or event is None:  # the coating shares the power between the two
            kept, shed = power * np.where(reflects, R, T), power * np.where(reflects, T, R)
        else:  # the ray keeps it all
            kept, shed = power, np.zeros(len(row))
        within = j < self.max_interactions
        goes_on = self._columns(point, np.where(reflects[:, None], reflected, transmitted), kept, opl, face.name,
                                np.where(reflects, REFLECT, TRANSMIT))
        met = [(row, j, goes_on, np.full(len(row), Status.IN_FLIGHT), within & (kept >= self.min_power))]
        new = within & (shed >= self.min_power) & (shed > 0.0)
        if not new.any():
            return met, []
        split_off = self._columns(point[new], np.where(reflects[:, None], transmitted, reflected)[new], shed[new], opl[new],
                                  face.name, np.where(reflects, TRANSMIT, REFLECT)[new])
        return met, [(row[new], j[new], split_off)]

    @staticmethod
    def _columns(point, direction, power, opl, boundary, event) -> dict:
        return {"trajectory": point, "_direction": direction, "_power": power, "_opl": opl, "_boundary": boundary, "_event": event}


__all__ = ["NonSequentialTracer"]
