"""Non-sequential tracing: light goes wherever the physics sends it.

No order is assumed. Every ray is tested against every physical boundary
of the system; at the first one it meets, what happens follows from the
boundary itself:

- an optical surface splits the ray into a reflected and a transmitted
  child, weighted by its coating (bare surfaces by Fresnel), with total
  internal reflection where Snell's law allows no transmission;
- a lens rim, a diaphragm blade or a detector absorbs it — a detector
  also records it.

With ``split=False`` each ray continues along one branch only, the one a
sequential trace would take (refraction through glass, reflection off a
mirror), which is the clean picture for drawing; a beam splitter still
splits, since physics offers no single branch there. Rays advance
generation by generation, all at once.

Virtual stops are reference planes of a sequential design, not hardware:
light passes them untouched here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from ..elements import BLADE, DETECTOR, EDGE
from ..interfaces import reflect, refract
from ..sources import Rays, Source
from ..system import OpticalSystem, PhysicalFace

PRIMARY, REFLECTED, TRANSMITTED = 0, 1, 2


@dataclass
class DetectorHits:
    """What one detector recorded, a row per arriving ray: where (in the world and on
    the detector), from which direction, with what power, wavelength and OPL."""

    name: str
    point: np.ndarray
    local_point: np.ndarray
    direction: np.ndarray
    power: np.ndarray
    wavelength_um: np.ndarray
    opl: np.ndarray
    generation: np.ndarray

    def __len__(self) -> int:
        return self.point.shape[0]

    def histogram(self, axis: int = 1, bins: int = 100, range=None) -> tuple[np.ndarray, np.ndarray]:
        """Power per bin along a local axis (0 = x, 1 = y): ``(power, bin_edges)``."""

        return np.histogram(self.local_point[:, axis], bins=bins, range=range, weights=self.power)


@dataclass
class RayTree:
    """Every traced segment, a row each, with its genealogy.

    Segment ``i`` runs from ``start[i]`` to ``end[i]``; ``parent[i]`` is the
    segment it was born from (−1 for a source ray), ``kind[i]`` says how
    (primary / reflected / transmitted), ``face[i]`` names the boundary it
    ended on (``None`` for a ray leaving the system), ``opl[i]`` is the
    optical path length at its end.
    """

    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    power: np.ndarray
    wavelength_um: np.ndarray
    generation: np.ndarray
    parent: np.ndarray
    kind: np.ndarray
    face: list
    opl: np.ndarray
    detectors: dict[str, DetectorHits] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.start.shape[0]

    @property
    def escaped(self) -> np.ndarray:
        return np.array([f is None for f in self.face], dtype=bool)

    def segments(self) -> np.ndarray:
        """``(M, 2, 3)`` segment end points, for drawing."""

        return np.stack([self.start, self.end], axis=1)

    def hits(self, detector: str) -> DetectorHits:
        return self.detectors[detector]


@dataclass
class _Front:
    """One generation of rays, all advanced together."""

    origin: np.ndarray
    direction: np.ndarray
    power: np.ndarray
    wavelength_um: np.ndarray
    generation: np.ndarray
    parent: np.ndarray
    kind: np.ndarray
    opl: np.ndarray

    def __len__(self) -> int:
        return self.origin.shape[0]

    def take(self, mask: np.ndarray) -> "_Front":
        return _Front(*(getattr(self, f.name)[mask] for f in fields(self)))

    @staticmethod
    def join(parts: "list[_Front]") -> "_Front":
        return _Front(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(_Front)))


class NonSequentialTracer:
    """Trace sources through a system with no assumed order."""

    def __init__(
        self,
        system: OpticalSystem,
        *,
        wavelength_um: float | None = None,
        max_interactions: int = 12,
        split: bool = True,
        min_power: float = 1e-4,
        escape_length: float = 1e3,
        t_min: float = 1e-7,
    ) -> None:
        self.system = system
        self.wavelength_um = wavelength_um
        self.max_interactions = int(max_interactions)
        self.split = bool(split)
        self.min_power = float(min_power)
        self.escape_length = float(escape_length)
        self.t_min = float(t_min)
        self.faces: list[PhysicalFace] = [f for f in system.faces if not f.virtual]

    # -- the trace, generation by generation ----------------------------------

    def trace(self, sources: Source | Rays | list) -> RayTree:
        if isinstance(sources, (Source, Rays)):
            sources = [sources]
        rays = Rays.concatenate([s if isinstance(s, Rays) else s.emit() for s in sources])
        n = len(rays)
        front = _Front(rays.origin, rays.direction, rays.power, rays.wavelength_um,
                       np.ones(n, dtype=int), np.full(n, -1), np.full(n, PRIMARY), np.zeros(n))
        segments, detections = [], {}
        first_id = 0
        while len(front):
            t, which = self._first_hits(front)
            ids = np.arange(first_id, first_id + len(front))
            first_id += len(front)
            segment, children = self._interact(front, t, which, ids, detections)
            segments.append(segment)
            front = _Front.join(children) if children else front.take(np.zeros(len(front), dtype=bool))
        return self._tree(segments, detections)

    def _first_hits(self, front: _Front) -> tuple[np.ndarray, np.ndarray]:
        """Distance to, and index of, the first boundary each ray meets (−1: none)."""

        best = np.full(len(front), np.inf)
        which = np.full(len(front), -1)
        for k, face in enumerate(self.faces):
            o = face.pose.to_local(front.origin)
            d = face.pose.direction_to_local(front.direction)
            if face.role == EDGE:
                t = face.wall.wall_hits(o, d, t_min=self.t_min).min(axis=1)
            else:
                hits = face.surface.intersect(o, d, mode="forward", t_min=self.t_min)
                t = np.where(hits.ok, hits.t, np.inf)
                q = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
                on_patch = np.ones(len(front), dtype=bool)
                if face.extent is not None:
                    on_patch &= face.extent.contains(q[:, 0], q[:, 1])
                if face.hole is not None:
                    on_patch &= ~face.hole.contains(q[:, 0], q[:, 1])
                t = np.where(on_patch, t, np.inf)
            closer = t < best
            best = np.where(closer, t, best)
            which = np.where(closer, k, which)
        return best, which

    def _interact(self, front: _Front, t: np.ndarray, which: np.ndarray, ids: np.ndarray, detections: dict):
        """Record this generation's segments; return them and the next generation."""

        end = front.origin + np.where(which < 0, self.escape_length, t)[:, None] * front.direction
        opl_end = front.opl.copy()
        names: list = [None] * len(front)
        children: list[_Front] = []
        for k in np.unique(which[which >= 0]):
            face, m = self.faces[k], which == k
            for j in np.flatnonzero(m):
                names[j] = face.name
            if face.role == EDGE:
                continue
            arriving = front.take(m)
            local = face.pose.to_local(end[m])
            normal_local = face.surface.normal(local)
            going_plus = np.einsum("ij,ij->i", face.pose.direction_to_local(arriving.direction), normal_local) > 0.0
            n_minus, n_plus = self._indices(face, arriving.wavelength_um)
            n1, n2 = np.where(going_plus, n_minus, n_plus), np.where(going_plus, n_plus, n_minus)
            opl_end[m] = arriving.opl + n1 * t[m]
            if face.role == DETECTOR:
                detections.setdefault(face.name, []).append(
                    (end[m], local, arriving.direction, arriving.power, arriving.wavelength_um, opl_end[m], arriving.generation)
                )
            elif face.role != BLADE:
                normal = face.pose.direction_to_world(normal_local)
                normal = np.where(going_plus[:, None], -normal, normal)  # facing the incoming ray
                children += self._children(face, arriving, end[m], normal, n1, n2, opl_end[m], ids[m])
        segment = (front, end, names, opl_end)
        return segment, children

    def _children(self, face, arriving: _Front, point, normal, n1, n2, opl, parent_ids) -> list[_Front]:
        """The reflected and transmitted rays a boundary sends on, weighted by its coating."""

        cos_i = -np.einsum("ij,ij->i", arriving.direction, normal)
        coeffs = face.coating.power(cos_i, n1, n2, None)
        transmitted, tir = refract(arriving.direction, normal, n1, n2)
        R, T = coeffs.R, np.where(tir, 0.0, coeffs.T)
        event = face.coating.sequential_event
        if self.split or event is None:
            branches = [(REFLECTED, reflect(arriving.direction, normal), R, n1), (TRANSMITTED, transmitted, T, n2)]
        else:
            takes_reflection = np.full(len(arriving), event == "reflect") | tir
            one = np.ones(len(arriving))
            branches = [
                (REFLECTED, reflect(arriving.direction, normal), np.where(takes_reflection, one, 0.0), n1),
                (TRANSMITTED, transmitted, np.where(takes_reflection, 0.0, one), n2),
            ]
        out = []
        for code, direction, fraction, n_after in branches:
            power = arriving.power * fraction
            keep = (power >= self.min_power) & (arriving.generation < self.max_interactions)
            if keep.any():
                out.append(_Front(
                    point[keep] + direction[keep] * self.t_min,
                    direction[keep],
                    power[keep],
                    arriving.wavelength_um[keep],
                    arriving.generation[keep] + 1,
                    parent_ids[keep],
                    np.full(int(keep.sum()), code),
                    opl[keep] + n_after[keep] * self.t_min,  # the child starts t_min past the hit
                ))
        return out

    # -- media and results -----------------------------------------------------

    def _indices(self, face: PhysicalFace, wavelength: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the media on the −z and +z sides of *face*, per ray wavelength."""

        default = np.nan if self.wavelength_um is None else self.wavelength_um
        wl = np.where(np.isnan(wavelength), default, wavelength)
        out = []
        for medium in (face.medium_minus, face.medium_plus):
            values = np.empty_like(wl)
            for w in np.unique(wl):
                try:
                    n = medium.index(0.0 if np.isnan(w) else float(w))
                except ValueError as exc:
                    raise ValueError(f"{exc} (give the sources or the tracer a wavelength_um)") from None
                values[(wl == w) | (np.isnan(wl) & np.isnan(w))] = n
            out.append(values)
        return out[0], out[1]

    @staticmethod
    def _tree(segments, detections) -> RayTree:
        fronts = _Front.join([f for f, _, _, _ in segments])
        hits = {
            name: DetectorHits(name, *(np.concatenate(column) for column in zip(*chunks)))
            for name, chunks in detections.items()
        }
        return RayTree(
            start=fronts.origin,
            end=np.vstack([e for _, e, _, _ in segments]),
            direction=fronts.direction,
            power=fronts.power,
            wavelength_um=fronts.wavelength_um,
            generation=fronts.generation,
            parent=fronts.parent,
            kind=fronts.kind,
            face=[n for _, _, names, _ in segments for n in names],
            opl=np.concatenate([o for _, _, _, o in segments]),
            detectors=hits,
        )


__all__ = ["NonSequentialTracer", "RayTree", "DetectorHits", "PRIMARY", "REFLECTED", "TRANSMITTED"]
