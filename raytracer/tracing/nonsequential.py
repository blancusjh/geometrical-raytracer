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
mirror), which is the clean picture for drawing. Rays advance generation
by generation, all at once.

Virtual stops are reference planes of a sequential design, not hardware:
light passes them untouched here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..elements import BLADE, DETECTOR, EDGE, Sensor
from ..interfaces import reflect, refract
from ..sources import Rays, Source
from ..system import OpticalSystem, PhysicalFace

PRIMARY, REFLECTED, TRANSMITTED = 0, 1, 2


@dataclass
class DetectorHits:
    """What one detector recorded: world and local points, directions, power, wavelength, OPL."""

    name: str
    points: np.ndarray
    local_points: np.ndarray
    directions: np.ndarray
    power: np.ndarray
    wavelength_um: np.ndarray
    opl: np.ndarray
    generation: np.ndarray

    def __len__(self) -> int:
        return self.points.shape[0]

    def histogram(self, axis: int = 1, bins: int = 100, range=None) -> tuple[np.ndarray, np.ndarray]:
        """Power per bin along a local axis (0 = x, 1 = y): ``(power, bin_edges)``."""

        return np.histogram(self.local_points[:, axis], bins=bins, range=range, weights=self.power)


@dataclass
class RayTree:
    """Every traced segment, with its genealogy.

    Segment ``i`` runs from ``start[i]`` to ``end[i]``; ``parent[i]`` is the
    segment it was born from (−1 for a source ray), ``kind[i]`` says how
    (primary / reflected / transmitted), ``face[i]`` names the boundary it
    ended on (``None`` for a ray leaving the system).
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
    opl_end: np.ndarray
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
        # A detector is recorded under its sensor's name, also where the sensor
        # coincides with another element's face (a detector on a lens's back).
        self._detector_names = {
            k: next((n for n, e in zip(f.names, f.elements) if isinstance(system[e], Sensor)), f.name)
            for k, f in enumerate(self.faces)
            if f.role == DETECTOR
        }

    def _indices(self, face: PhysicalFace, wavelength: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wl = np.where(np.isnan(wavelength), self.wavelength_um if self.wavelength_um is not None else np.nan, wavelength)
        out = []
        for medium in (face.medium_minus, face.medium_plus):
            values = np.empty_like(wl)
            for w in np.unique(wl):
                key = 0.0 if np.isnan(w) else float(w)
                try:
                    n = medium.index(key)
                except ValueError as exc:
                    raise ValueError(f"{exc} (give the sources or the tracer a wavelength_um)") from None
                values[(wl == w) | (np.isnan(wl) & np.isnan(w))] = n
            out.append(values)
        return out[0], out[1]

    def _first_hits(self, o: np.ndarray, d: np.ndarray):
        n = o.shape[0]
        best = np.full(n, np.inf)
        which = np.full(n, -1)
        for k, face in enumerate(self.faces):
            pose = face.pose
            ol = pose.to_local(o)
            dl = pose.direction_to_local(d)
            if face.role == EDGE:
                w = face.wall
                a = dl[:, 0] ** 2 + dl[:, 1] ** 2
                b = ol[:, 0] * dl[:, 0] + ol[:, 1] * dl[:, 1]
                c = ol[:, 0] ** 2 + ol[:, 1] ** 2 - w.radius**2
                with np.errstate(divide="ignore", invalid="ignore"):
                    root = np.sqrt(np.where(b * b - a * c >= 0.0, b * b - a * c, np.nan))
                    roots = np.stack([(-b - root) / a, (-b + root) / a], axis=1)
                z = ol[:, 2:3] + roots * dl[:, 2:3]
                lo, hi = min(w.z_front, w.z_back), max(w.z_front, w.z_back)
                ok = np.isfinite(roots) & (roots > self.t_min) & (z >= lo) & (z <= hi)
                t = np.where(ok, roots, np.inf).min(axis=1)
            else:
                hits = face.surface.intersect(ol, dl, mode="forward", t_min=self.t_min)
                t = np.where(hits.ok, hits.t, np.inf)
                q = ol + np.where(np.isfinite(t), t, 0.0)[:, None] * dl
                inside = np.ones(n, dtype=bool)
                if face.extent is not None:
                    inside &= face.extent.contains(q[:, 0], q[:, 1])
                if face.hole is not None:
                    inside &= ~face.hole.contains(q[:, 0], q[:, 1])
                t = np.where(inside, t, np.inf)
            closer = t < best
            best = np.where(closer, t, best)
            which = np.where(closer, k, which)
        return best, which

    def trace(self, sources: Source | Rays | list) -> RayTree:
        if isinstance(sources, (Source, Rays)):
            sources = [sources]
        batch = Rays.concatenate([s if isinstance(s, Rays) else s.emit() for s in sources])

        o, d = batch.origins, batch.directions
        power, wl = batch.power, batch.wavelength_um
        generation = np.ones(len(batch), dtype=int)
        parent = np.full(len(batch), -1)
        kind = np.full(len(batch), PRIMARY)
        opl = np.zeros(len(batch))

        record = {k: [] for k in ("start", "end", "direction", "power", "wl", "gen", "parent", "kind", "face", "opl")}
        hits_by_detector: dict[str, list] = {}
        offset = 0

        while o.shape[0]:
            t, which = self._first_hits(o, d)
            escaped = which < 0
            end = o + np.where(escaped, self.escape_length, t)[:, None] * d
            ids = np.arange(offset, offset + o.shape[0])
            offset += o.shape[0]
            seg_opl = np.zeros(o.shape[0])

            new = {k: [] for k in ("o", "d", "power", "wl", "gen", "parent", "kind", "opl")}
            face_names: list = [None] * o.shape[0]
            for k in np.unique(which[~escaped]):
                face = self.faces[k]
                m = which == k
                for j in np.flatnonzero(m):
                    face_names[j] = face.name
                pose = face.pose
                ql = pose.to_local(end[m])
                dl = pose.direction_to_local(d[m])
                if face.role == EDGE:
                    continue
                normal_local = face.surface.normal(ql)
                going_plus = np.einsum("ij,ij->i", dl, normal_local) > 0.0
                n_minus, n_plus = self._indices(face, wl[m])
                n1 = np.where(going_plus, n_minus, n_plus)
                n2 = np.where(going_plus, n_plus, n_minus)
                seg_opl[m] = n1 * t[m]
                if face.role == DETECTOR:
                    hits_by_detector.setdefault(self._detector_names[k], []).append(
                        (end[m], ql, d[m], power[m], wl[m], opl[m] + n1 * t[m], generation[m])
                    )
                    continue
                if face.role == BLADE:
                    continue
                normal = pose.direction_to_world(normal_local)
                normal = np.where(going_plus[:, None], -normal, normal)
                cos_i = -np.einsum("ij,ij->i", d[m], normal)
                coeffs = face.coating.power(cos_i, n1, n2, None)
                transmitted, tir = refract(d[m], normal, n1, n2)
                R = coeffs.R
                T = np.where(tir, 0.0, coeffs.T)
                reflected = reflect(d[m], normal)
                if self.split:
                    branches = [(REFLECTED, reflected, R, np.ones_like(R, bool)), (TRANSMITTED, transmitted, T, np.ones_like(T, bool))]
                else:
                    event = face.coating.sequential_event
                    take_r = (R > T) if event is None else (np.full(R.shape, event == "reflect") | tir)
                    one = np.ones_like(R)
                    branches = [(REFLECTED, reflected, one, take_r), (TRANSMITTED, transmitted, one, ~take_r)]
                for code, direction, fraction, keep in branches:
                    p_child = power[m] * fraction
                    keep = keep & (p_child >= self.min_power) & (generation[m] < self.max_interactions)
                    if not keep.any():
                        continue
                    new["o"].append(end[m][keep] + direction[keep] * self.t_min)
                    new["d"].append(direction[keep])
                    new["power"].append(p_child[keep])
                    new["wl"].append(wl[m][keep])
                    new["gen"].append(generation[m][keep] + 1)
                    new["parent"].append(ids[m][keep])
                    new["kind"].append(np.full(int(keep.sum()), code))
                    # The child starts t_min past the hit, in the medium it travels in.
                    n_child = n1 if code == REFLECTED else n2
                    new["opl"].append(opl[m][keep] + n1[keep] * t[m][keep] + n_child[keep] * self.t_min)

            record["start"].append(o)
            record["end"].append(end)
            record["direction"].append(d)
            record["power"].append(power)
            record["wl"].append(wl)
            record["gen"].append(generation)
            record["parent"].append(parent)
            record["kind"].append(kind)
            record["face"].extend(face_names)
            record["opl"].append(opl + seg_opl)

            if not new["o"]:
                break
            o = np.vstack(new["o"])
            d = np.vstack(new["d"])
            power = np.concatenate(new["power"])
            wl = np.concatenate(new["wl"])
            generation = np.concatenate(new["gen"])
            parent = np.concatenate(new["parent"])
            kind = np.concatenate(new["kind"])
            opl = np.concatenate(new["opl"])

        detectors = {}
        for name, chunks in hits_by_detector.items():
            cols = list(zip(*chunks))
            detectors[name] = DetectorHits(
                name,
                np.vstack(cols[0]),
                np.vstack(cols[1]),
                np.vstack(cols[2]),
                np.concatenate(cols[3]),
                np.concatenate(cols[4]),
                np.concatenate(cols[5]),
                np.concatenate(cols[6]),
            )
        cat = np.concatenate
        return RayTree(
            start=np.vstack(record["start"]),
            end=np.vstack(record["end"]),
            direction=np.vstack(record["direction"]),
            power=cat(record["power"]),
            wavelength_um=cat(record["wl"]),
            generation=cat(record["gen"]),
            parent=cat(record["parent"]),
            kind=cat(record["kind"]),
            face=record["face"],
            opl_end=cat(record["opl"]),
            detectors=detectors,
        )


__all__ = ["NonSequentialTracer", "RayTree", "DetectorHits", "PRIMARY", "REFLECTED", "TRANSMITTED"]
