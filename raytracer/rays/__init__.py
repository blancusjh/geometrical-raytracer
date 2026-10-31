"""Rays: light as geometrical optics knows it, one thing however it is traced.

A ray is the whole broken line light follows. It is born at a source, or
where it splits off another ray; it runs straight through each homogeneous
medium and turns where it meets a boundary, until it is detected, absorbed,
leaves the system, or is no longer followed. Sources create rays; tracers
carry them on — the sequential tracer along the system's designed path, the
non-sequential one wherever physics sends them — and every analysis reads
them the same, whichever tracer ran.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike


class Status(IntEnum):
    """How a ray ended, or that it has not.

    ``IN_FLIGHT``: still travelling — just born, or no longer followed (past
    a tracer's interaction or power limit). ``DETECTED`` on a detector,
    ``ESCAPED`` with nothing ahead, ``ABSORBED`` by a lens rim, a diaphragm
    blade or an absorbing coating. The rest are a sequential path the ray
    could not follow: ``VIGNETTED`` outside a clear aperture, ``TIR`` where
    the path transmits, ``MISSED`` / ``PARALLEL`` the next surface is not on
    its line, ``DIVERGED`` its intersection was not found, ``BEHIND`` the
    next surface lies behind it (allowed only for a stop, which
    prescriptions may place behind the previous vertex).
    """

    IN_FLIGHT = 0
    DETECTED = 1
    ESCAPED = 2
    ABSORBED = 3
    VIGNETTED = 4
    TIR = 5
    MISSED = 6
    PARALLEL = 7
    DIVERGED = 8
    BEHIND = 9


#: What a ray's record holds at each of its points, beside the point itself.
_AT_POINTS = ("_direction", "_power", "_opl", "_boundary", "_event")
#: The rest of the record, as it starts for a ray a tracer adds.
_NEWBORN = {"_last": 0, "wavelength_um": np.nan, "status": Status.IN_FLIGHT, "parent": -1}


class Rays:
    """Rays of light, a row each: where each ray is and where it goes, the light it carries, the way it came.

    Each ray carries a wavelength (µm; NaN until stated, and a tracer states
    its own), a power, and the optical path length travelled since its
    source. Its ``trajectory`` ``(N, V, 3)`` holds the points it has passed:
    point ``j`` is where it was after ``j`` interactions since the light
    left its source, so rays traced along one sequential path line up step
    by step. A ray split off another — at a beam splitter, or a reflection
    off glass — is born at its ``parent``'s point there and has no points
    before it; ``parent`` is a row of the rays the tracer returned (−1 for
    a ray from a source). NaN marks a point a ray does not have.

    At each point a ray also has the direction it left in, its power and
    optical path there, the boundary it lies on and what happened there:
    ``"transmit"``, ``"reflect"``, ``"stop"``, ``"detect"`` or
    ``"absorb"`` (None where it was born). ``position``, ``direction``,
    ``power``, ``opl``, ``boundary`` and ``event`` are their values now, at
    each ray's last point; :meth:`at` gives them at another.

    ``Rays(position, direction, ...)`` makes rays just born there.
    """

    def __init__(
        self,
        position: ArrayLike,
        direction: ArrayLike,
        wavelength_um: ArrayLike = np.nan,
        power: ArrayLike = 1.0,
        opl: ArrayLike = 0.0,
    ) -> None:
        position = np.atleast_2d(np.asarray(position, dtype=float))
        d = np.atleast_2d(np.asarray(direction, dtype=float))
        if position.shape[1:] != (3,) or d.shape != position.shape:
            raise ValueError("position and direction are (N, 3), alike in shape")
        n = position.shape[0]

        def column(value):
            return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()

        self._set(
            trajectory=position[:, None, :].copy(),
            _direction=(d / np.linalg.norm(d, axis=1, keepdims=True))[:, None, :],
            _power=column(power)[:, None],
            _opl=column(opl)[:, None],
            _boundary=np.full((n, 1), None, dtype=object),
            _event=np.full((n, 1), None, dtype=object),
            _last=np.zeros(n, dtype=np.int64),
            wavelength_um=column(wavelength_um),
            status=np.full(n, Status.IN_FLIGHT, dtype=np.int64),
            parent=np.full(n, -1, dtype=np.int64),
        )

    def _set(self, **columns) -> None:
        self.__dict__.update(columns)

    @classmethod
    def _of(cls, **columns) -> "Rays":
        """Rays from their whole record — what a tracer builds."""

        rays = cls.__new__(cls)
        rays._set(**columns)
        return rays

    def _record(self, points: int = 0, rays: int = 0) -> dict:
        """A copy of the whole record, with room for *points* points per ray and *rays* more rays — for a tracer to fill."""

        n, v = len(self), self.trajectory.shape[1]
        record = {}
        for name, value in self.__dict__.items():
            if name == "trajectory" or name in _AT_POINTS:
                grown = np.full((n + rays, max(points, v)) + value.shape[2:], None if value.dtype == object else np.nan, dtype=value.dtype)
                grown[:n, :v] = value
            else:
                grown = np.full(n + rays, _NEWBORN[name], dtype=value.dtype)
                grown[:n] = value
            record[name] = grown
        return record

    # -- now ---------------------------------------------------------------------

    def _now(self, values: np.ndarray) -> np.ndarray:
        return values[np.arange(len(self)), self._last]

    @property
    def position(self) -> np.ndarray:
        return self._now(self.trajectory)

    @property
    def direction(self) -> np.ndarray:
        return self._now(self._direction)

    @property
    def power(self) -> np.ndarray:
        return self._now(self._power)

    @property
    def opl(self) -> np.ndarray:
        return self._now(self._opl)

    @property
    def boundary(self) -> np.ndarray:
        return self._now(self._boundary)

    @property
    def event(self) -> np.ndarray:
        return self._now(self._event)

    @property
    def detected(self) -> np.ndarray:
        """Which rays reached a detector."""

        return self.status == Status.DETECTED

    # -- the way they came -------------------------------------------------------------

    def at(self, where: int | str) -> "Rays":
        """These rays as they were at their point *where* (a number), or where each first met the boundary *where*.

        Rows stay aligned with these rays; one never there has NaN position,
        direction, power and optical path. A ray that went on from there is
        in flight there.
        """

        n, points = len(self), self.trajectory.shape[1]
        if isinstance(where, str):
            met = self._boundary == where
            j = np.where(met.any(axis=1), met.argmax(axis=1), -1)
        else:
            j = np.full(n, int(where)) if 0 <= int(where) < points else np.full(n, -1)
            j[(j >= 0) & ~np.isfinite(self.trajectory[np.arange(n), np.maximum(j, 0), 0])] = -1
        record = self._record()
        later = np.arange(points)[None, :] > j[:, None]
        for name in ("trajectory", *_AT_POINTS):
            record[name][later] = None if record[name].dtype == object else np.nan
        record["_last"] = np.maximum(j, 0)
        record["status"] = np.where((j >= 0) & (j < self._last), Status.IN_FLIGHT, self.status)
        return Rays._of(**record)

    def segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The straight pieces of all the rays, for drawing: ``(points, power, ray, j)``.

        ``points[m]`` ``(2, 3)`` runs from point ``j[m]`` of ray ``ray[m]`` to
        its next, carrying ``power[m]``; a ray that escaped ends in an open
        piece, its second point NaN. Pieces come point by point, rays in order.
        """

        n, points = len(self), self.trajectory.shape[1]
        has = np.isfinite(self.trajectory[:, :, 0])
        closed = np.zeros((n, points), dtype=bool)
        closed[:, :-1] = has[:, :-1] & has[:, 1:]
        open_ = np.zeros((n, points), dtype=bool)
        escaped = self.status == Status.ESCAPED
        open_[np.flatnonzero(escaped), self._last[escaped]] = True
        j, ray = np.nonzero((closed | open_).T)
        ends = np.full((ray.size, 3), np.nan)
        shut = closed[ray, j]
        ends[shut] = self.trajectory[ray[shut], j[shut] + 1]
        pieces = np.stack([self.trajectory[ray, j], ends], axis=1)
        return pieces, self._power[ray, j], ray, j

    # -- a table -----------------------------------------------------------------------

    def __len__(self) -> int:
        return self.trajectory.shape[0]

    def __getitem__(self, rows) -> "Rays":
        """The rays in *rows* (a mask, indices or a slice); ``parent`` keeps naming rows of the whole."""

        if isinstance(rows, (int, np.integer)):
            rows = [rows]
        return Rays._of(**{name: value[rows] for name, value in self.__dict__.items()})

    def __repr__(self) -> str:
        counts = {s.name.lower(): int((self.status == s).sum()) for s in Status if (self.status == s).any()}
        return f"Rays({len(self)}: {', '.join(f'{k} {v}' for k, v in counts.items())})"

    @staticmethod
    def concatenate(batches: "list[Rays]") -> "Rays":
        """One table of the rays of every batch; parents are renumbered to match."""

        batches = [b for b in batches if len(b)] or batches[:1]
        points = max(b.trajectory.shape[1] for b in batches)
        records = [b._record(points) for b in batches]
        offsets = np.cumsum([0] + [len(b) for b in batches[:-1]])
        for record, offset in zip(records, offsets):
            record["parent"] = np.where(record["parent"] >= 0, record["parent"] + offset, -1)
        return Rays._of(**{name: np.concatenate([r[name] for r in records]) for name in records[0]})

    def stated(self, wavelength_um: float | None) -> "Rays":
        """These rays, those with no wavelength given *wavelength_um* (unchanged when it is None)."""

        if wavelength_um is None or not np.isnan(self.wavelength_um).any():
            return self
        record = self._record()
        record["wavelength_um"] = np.where(np.isnan(self.wavelength_um), float(wavelength_um), self.wavelength_um)
        return Rays._of(**record)


__all__ = ["Rays", "Status"]
