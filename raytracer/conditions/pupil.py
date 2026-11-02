"""Samplings of the pupil: which normalized pupil points to trace, and the share of the pupil each stands for.

- :func:`hexapolar` — rings of 6, 12, 18, … points, each standing for an equal area;
- :func:`polar` — rings of a fixed number of arms;
- :func:`grid` — a square grid clipped to the unit disk;
- :func:`fan` — a line across the pupil, tangential (along y) or sagittal (along x);
- :func:`random` — uniform in the disk, from an explicit seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sampling:
    """A sampling of the pupil: :meth:`points` ``(N, 2)`` in normalized coordinates, :meth:`weights` summing to 1."""

    kind: str
    n: int
    arms: int = 0
    direction: str = ""
    seed: int | None = None

    def points(self) -> np.ndarray:
        if self.kind == "hexapolar":
            rings = [np.zeros((1, 2))] + [_ring(k / self.n, 6 * k) for k in range(1, self.n + 1)]
            return np.vstack(rings)
        if self.kind == "polar":
            return np.vstack([np.zeros((1, 2))] + [_ring(k / self.n, self.arms) for k in range(1, self.n + 1)])
        if self.kind == "grid":
            axis = np.linspace(-1.0, 1.0, self.n)
            x, y = np.meshgrid(axis, axis)
            inside = x**2 + y**2 <= 1.0 + 1e-12
            return np.column_stack([x[inside], y[inside]])
        if self.kind == "fan":
            line = np.linspace(-1.0, 1.0, self.n)
            zero = np.zeros_like(line)
            return np.column_stack([zero, line] if self.direction == "tangential" else [line, zero])
        rng = np.random.default_rng(self.seed)
        r, phi = np.sqrt(rng.random(self.n)), 2.0 * np.pi * rng.random(self.n)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi)])

    def weights(self) -> np.ndarray:
        """The share of the pupil's area each point stands for (equal along a fan)."""

        if self.kind in ("hexapolar", "polar"):
            per_ring = 6 * np.arange(1, self.n + 1) if self.kind == "hexapolar" else np.full(self.n, self.arms)
            # Ring k stands for the annulus from (k - 1/2)/n to (k + 1/2)/n, the last one to the rim.
            edges = np.concatenate([[0.0], (np.arange(1, self.n + 1) - 0.5) / self.n, [1.0]])
            areas = np.diff(edges**2)
            weights = np.concatenate([[areas[0]]] + [np.full(m, a / m) for m, a in zip(per_ring, areas[1:])])
        else:
            weights = np.ones(len(self.points()))
        return weights / weights.sum()


def _ring(radius: float, count: int) -> np.ndarray:
    phi = 2.0 * np.pi * np.arange(count) / count
    return radius * np.column_stack([np.cos(phi), np.sin(phi)])


def hexapolar(rings: int = 6) -> Sampling:
    """The centre and *rings* rings of 6, 12, 18, … points: every point stands for an equal area."""

    return Sampling("hexapolar", _count(rings, "rings"))


def polar(rings: int = 6, arms: int = 24) -> Sampling:
    """The centre and *rings* rings of *arms* points each."""

    return Sampling("polar", _count(rings, "rings"), arms=_count(arms, "arms"))


def grid(n: int = 21) -> Sampling:
    """An *n* × *n* grid across the pupil, clipped to the unit disk."""

    return Sampling("grid", _count(n, "a grid side"))


def fan(direction: str = "tangential", n: int = 41) -> Sampling:
    """*n* points across the pupil: ``"tangential"`` along y, ``"sagittal"`` along x."""

    if direction not in ("tangential", "sagittal"):
        raise ValueError(f"a fan is 'tangential' or 'sagittal', got {direction!r}")
    return Sampling("fan", _count(n, "a fan's points"), direction=direction)


def random(n: int, *, seed: int) -> Sampling:
    """*n* points uniform in the disk, drawn from *seed* — the same seed, the same points."""

    return Sampling("random", _count(n, "random points"), seed=int(seed))


def _count(value: int, what: str) -> int:
    if int(value) != value or int(value) < 1:
        raise ValueError(f"{what}: a positive whole number, got {value!r}")
    return int(value)


__all__ = ["Sampling", "hexapolar", "polar", "grid", "fan", "random"]
