"""Light source primitives that emit batches of rays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("Direction vector cannot be zero.")
    return vec / norm


@dataclass
class RaySeed:
    """Pack origin/direction parameters before the Ray object is created."""

    origin: np.ndarray
    direction: np.ndarray
    params: Dict[str, Any] | None = None


class SourceND:
    """Base class for sources operating in N-dimensional space."""

    dimension: int

    def emit(self) -> List[RaySeed]:  # pragma: no cover - abstract
        raise NotImplementedError


class Source2D(SourceND):
    """Base class for 2D sources."""

    dimension = 2


class Source3D(SourceND):
    """Base class for 3D sources."""

    dimension = 3


@dataclass
class PointSource2D(Source2D):
    """Point source emitting rays within an angular aperture."""

    origin: np.ndarray
    axis_direction: np.ndarray
    aperture: float
    samples: int

    def emit(self) -> List[RaySeed]:
        axis = _unit(self.axis_direction)
        alpha0 = np.arctan2(axis[1], axis[0])
        half = self.aperture / 2.0
        thetas = np.linspace(-half, half, self.samples)
        seeds: List[RaySeed] = []
        base_origin = np.asarray(self.origin, dtype=float)
        for theta in thetas:
            angle = alpha0 + theta
            seeds.append(
                RaySeed(
                    origin=base_origin.copy(),
                    direction=np.array([np.cos(angle), np.sin(angle)]),
                    params={"theta": float(angle)},
                )
            )
        return seeds


@dataclass
class ParallelSource2D(Source2D):
    """Generator for a bundle of parallel rays across a finite width."""

    origin: np.ndarray
    direction: np.ndarray
    width: float
    samples: int

    def emit(self) -> List[RaySeed]:
        direction = _unit(self.direction)
        ortho = np.array([-direction[1], direction[0]])
        ortho = _unit(ortho)
        offsets = np.linspace(-self.width / 2.0, self.width / 2.0, self.samples)
        seeds: List[RaySeed] = []
        base_origin = np.asarray(self.origin, dtype=float)
        for offset in offsets:
            seeds.append(
                RaySeed(
                    origin=base_origin + offset * ortho,
                    direction=direction.copy(),
                    params={"offset": float(offset)},
                )
            )
        return seeds


def _orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _unit(normal)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(trial, w)) > 0.999:
        trial = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(trial, w))
    v = np.cross(w, u)
    return u, v, w


@dataclass
class PointSource3D(Source3D):
    """Point source emitting a spherical-cap distribution of rays."""

    origin: np.ndarray
    axis_direction: np.ndarray
    aperture: float
    theta_samples: int
    phi_samples: int

    def emit(self) -> List[RaySeed]:
        base_origin = np.asarray(self.origin, dtype=float)
        u, v, w = _orthonormal_basis(self.axis_direction)

        theta_max = float(self.aperture)
        theta_values = np.linspace(0.0, theta_max, self.theta_samples)
        phi_values = np.linspace(0.0, 2.0 * np.pi, self.phi_samples, endpoint=False)

        seeds: List[RaySeed] = []
        for theta in theta_values:
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for phi in phi_values:
                direction = (
                    sin_theta * np.cos(phi) * u
                    + sin_theta * np.sin(phi) * v
                    + cos_theta * w
                )
                seeds.append(
                    RaySeed(
                        origin=base_origin.copy(),
                        direction=_unit(direction),
                        params={"theta": float(theta), "phi": float(phi)},
                    )
                )

        if not seeds:
            direction = _unit(w)
            seeds.append(
                RaySeed(
                    origin=base_origin.copy(),
                    direction=direction,
                    params={"theta": 0.0, "phi": 0.0},
                )
            )

        return seeds
