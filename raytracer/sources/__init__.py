"""Sources: emitters that start light for non-sequential tracing.

A source is a description of emission — where light starts, in which
directions, at which wavelength and with how much power. It produces a
batch of :class:`Rays`; carrying them through a system is the tracer's job.

Every source can emit a full 3-D bundle or, with ``fan="y"`` (or ``"x"``),
a flat fan in the plane containing its direction and that axis — the
meridional picture the 2-D figures and the interactive viewer use.
Angles are in degrees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..geometry.transforms import Rotation
from ..geometry.vectors import as_vector, normalize


@dataclass
class Rays:
    """A batch of rays, a row each: origin and unit direction ``(N, 3)``, wavelength (µm) and power ``(N,)``."""

    origin: np.ndarray
    direction: np.ndarray
    wavelength_um: np.ndarray
    power: np.ndarray

    def __post_init__(self) -> None:
        self.origin = np.atleast_2d(np.asarray(self.origin, dtype=float))
        d = np.atleast_2d(np.asarray(self.direction, dtype=float))
        self.direction = d / np.linalg.norm(d, axis=1, keepdims=True)
        n = self.origin.shape[0]
        self.wavelength_um = np.broadcast_to(np.asarray(self.wavelength_um, dtype=float), (n,)).copy()
        self.power = np.broadcast_to(np.asarray(self.power, dtype=float), (n,)).copy()

    def __len__(self) -> int:
        return self.origin.shape[0]

    @staticmethod
    def concatenate(batches: "list[Rays]") -> "Rays":
        return Rays(
            np.vstack([b.origin for b in batches]),
            np.vstack([b.direction for b in batches]),
            np.concatenate([b.wavelength_um for b in batches]),
            np.concatenate([b.power for b in batches]),
        )


def _wavelength(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _cone(direction: np.ndarray, half_angle_deg: float, rays: int, fan: str | None) -> np.ndarray:
    """Unit directions filling a cone (or a fan) of *half_angle_deg* about *direction*."""

    frame = Rotation.align_z(direction)
    half = np.radians(half_angle_deg)
    if fan is not None:
        angles = np.linspace(-half, half, rays) if rays > 1 else np.zeros(1)
        local = np.zeros((angles.size, 3))
        local[:, 2] = np.cos(angles)
        local[:, 1 if fan == "y" else 0] = np.sin(angles)
    else:
        rings = max(1, int(round(np.sqrt(max(rays - 1, 0) / 3.0))))
        local = [(0.0, 0.0, 1.0)]
        for k in range(1, rings + 1):
            theta = half * k / rings
            for phi in np.linspace(0.0, 2 * np.pi, 6 * k, endpoint=False):
                local.append((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
        local = np.asarray(local)
    return frame.apply(local)


def _disk(radius: float, rays: int, fan: str | None) -> np.ndarray:
    """Points ``(N, 3)`` on a disk of *radius* in the local x-y plane (or a line across it)."""

    if fan is not None:
        s = np.linspace(-radius, radius, rays) if rays > 1 else np.zeros(1)
        pts = np.zeros((s.size, 3))
        pts[:, 1 if fan == "y" else 0] = s
        return pts
    rings = max(1, int(round(np.sqrt(max(rays - 1, 0) / 3.0))))
    pts = [(0.0, 0.0, 0.0)]
    for k in range(1, rings + 1):
        r = radius * k / rings
        for phi in np.linspace(0.0, 2 * np.pi, 6 * k, endpoint=False):
            pts.append((r * np.cos(phi), r * np.sin(phi), 0.0))
    return np.asarray(pts)


def _check_fan(fan: str | None) -> str | None:
    if fan not in (None, "x", "y"):
        raise ValueError("fan is None (a 3-D bundle), 'y' or 'x'")
    return fan


class Source(ABC):
    """An emitter of rays."""

    @abstractmethod
    def emit(self) -> Rays:
        """The rays this source starts."""


class PointSource(Source):
    """Rays from one point, filling a cone of half-angle *half_angle_deg* about *direction*."""

    def __init__(
        self,
        position: ArrayLike,
        *,
        direction: ArrayLike = (0.0, 0.0, 1.0),
        half_angle_deg: float,
        rays: int = 21,
        fan: str | None = None,
        wavelength_um: float | None = None,
        power: float = 1.0,
    ) -> None:
        self.position = as_vector(position, dim=3)
        self.direction = normalize(as_vector(direction, dim=3))
        self.half_angle_deg = float(half_angle_deg)
        self.rays = int(rays)
        self.fan = _check_fan(fan)
        self.wavelength_um = wavelength_um
        self.power = float(power)

    def emit(self) -> Rays:
        d = _cone(self.direction, self.half_angle_deg, self.rays, self.fan)
        n = d.shape[0]
        return Rays(np.repeat(self.position[None, :], n, axis=0), d, _wavelength(self.wavelength_um), self.power / n)


class CollimatedSource(Source):
    """A parallel beam of radius *beam_radius*, centred on *origin*, travelling along *direction*."""

    def __init__(
        self,
        origin: ArrayLike,
        *,
        direction: ArrayLike = (0.0, 0.0, 1.0),
        beam_radius: float,
        rays: int = 21,
        fan: str | None = None,
        wavelength_um: float | None = None,
        power: float = 1.0,
    ) -> None:
        self.origin = as_vector(origin, dim=3)
        self.direction = normalize(as_vector(direction, dim=3))
        self.beam_radius = float(beam_radius)
        self.rays = int(rays)
        self.fan = _check_fan(fan)
        self.wavelength_um = wavelength_um
        self.power = float(power)

    def emit(self) -> Rays:
        frame = Rotation.align_z(self.direction)
        pts = frame.apply(_disk(self.beam_radius, self.rays, self.fan)) + self.origin
        n = pts.shape[0]
        return Rays(pts, np.repeat(self.direction[None, :], n, axis=0), _wavelength(self.wavelength_um), self.power / n)


class ImageSource(Source):
    """An extended object: a 1-D intensity profile along the segment ``p0 → p1``.

    Every sample of *profile* is a point emitter whose fan (in the plane of
    the segment and *direction*) carries that sample's intensity. Tracing it
    through a system and histogramming a sensor reconstructs the geometric
    image of the profile.
    """

    def __init__(
        self,
        profile: ArrayLike,
        *,
        p0: ArrayLike,
        p1: ArrayLike,
        direction: ArrayLike = (0.0, 0.0, 1.0),
        half_angle_deg: float,
        rays_per_point: int = 15,
        threshold: float = 1e-3,
        wavelength_um: float | None = None,
    ) -> None:
        self.profile = np.asarray(profile, dtype=float)
        self.p0 = as_vector(p0, dim=3)
        self.p1 = as_vector(p1, dim=3)
        self.direction = normalize(as_vector(direction, dim=3))
        self.half_angle_deg = float(half_angle_deg)
        self.rays_per_point = int(rays_per_point)
        self.threshold = float(threshold)
        self.wavelength_um = wavelength_um

    @classmethod
    def from_image(cls, path, *, column: int | None = None, pixels: int | None = None, **kwargs) -> "ImageSource":
        """The profile from an image file: the mean over rows (or one *column*), top row last."""

        import matplotlib.image as mpimg

        raw = np.asarray(mpimg.imread(path), dtype=float)
        if raw.ndim == 3:
            raw = raw[..., :3].mean(axis=2)
        if raw.max() > 1.0:
            raw = raw / 255.0
        profile = (raw[:, column] if column is not None else raw.mean(axis=1))[::-1]
        if pixels is not None and pixels != profile.size:
            profile = np.interp(np.linspace(0, 1, pixels), np.linspace(0, 1, profile.size), profile)
        if profile.max() > 0:
            profile = profile / profile.max()
        return cls(profile, **kwargs)

    def emit(self) -> Rays:
        span = self.p1 - self.p0
        # The fan lies in the plane of the segment and the emission direction.
        across = span - (span @ self.direction) * self.direction
        fan_axis = normalize(across) if np.linalg.norm(across) > 0 else np.array([0.0, 1.0, 0.0])
        half = np.radians(self.half_angle_deg)
        angles = np.linspace(-half, half, self.rays_per_point)
        dirs = np.cos(angles)[:, None] * self.direction + np.sin(angles)[:, None] * fan_axis
        origins, directions, power = [], [], []
        for fraction, intensity in zip(np.linspace(0.0, 1.0, self.profile.size), self.profile):
            if intensity <= self.threshold:
                continue
            origins.append(np.repeat((self.p0 + fraction * span)[None, :], len(angles), axis=0))
            directions.append(dirs)
            power.append(np.full(len(angles), intensity / len(angles)))
        if not origins:
            return Rays(np.zeros((0, 3)), np.zeros((0, 3)) + [0, 0, 1], np.nan, 0.0)
        return Rays(np.vstack(origins), np.vstack(directions), _wavelength(self.wavelength_um), np.concatenate(power))


__all__ = ["Rays", "Source", "PointSource", "CollimatedSource", "ImageSource"]
