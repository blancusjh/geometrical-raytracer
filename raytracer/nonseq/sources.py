"""Ray sources for 2-D non-sequential scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.vectors import normalize, perpendicular_2d


@dataclass
class RaySeed:
    """Origin/direction pair emitted by a light source."""

    origin: np.ndarray
    direction: np.ndarray
    intensity: float = 1.0
    wavelength_um: Optional[float] = None
    params: Optional[dict] = None


class Source2D:
    """Base class for 2-D ray sources."""

    samples: int

    def emit(self) -> List[RaySeed]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class PointSource2D(Source2D):
    """Point source emitting rays across an angular aperture."""

    origin: np.ndarray
    axis_direction: np.ndarray
    aperture: float
    samples: int
    intensity: float = 1.0
    wavelength_um: Optional[float] = None

    def emit(self) -> List[RaySeed]:
        axis = normalize(self.axis_direction)
        base_angle = float(np.arctan2(axis[1], axis[0]))
        half = float(self.aperture) / 2.0
        thetas = np.linspace(-half, half, self.samples)
        base_origin = np.asarray(self.origin, dtype=float)
        seeds: List[RaySeed] = []
        for offset in thetas:
            angle = base_angle + offset
            seeds.append(
                RaySeed(
                    origin=base_origin.copy(),
                    direction=np.array([np.cos(angle), np.sin(angle)]),
                    intensity=self.intensity,
                    wavelength_um=self.wavelength_um,
                    params={"theta": float(angle)},
                )
            )
        return seeds


@dataclass
class ParallelSource2D(Source2D):
    """Bundle of parallel rays sampled across a finite width."""

    origin: np.ndarray
    direction: np.ndarray
    width: float
    samples: int
    intensity: float = 1.0
    wavelength_um: Optional[float] = None

    def emit(self) -> List[RaySeed]:
        direction = normalize(self.direction)
        ortho = perpendicular_2d(direction)
        offsets = np.linspace(-self.width / 2.0, self.width / 2.0, self.samples)
        base_origin = np.asarray(self.origin, dtype=float)
        seeds: List[RaySeed] = []
        for offset in offsets:
            seeds.append(
                RaySeed(
                    origin=base_origin + offset * ortho,
                    direction=direction.copy(),
                    intensity=self.intensity,
                    wavelength_um=self.wavelength_um,
                    params={"offset": float(offset)},
                )
            )
        return seeds


@dataclass
class ImageSource2D(Source2D):
    """Extended object: a 1-D intensity profile emitting weighted ray fans.

    Each pixel of *profile* becomes a point emitter on the segment
    ``p0 -> p1`` whose rays carry the pixel intensity. Tracing through a
    system and histogramming a :class:`~raytracer.nonseq.detectors.Screen2D`
    reconstructs the (magnified, inverted) geometric image of the profile.
    """

    profile: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    axis_direction: np.ndarray
    aperture: float
    rays_per_point: int = 15
    intensity_threshold: float = 1e-3
    wavelength_um: Optional[float] = None

    @classmethod
    def from_image(
        cls,
        path,
        *,
        p0,
        p1,
        axis_direction,
        aperture: float,
        rays_per_point: int = 15,
        column: int | None = None,
        pixels: int | None = None,
    ) -> "ImageSource2D":
        """Build the profile from an image file (grayscale mean or a column)."""

        import matplotlib.image as mpimg

        raw = mpimg.imread(path)
        if raw.ndim == 3:
            raw = raw[..., :3].mean(axis=2)
        raw = np.asarray(raw, dtype=float)
        if raw.max() > 1.0:
            raw = raw / 255.0
        profile = raw[:, column] if column is not None else raw.mean(axis=1)
        profile = profile[::-1]  # image row 0 is at the top; segment runs p0->p1
        if pixels is not None and pixels != profile.size:
            positions = np.linspace(0.0, 1.0, profile.size)
            wanted = np.linspace(0.0, 1.0, pixels)
            profile = np.interp(wanted, positions, profile)
        peak = profile.max()
        if peak > 0:
            profile = profile / peak
        return cls(
            profile=profile,
            p0=np.asarray(p0, dtype=float),
            p1=np.asarray(p1, dtype=float),
            axis_direction=np.asarray(axis_direction, dtype=float),
            aperture=aperture,
            rays_per_point=rays_per_point,
        )

    @property
    def samples(self) -> int:  # total emitted rays (Source2D contract)
        return int(np.count_nonzero(self.profile > self.intensity_threshold)) * (
            self.rays_per_point
        )

    def emit(self) -> List[RaySeed]:
        profile = np.asarray(self.profile, dtype=float)
        p0 = np.asarray(self.p0, dtype=float)
        p1 = np.asarray(self.p1, dtype=float)
        axis = normalize(self.axis_direction)
        base_angle = float(np.arctan2(axis[1], axis[0]))
        half = float(self.aperture) / 2.0
        thetas = np.linspace(-half, half, self.rays_per_point)

        seeds: List[RaySeed] = []
        positions = np.linspace(0.0, 1.0, profile.size)
        for fraction, intensity in zip(positions, profile):
            if intensity <= self.intensity_threshold:
                continue
            origin = p0 + fraction * (p1 - p0)
            for offset in thetas:
                angle = base_angle + offset
                seeds.append(
                    RaySeed(
                        origin=origin.copy(),
                        direction=np.array([np.cos(angle), np.sin(angle)]),
                        intensity=float(intensity) / self.rays_per_point,
                        wavelength_um=self.wavelength_um,
                        params={"pixel_fraction": float(fraction)},
                    )
                )
        return seeds


__all__ = [
    "RaySeed",
    "Source2D",
    "PointSource2D",
    "ParallelSource2D",
    "ImageSource2D",
]
