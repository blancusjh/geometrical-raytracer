"""Coatings: how much light an interface reflects and transmits.

A coating does not decide *where* light goes — the media on the two sides
and the laws of reflection and refraction do. It decides *how much* goes
each way, per polarisation, as a function of the angle of incidence. A
"mirror" is therefore not a kind of surface: it is any surface carrying a
reflective coating.

Every coating also states what a sequential trace should follow by
default (:attr:`Coating.sequential_event`): ``"transmit"``, ``"reflect"``,
``"absorb"``, or ``None`` when both branches matter and the path must say
which one it takes (a beam splitter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike


class PowerCoefficients(NamedTuple):
    """Power reflectance and transmittance for s and p polarisation, per ray."""

    Rs: np.ndarray
    Rp: np.ndarray
    Ts: np.ndarray
    Tp: np.ndarray

    @property
    def R(self) -> np.ndarray:
        """Reflectance of unpolarised light."""

        return 0.5 * (self.Rs + self.Rp)

    @property
    def T(self) -> np.ndarray:
        """Transmittance of unpolarised light."""

        return 0.5 * (self.Ts + self.Tp)


def _transmitted_cosine(cos_i: np.ndarray, n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """``cos θ_t`` from Snell's law; NaN under total internal reflection."""

    sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i * cos_i)
    return np.where(sin_t2 <= 1.0, np.sqrt(np.maximum(1.0 - sin_t2, 0.0)), np.nan)


class Coating(ABC):
    """What an interface does with the power arriving on it."""

    sequential_event: str | None = "transmit"

    @abstractmethod
    def power(self, cos_i: ArrayLike, n1: ArrayLike, n2: ArrayLike, wavelength_um: float | None = None) -> PowerCoefficients:
        """Coefficients for incidence cosines ``cos_i`` (``|d · n|``) from index ``n1`` into ``n2``."""

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)

    def __hash__(self) -> int:
        return hash((type(self).__name__, tuple(sorted(vars(self).items()))))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


def _arrays(cos_i, n1, n2):
    cos_i = np.abs(np.asarray(cos_i, dtype=float))
    shape = cos_i.shape
    return cos_i, np.broadcast_to(np.asarray(n1, dtype=float), shape), np.broadcast_to(np.asarray(n2, dtype=float), shape)


class Uncoated(Coating):
    """A bare interface: the Fresnel equations, with total internal reflection."""

    sequential_event = "transmit"

    def power(self, cos_i, n1, n2, wavelength_um=None):
        cos_i, n1, n2 = _arrays(cos_i, n1, n2)
        cos_t = _transmitted_cosine(cos_i, n1, n2)
        tir = np.isnan(cos_t)
        ct = np.where(tir, 0.0, cos_t)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = (n1 * cos_i - n2 * ct) / (n1 * cos_i + n2 * ct)
            rp = (n2 * cos_i - n1 * ct) / (n2 * cos_i + n1 * ct)
        Rs = np.where(tir, 1.0, rs * rs)
        Rp = np.where(tir, 1.0, rp * rp)
        return PowerCoefficients(Rs, Rp, 1.0 - Rs, 1.0 - Rp)


class AR(Coating):
    """Ideal anti-reflection coating: everything transmitted (except under TIR)."""

    sequential_event = "transmit"

    def power(self, cos_i, n1, n2, wavelength_um=None):
        cos_i, n1, n2 = _arrays(cos_i, n1, n2)
        tir = np.isnan(_transmitted_cosine(cos_i, n1, n2))
        R = np.where(tir, 1.0, 0.0)
        return PowerCoefficients(R, R, 1.0 - R, 1.0 - R)


class PerfectMirror(Coating):
    """Reflects everything, on either side."""

    sequential_event = "reflect"

    def power(self, cos_i, n1, n2, wavelength_um=None):
        one = np.ones_like(np.asarray(cos_i, dtype=float))
        return PowerCoefficients(one, one, 0.0 * one, 0.0 * one)


class IdealBeamSplitter(Coating):
    """Reflects ``R`` and transmits ``T`` of the power, independent of angle and polarisation.

    ``R + T ≤ 1``; the remainder is absorbed. A sequential path through it
    must state which branch it follows.
    """

    sequential_event = None

    def __init__(self, R: float = 0.5, T: float = 0.5) -> None:
        if not (0.0 <= R and 0.0 <= T and R + T <= 1.0 + 1e-12):
            raise ValueError("a beam splitter has R, T >= 0 and R + T <= 1")
        self.R, self.T = float(R), float(T)

    def power(self, cos_i, n1, n2, wavelength_um=None):
        one = np.ones_like(np.asarray(cos_i, dtype=float))
        return PowerCoefficients(self.R * one, self.R * one, self.T * one, self.T * one)


class Absorbing(Coating):
    """Absorbs everything: a blackened edge, a diaphragm blade, a detector."""

    sequential_event = "absorb"

    def power(self, cos_i, n1, n2, wavelength_um=None):
        zero = np.zeros_like(np.asarray(cos_i, dtype=float))
        return PowerCoefficients(zero, zero, zero, zero)


__all__ = [
    "Coating",
    "PowerCoefficients",
    "Uncoated",
    "AR",
    "PerfectMirror",
    "IdealBeamSplitter",
    "Absorbing",
]
