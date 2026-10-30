"""Surfaces with two principal curvatures at the vertex: toroids and cylinders.

A toroidal surface sweeps a meridional curve in the y-z plane — a conic of
vertex radius ``Ry`` and constant ``k`` — around an axis parallel to y,
placed on the z axis at the radius of rotation ``Rx``:

    s(x, y) = Rx − sign(Rx) · √((Rx − s_y(y))² − x²).

A cylinder is the toroid whose radius of rotation is infinite, so
:class:`CylindricalSurface` is a subclass of :class:`ToroidalSurface`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._value import SurfaceValue
from .base import Surface
from .conics import conic_sag_and_slope


def _radius(R: float, name: str) -> float:
    R = float(R)
    if R == 0.0:
        raise ValueError(f"{name} = 0 is degenerate; a flat direction is {name} = inf")
    return R


class ToroidalSurface(SurfaceValue, Surface):
    """Toroid: conic ``(Ry, k)`` in the y-z plane, rotated with radius ``Rx`` about an axis parallel to y."""

    _params = ("Ry", "Rx", "k")

    def __init__(self, Ry: float, Rx: float, k: float = 0.0) -> None:
        self.Ry = _radius(Ry, "Ry")
        self.Rx = _radius(Rx, "Rx")
        self.k = float(k)
        self._freeze()

    def sag_gradient(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cy = 0.0 if np.isinf(self.Ry) else 1.0 / self.Ry
        sy, dsy = conic_sag_and_slope(np.abs(y), cy, self.k)
        dsy = dsy * np.sign(y)
        if np.isinf(self.Rx):
            return sy, np.zeros_like(x), dsy
        R = self.Rx
        w = np.sqrt(np.maximum((R - sy) ** 2 - x * x, 1e-30))
        sign = np.sign(R)
        s = R - sign * w
        dsdx = sign * x / w
        dsdy = sign * (R - sy) * dsy / w
        return s, dsdx, dsdy

    @property
    def vertex_curvature(self) -> tuple[float, float]:
        cx = 0.0 if np.isinf(self.Rx) else 1.0 / self.Rx
        cy = 0.0 if np.isinf(self.Ry) else 1.0 / self.Ry
        return (cx, cy)


class CylindricalSurface(ToroidalSurface):
    """Cylinder curved in one direction: the toroid with infinite radius of rotation.

    ``axis`` names the direction of the generating lines (the flat
    direction): ``"x"`` curves in y, ``"y"`` curves in x. A curved
    cross-section ``(R, k)`` other than a circle makes it a conic cylinder.
    """

    _params = ("R", "k", "axis")

    def __init__(self, R: float, k: float = 0.0, axis: str = "x") -> None:
        if axis not in ("x", "y"):
            raise ValueError("axis is 'x' or 'y': the direction of the generating lines")
        self.R = _radius(R, "R")
        self.axis = axis
        # The toroid with infinite radius of rotation, swept along `axis`.
        self.Ry, self.Rx, self.k = self.R, np.inf, float(k)
        self._freeze()

    def sag_gradient(self, x: ArrayLike, y: ArrayLike):
        if self.axis == "x":
            return super().sag_gradient(x, y)
        s, dsdy, dsdx = super().sag_gradient(y, x)
        return s, dsdx, dsdy

    @property
    def vertex_curvature(self) -> tuple[float, float]:
        c = 1.0 / self.R
        return (0.0, c) if self.axis == "x" else (c, 0.0)


__all__ = ["ToroidalSurface", "CylindricalSurface"]
