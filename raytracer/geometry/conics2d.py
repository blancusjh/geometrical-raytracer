"""Conic sections in quadratic form: pure 2-D algebra, no engine dependency.

Conics are expressed in focal (polar) form r(theta) = p / (1 + e cos theta)
and intersected through their implicit quadratic form. This module only
solves that algebra (ray/conic intersection, normal direction); turning it
into a traceable surface is an engine concern — see
:mod:`raytracer.nonseq.conics2d` for the ``Surface2D`` adapter
(:class:`~raytracer.nonseq.conics2d.ConicInterface2D` and its
ellipse/circle/parabola/hyperbola shorthands) built on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..math.vectors import normalize


def quadratic_coeffs_from_ep(e: float, p: float) -> tuple[float, float, float, float, float, float]:
    """Return quadratic-form coefficients for a conic in polar form."""

    A: float = 1.0 - e**2
    B: float = 0.0
    C: float = 1.0
    D: float = 2.0 * e * p
    E: float = 0.0
    F: float = -(p**2)

    return (A, B, C, D, E, F)


@dataclass
class ConicProfile:
    """Reusable solver for conic sections expressed in quadratic form."""

    coeffs: Tuple[float, float, float, float, float, float]

    def intersect(
        self, origin: np.ndarray, direction: np.ndarray, eps: float = 1e-12
    ) -> Tuple[np.ndarray, float] | None:
        x0, y0 = origin
        dx, dy = direction
        A, B, C, D, E, F = self.coeffs

        a = A * dx * dx + B * dx * dy + C * dy * dy
        b = 2.0 * A * x0 * dx + B * (x0 * dy + y0 * dx) + 2.0 * C * y0 * dy + D * dx + E * dy
        c = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0 + F

        lam: float | None = None

        if abs(a) < eps:
            if abs(b) >= eps:
                candidate = -c / b
                if candidate >= 0.0:
                    lam = candidate
        else:
            disc = b * b - 4.0 * a * c
            if disc >= -eps:
                disc = max(0.0, disc)
                sqrt_disc = np.sqrt(disc)
                lam1 = (-b - sqrt_disc) / (2.0 * a)
                lam2 = (-b + sqrt_disc) / (2.0 * a)
                candidates = [val for val in (lam1, lam2) if val >= 0.0]
                if candidates:
                    lam = min(candidates)

        if lam is None:
            return None

        point = origin + lam * direction
        return point, float(lam)

    def normal(self, point: np.ndarray) -> np.ndarray:
        x, y = point
        A, B, C, D, E, _ = self.coeffs
        nx = 2.0 * A * x + B * y + D
        ny = B * x + 2.0 * C * y + E
        return normalize(np.array([nx, ny], dtype=float))


__all__ = ["quadratic_coeffs_from_ep", "ConicProfile"]


def __getattr__(name: str):
    # Deprecated: ConicInterface2D and its shorthands are Surface2D adapters
    # (they depend on the non-sequential engine's Ray2D/Intersection2D), not
    # engine-agnostic geometry, so they moved to raytracer.nonseq.conics2d.
    if name in ("ConicInterface2D", "EllipseConic", "CircleConic", "ParabolaConic", "HyperbolaConic"):
        import warnings

        from ..nonseq import conics2d as _nonseq_conics2d

        warnings.warn(
            f"raytracer.geometry.conics2d.{name} is deprecated; import it from "
            "raytracer.nonseq.conics2d instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_nonseq_conics2d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
