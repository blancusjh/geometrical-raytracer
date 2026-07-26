"""Deprecated: moved to :mod:`raytracer.physics.laws`."""

from __future__ import annotations

from ..physics.laws import (  # noqa: F401
    FresnelCoefficients,
    SnellResult,
    fresnel_coefficients,
    reflect,
    refract,
    snell,
)

__all__ = [
    "SnellResult",
    "FresnelCoefficients",
    "snell",
    "reflect",
    "refract",
    "fresnel_coefficients",
]
