"""Shared helper: find the air gap that makes a two-element system afocal.

An afocal (telescope) system has an effective focal length of infinity —
equivalently, the ``C`` element of its first-vertex-to-image ABCD matrix
(:class:`raytracer.propagation.ParaxialModel`) is exactly zero. Rather than
trust the thin-lens approximation for the objective/eyepiece separation,
this solves for the gap numerically so the afocal condition holds for the
*actual* finite-thickness system.
"""

from __future__ import annotations

from typing import Callable

from raytracer.design import OpticalSystem
from raytracer.propagation import ParaxialModel


def solve_afocal_gap(
    build_system: Callable[[float], OpticalSystem], low: float, high: float, *, tol: float = 1e-9
) -> tuple[float, OpticalSystem]:
    """Bisect for the air gap in ``[low, high]`` where ``C(gap) == 0``."""

    def c_of_gap(gap: float) -> float:
        return ParaxialModel(build_system(gap)).matrix_first_vertex_to_image[1, 0]

    lo, hi = low, high
    c_lo, c_hi = c_of_gap(lo), c_of_gap(hi)
    if c_lo * c_hi > 0.0:
        raise ValueError("C(gap) does not change sign on [low, high]; widen the bracket")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c_mid = c_of_gap(mid)
        if abs(c_mid) < tol or (hi - lo) < 1e-9:
            return mid, build_system(mid)
        if (c_mid > 0.0) == (c_lo > 0.0):
            lo, c_lo = mid, c_mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, build_system(mid)


__all__ = ["solve_afocal_gap"]
