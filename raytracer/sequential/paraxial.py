"""Paraxial layer: conjugate recovery and ABCD system matrices.

Two independent methods are provided and cross-checked in the tests:

- :func:`differential_conjugates` probes the *exact* tracer with differential
  rays (the reference notebook's method) — it works for any system the
  tracer can handle, including folded catadioptrics.
- :class:`ParaxialModel` builds the analytic ABCD matrix using the
  signed-index convention (index and thickness change sign on reflection).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surfaces import SurfaceKind
from .system import OpticalSystem
from .trace import SequentialTracer


def direction_from_slopes(sx: float, sy: float) -> np.ndarray:
    """Unit direction from transverse slopes (dx/dz, dy/dz)."""

    return np.array([sx, sy, 1.0]) / np.sqrt(1.0 + sx * sx + sy * sy)


@dataclass(frozen=True)
class ConjugateSolution:
    object_z: float
    magnification: float


def differential_conjugates(
    tracer: SequentialTracer,
    *,
    eps: float = 1e-4,
    start_z: float = -1e-6,
) -> ConjugateSolution:
    """Recover the object plane imposing B = 0 at the tabulated image plane.

    Differential height and angle rays are traced from just before the first
    vertex; the object distance follows from the ray-transfer coefficients.
    """

    base = np.array([0.0, 0.0, start_z])
    height = tracer.trace(base + [0.0, eps, 0.0], direction_from_slopes(0.0, 0.0), keep_path=False)
    angle = tracer.trace(base, direction_from_slopes(0.0, eps), keep_path=False)
    if not (height.ok and angle.ok):
        raise RuntimeError("Differential rays did not survive the system")
    a = height.image_point[1] / eps
    b = angle.image_point[1] / eps
    distance = -b / a
    return ConjugateSolution(object_z=float(-distance), magnification=float(a))


def solve_object_plane(tracer: SequentialTracer, *, assign: bool = True) -> ConjugateSolution:
    """Convenience wrapper that stores the recovered plane on the system."""

    solution = differential_conjugates(tracer)
    if assign:
        tracer.system.object_z = solution.object_z
    return solution


class ParaxialModel:
    """Analytic ABCD model with the signed-index convention for mirrors."""

    def __init__(self, system: OpticalSystem) -> None:
        self.system = system
        self._build()

    def _build(self) -> None:
        system = self.system
        matrix = np.eye(2)
        sign = 1.0
        n1 = system.n_before[0]
        for i, row in enumerate(system.rows):
            n1_signed = sign * system.n_before[i]
            if row.kind is SurfaceKind.MIRROR:
                sign = -sign
            n2_signed = sign * system.n_after[i]
            c = row.profile.curvature
            power = c * (n2_signed - n1_signed)
            refraction = np.array([[1.0, 0.0], [-power, 1.0]])
            matrix = refraction @ matrix
            translation = np.array([[1.0, row.thickness / n2_signed], [0.0, 1.0]])
            matrix = translation @ matrix
        # State is (y, n*u); the final translation lands on the image plane.
        self.matrix_first_vertex_to_image = matrix
        self.n_object = float(n1)

    def solve_object_plane(self) -> ConjugateSolution:
        """Impose B = 0 at the image plane; returns object z and magnification."""

        A, B = self.matrix_first_vertex_to_image[0]
        # Adding an object-space translation L: B' = A*L/n_obj + B = 0.
        L = -B * self.n_object / A
        return ConjugateSolution(object_z=float(-L), magnification=float(A))

    def effective_focal_length(self) -> float:
        C = self.matrix_first_vertex_to_image[1, 0]
        if C == 0.0:
            return np.inf
        return float(-1.0 / C)


__all__ = [
    "direction_from_slopes",
    "ConjugateSolution",
    "differential_conjugates",
    "solve_object_plane",
    "ParaxialModel",
]
