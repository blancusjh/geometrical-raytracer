"""Paraxial optics: conjugate recovery and ABCD matrices along a coaxial path.

Two independent methods, cross-checked in the tests:

- :func:`differential_conjugates` probes the *exact* tracer with
  differential rays — it works for anything the tracer can trace,
  including folded catadioptrics;
- :class:`ParaxialModel` builds the analytic ABCD matrix of the path with
  the signed-index convention (index and distance change sign on
  reflection). It needs a coaxial path: every boundary centred on, and
  normal to, the z axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.vectors import direction_from_slopes
from ..rays import Rays
from ..system import DETECT, REFLECT, TRANSMIT
from .sequential import SequentialTracer


@dataclass(frozen=True)
class ConjugateSolution:
    object_z: float
    magnification: float


def differential_conjugates(tracer: SequentialTracer, *, eps: float = 1e-4, start_z: float = -1e-6) -> ConjugateSolution:
    """Recover the object plane imposing B = 0 at the detector.

    Differential height and angle rays are traced from just before the
    first vertex; the object distance follows from the ray-transfer
    coefficients.
    """

    base = np.array([0.0, 0.0, tracer.step_z(0) + start_z])
    height = tracer.trace(Rays(base + [0.0, eps, 0.0], direction_from_slopes(0.0, 0.0)))
    angle = tracer.trace(Rays(base, direction_from_slopes(0.0, eps)))
    if not (height.detected[0] and angle.detected[0]):
        raise RuntimeError("Differential rays did not survive the system")
    a = height.position[0, 1] / eps
    b = angle.position[0, 1] / eps
    distance = -b / a
    return ConjugateSolution(object_z=float(base[2] - distance), magnification=float(a))


def solve_object_plane(tracer: SequentialTracer, *, assign: bool = True) -> ConjugateSolution:
    """Find the object plane conjugate to the detector, and (by default) set it on *tracer*.

    For an all-refractive system focused near infinity the B = 0 condition
    is degenerate: the sign of the recovered ``object_z`` hinges on whether
    the detector sits a fraction of a millimetre before or after the back
    focal plane. The physical convention — a real object in front,
    ``object_z < 0`` — is enforced whenever the raw solution places a
    virtual object far behind the system.
    """

    solution = differential_conjugates(tracer)
    length = tracer.image_z - tracer.step_z(0)
    reflects = any(step.event == REFLECT for step in tracer.steps)
    if not reflects and abs(solution.object_z) > 50 * max(length, 1.0) and solution.object_z > tracer.image_z:
        solution = ConjugateSolution(object_z=-abs(solution.object_z), magnification=-abs(solution.magnification))
    if assign:
        tracer.object_z = solution.object_z
    return solution


class ParaxialModel:
    """Analytic ABCD model of a coaxial path, in the reduced state ``(y, n u)``."""

    def __init__(self, tracer: SequentialTracer) -> None:
        self.tracer = tracer
        self._build()

    def _build(self) -> None:
        tracer = self.tracer
        steps = tracer.steps
        for step in steps:
            pose = step.face.pose
            if not (np.allclose(pose.axis, (0.0, 0.0, 1.0), atol=1e-12) and np.allclose(pose.position[:2], 0.0)):
                raise ValueError(f"{step.name} is off the z axis; a paraxial model needs a coaxial path")
        matrix = np.eye(2)
        sign = 1.0  # +1 travelling toward +z
        n_current = tracer._n[0][0]  # light arrives at the first boundary from its -z side
        self.n_object = float(n_current)
        z_prev = tracer.step_z(0)
        for i, step in enumerate(steps):
            z = tracer.step_z(i)
            matrix = np.array([[1.0, (z - z_prev) / (sign * n_current)], [0.0, 1.0]]) @ matrix
            z_prev = z
            if step.event == DETECT:
                break
            n_minus, n_plus = tracer._n[i]
            n1 = n_minus if sign > 0 else n_plus
            n1_signed = sign * n1
            if step.event == REFLECT:
                sign = -sign
                n2 = n1
            elif step.event == TRANSMIT:
                n2 = n_plus if sign > 0 else n_minus
            else:
                n2 = n1
            power = step.face.surface.vertex_curvature[1] * (sign * n2 - n1_signed)
            matrix = np.array([[1.0, 0.0], [-power, 1.0]]) @ matrix
            n_current = n2
        self.matrix_first_vertex_to_image = matrix

    def solve_object_plane(self) -> ConjugateSolution:
        """Impose B = 0 at the detector; returns object z and magnification."""

        A, B = self.matrix_first_vertex_to_image[0]
        L = -B * self.n_object / A
        return ConjugateSolution(object_z=float(self.tracer.step_z(0) - L), magnification=float(A))

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
