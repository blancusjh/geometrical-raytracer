"""Paraxial optics of a coaxial path.

- :class:`Paraxial` is the paraxial optics of a configuration at one
  wavelength: y-nu rays traced step by step with the signed-index
  convention (index and distance change sign on reflection), and what they
  fix — focal lengths and points, principal and nodal points, pupils, the
  filled stop, the marginal and chief rays. :func:`trace` gives those two
  rays.
- :func:`differential_conjugates` probes the *exact* tracer with
  differential rays, and :class:`ParaxialModel` builds the ABCD matrix of
  the path; both recover an object plane for a tracer that has none.

Paraxial optics needs a coaxial path: every boundary centred on, and
normal to, the z axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..conditions import Configuration
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


def _require_coaxial(steps) -> None:
    for step in steps:
        pose = step.face.pose
        if not (np.allclose(pose.axis, (0.0, 0.0, 1.0), atol=1e-12) and np.allclose(pose.position[:2], 0.0)):
            raise ValueError(f"{step.name} is off the z axis; paraxial optics needs a coaxial path")


@dataclass(frozen=True)
class ParaxialRay:
    """A paraxial ray step by step: its height on each vertex plane, and its slope ``u = dy/dz`` leaving it."""

    y: np.ndarray
    u: np.ndarray


@dataclass(frozen=True)
class ParaxialPupil:
    """A paraxial pupil: its axial position ``z_mm`` (world; infinite when telecentric) and diameter."""

    z_mm: float
    diameter_mm: float


@dataclass(frozen=True)
class ParaxialTrace:
    """The paraxial marginal ray, and the chief ray of the largest declared field, of a configuration."""

    marginal: ParaxialRay
    chief: ParaxialRay
    wavelength_um: float


class Paraxial:
    """The paraxial optics of a configuration at one wavelength (the primary unless told otherwise).

    Rays are carried in reduced form ``(y, w)``: the height on each vertex
    plane, and ``w = N u`` — the slope ``u = dy/dz`` times the index of the
    medium, signed by the direction of travel (negative after an odd number
    of reflections). Positions are world ``z`` (mm); distances along the
    light (BFL, FFL) are signed in its direction, and run from the first and
    last boundaries that bend light — not from a stop or a dummy plane.

    The aperture is stated paraxially: it fixes the marginal ray, and with it
    :attr:`stop_radius` — how far real rays fill the stop.
    """

    def __init__(self, configuration: Configuration, wavelength_um: float | None = None) -> None:
        self.configuration = configuration
        conditions = configuration.conditions
        self.wavelength_um = conditions.spectrum.primary if wavelength_um is None else float(wavelength_um)
        system = configuration.system
        steps = system.path
        _require_coaxial(steps)
        self.steps = steps
        self.z = np.array([float(step.face.pose.position[2]) for step in steps])
        self.c = np.array([0.0 if step.face.surface is None else step.face.surface.vertex_curvature[1] for step in steps])
        n_in, n_out, sign = [], [], 1.0
        for step in steps:
            n_minus = float(step.face.medium_minus.index(self.wavelength_um))
            n_plus = float(step.face.medium_plus.index(self.wavelength_um))
            n1 = n_minus if sign > 0 else n_plus
            after = -sign if step.event == REFLECT else sign
            n2 = (n_plus if sign > 0 else n_minus) if step.event == TRANSMIT else n1
            n_in.append(sign * n1)
            n_out.append(after * n2)
            sign = after
        self.n_in, self.n_out = np.array(n_in), np.array(n_out)
        self.stop = system.stop_index
        bending = [i for i, step in enumerate(steps[:-1])
                   if step.event == REFLECT or abs(self.n_in[i]) != abs(self.n_out[i])]
        if not bending:
            raise ValueError("no boundary of the path bends light: there is no first-order optics to speak of")
        self.first, self.last = bending[0], bending[-1]

    # -- rays ---------------------------------------------------------------------

    def trace(self, y, w, z: float, first: int = 0, upto: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Carry rays ``(y, w)`` from the plane *z* through steps *first* … *upto*: heights and ``w`` after each."""

        upto = len(self.z) - 1 if upto is None else upto
        ys, ws = [], []
        for i in range(first, upto + 1):
            y = y + (self.z[i] - z) / self.n_in[i] * w
            z = self.z[i]
            if self.steps[i].event in (TRANSMIT, REFLECT):
                w = w - y * self.c[i] * (self.n_out[i] - self.n_in[i])
            ys.append(y)
            ws.append(w)
        return np.array(ys), np.array(ws)

    def _ray(self, ys: np.ndarray, ws: np.ndarray, first: int = 0) -> ParaxialRay:
        return ParaxialRay(ys, ws / self.n_out[first : first + len(ws)])

    @property
    def object_z(self) -> float:
        """Where the object plane is (``-inf`` at infinity)."""

        return float(self.z[0] - self.configuration.conditions.object.distance)

    # -- the system: first vertex to last ------------------------------------------

    @cached_property
    def matrix(self) -> tuple[float, float, float, float]:
        """``(A, B, C, D)`` of the reduced ray-transfer matrix from the first bending vertex to the last."""

        f = self.first
        ya, wa = self.trace(1.0, 0.0, self.z[f], first=f, upto=self.last)
        yb, wb = self.trace(0.0, 1.0, self.z[f], first=f, upto=self.last)
        return float(ya[-1]), float(yb[-1]), float(wa[-1]), float(wb[-1])

    @property
    def n_object(self) -> float:
        return float(abs(self.n_in[0]))

    @property
    def n_image(self) -> float:
        return float(abs(self.n_out[self.last]))

    @property
    def efl(self) -> float:
        """Effective (equivalent) focal length, the reciprocal of the power."""

        C = self.matrix[2]
        return np.inf if C == 0.0 else -1.0 / C

    @property
    def bfl(self) -> float:
        """From the last vertex to the rear focal point, along the light."""

        A, _, C, _ = self.matrix
        return np.inf if C == 0.0 else -A * self.n_image / C

    @property
    def ffl(self) -> float:
        """From the first bending vertex to the front focal point, along the light (negative: in front of it)."""

        _, _, C, D = self.matrix
        return -np.inf if C == 0.0 else self.n_in[self.first] * D / C

    @property
    def principal_planes(self) -> tuple[float, float]:
        A, _, C, D = self.matrix
        if C == 0.0:
            return np.inf, np.inf
        f, k = self.first, self.last
        return float(self.z[f] + self.n_in[f] * (D - 1.0) / C), float(self.z[k] + self.n_out[k] * (1.0 - A) / C)

    @property
    def nodal_points(self) -> tuple[float, float]:
        A, _, C, D = self.matrix
        if C == 0.0:
            return np.inf, np.inf
        f, k = self.first, self.last
        n0, n1 = self.n_in[f], self.n_out[k]
        return float(self.z[f] + (D * n0 - n1) / C), float(self.z[k] + (n0 - A * n1) / C)

    # -- the stop and its images ---------------------------------------------------------

    @cached_property
    def stop_heights(self) -> tuple[float, float]:
        """``(A, B)``: heights at the stop of the rays ``(y, w) = (1, 0)`` and ``(0, 1)`` from the first vertex."""

        ya, _ = self.trace(1.0, 0.0, self.z[0], upto=self.stop)
        yb, _ = self.trace(0.0, 1.0, self.z[0], upto=self.stop)
        return float(ya[-1]), float(yb[-1])

    @cached_property
    def _unit_marginal(self) -> tuple[np.ndarray, np.ndarray]:
        """The marginal ray at unit size: unit ``w`` from a finite object, unit height from infinity."""

        if self.configuration.conditions.object.infinite:
            return self.trace(1.0, 0.0, self.z[0])
        return self.trace(0.0, 1.0, self.object_z)

    @cached_property
    def stop_radius(self) -> float:
        """How far the aperture fills the stop: the marginal ray's height there (mm)."""

        aperture = self.configuration.conditions.aperture
        if aperture.kind == "stop":
            return float(self.configuration.system[self.configuration.system.stop].aperture.radius)
        ys, ws = self._unit_marginal
        at_stop = abs(ys[self.stop])
        if aperture.kind in ("epd", "f_number"):
            epd = aperture.value if aperture.kind == "epd" else self.efl / aperture.value
            magnification = self.stop_heights[0]
            if abs(magnification) < 1e-12:
                raise ValueError("the entrance pupil is at infinity (telecentric in object space): "
                                 "state the aperture as an NA, or by the stop")
            return 0.5 * epd * abs(magnification)
        if aperture.kind == "object_na":
            return aperture.value * at_stop
        return aperture.value / abs(ws[self.last]) * at_stop  # image_na

    @property
    def object_na(self) -> float:
        """``n |u|`` of the marginal ray in object space (0 for an object at infinity)."""

        if self.configuration.conditions.object.infinite:
            return 0.0
        ys, _ = self._unit_marginal  # it left the object with w = 1
        return float(abs(self.stop_radius / ys[self.stop]))

    @property
    def image_na(self) -> float:
        """``n |u|`` of the marginal ray in image space."""

        return float(abs(self.marginal.u[self.last] * self.n_out[self.last]))

    @property
    def entrance_pupil(self) -> ParaxialPupil:
        """The stop as seen from object space."""

        A, B = self.stop_heights
        if abs(A) < 1e-12:  # telecentric in object space
            return ParaxialPupil(np.inf, np.inf)
        return ParaxialPupil(float(self.z[0] + self.n_in[0] * B / A), 2.0 * self.stop_radius / abs(A))

    @property
    def exit_pupil(self) -> ParaxialPupil:
        """The stop as seen from image space."""

        z_stop, k, n1 = self.z[self.stop], self.last, self.n_out[self.last]
        if self.stop >= k:
            return ParaxialPupil(float(z_stop), 2.0 * self.stop_radius)
        ya, wa = self.trace(1.0, 0.0, z_stop, first=self.stop + 1, upto=k)
        yb, wb = self.trace(0.0, 1.0, z_stop, first=self.stop + 1, upto=k)
        A, B, C, D = ya[-1], yb[-1], wa[-1], wb[-1]
        if abs(D) < 1e-12:  # telecentric in image space
            return ParaxialPupil(np.inf, np.inf)
        z = self.z[k] - B * n1 / D
        return ParaxialPupil(float(z), float(2.0 * self.stop_radius * abs(A + (z - self.z[k]) / n1 * C)))

    # -- marginal and chief rays ---------------------------------------------------------

    @cached_property
    def marginal(self) -> ParaxialRay:
        """The axial ray to the rim of the filled stop, positive there."""

        ys, ws = self._unit_marginal
        scale = self.stop_radius / ys[self.stop]
        return self._ray(ys * scale, ws * scale)

    def chief_of(self, field: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """The paraxial chief ray ``(y, w)`` of an object-side field (its meridional size): an angle, or an object height."""

        conditions = self.configuration.conditions
        kind = conditions.fields.kind
        value = float(np.hypot(*field)) * np.sign(field[1] if field[1] else field[0])
        telecentric = ValueError("the entrance pupil is at infinity (telecentric in object space), so every chief "
                                 "ray is parallel to the axis there: field angles mean nothing; state object heights")
        if conditions.object.infinite:
            w0 = self.n_in[0] * np.tan(np.radians(value))
            A, B = self.stop_heights
            if abs(A) < 1e-12:
                raise telecentric
            return self.trace(-B * w0 / A, w0, self.z[0])
        P = self.trace(1.0, 0.0, self.object_z, upto=self.stop)[0][-1]
        Q = self.trace(0.0, 1.0, self.object_z, upto=self.stop)[0][-1]
        if kind == "angle":
            if abs(P) < 1e-12:
                raise telecentric
            w0 = self.n_in[0] * np.tan(np.radians(value))
            return self.trace(-w0 * Q / P, w0, self.object_z)
        return self.trace(value, -value * P / Q, self.object_z)

    @cached_property
    def chief(self) -> ParaxialRay:
        """The ray of the largest declared field through the centre of the stop."""

        fields = self.configuration.conditions.fields
        largest = max(fields.points, key=lambda p: np.hypot(*p))
        if fields.kind != "image_height":
            return self._ray(*self.chief_of(largest))
        # Image heights: paraxially, the chief ray is linear in them. Unit object field: tan = 1, or 1 mm.
        ys, ws = self.chief_of((0.0, 45.0) if self.configuration.conditions.object.infinite else (0.0, 1.0))
        height = self._at_image(ys, ws)
        scale = float(np.hypot(*largest)) / height
        return self._ray(ys * scale, ws * scale)

    def _at_image(self, ys: np.ndarray, ws: np.ndarray) -> float:
        k = self.last
        return float(ys[k] + (self.image_z - self.z[k]) / self.n_out[k] * ws[k])

    @property
    def image_z(self) -> float:
        """The paraxial image plane, where the marginal ray meets the axis after the last step."""

        ys, ws = self._unit_marginal
        k = self.last
        return np.inf if ws[k] == 0.0 else float(self.z[k] - ys[k] * self.n_out[k] / ws[k])

    @property
    def magnification(self) -> float:
        """Paraxial lateral magnification from object to paraxial image (0 for an object at infinity)."""

        if self.configuration.conditions.object.infinite:
            return 0.0
        ys, ws = self.trace(1.0, 0.0, self.object_z)
        return self._at_image(ys, ws)

    @property
    def lagrange_invariant(self) -> float:
        m, c = self.marginal, self.chief
        return float(c.y[0] * m.u[0] * self.n_out[0] - m.y[0] * c.u[0] * self.n_out[0])

    def ray_trace(self) -> ParaxialTrace:
        return ParaxialTrace(self.marginal, self.chief, self.wavelength_um)


def trace(configuration: Configuration, wavelength_um: float | None = None) -> ParaxialTrace:
    """The paraxial marginal and chief rays of *configuration*, step by step (at the primary wavelength by default)."""

    return Paraxial(configuration, wavelength_um).ray_trace()


__all__ = [
    "Paraxial",
    "ParaxialRay",
    "ParaxialPupil",
    "ParaxialTrace",
    "trace",
    "direction_from_slopes",
    "ConjugateSolution",
    "differential_conjugates",
    "solve_object_plane",
    "ParaxialModel",
]
