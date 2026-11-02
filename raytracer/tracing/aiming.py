"""Ray aiming, and tracing a configuration: from field, pupil and wavelength to real rays.

Every ray of a configuration leaves its field — an object point, or, for an
object at infinity, a direction — and crosses the stop at the point its
pupil coordinates name: ``p · r``, where ``r`` is how far the aperture
fills the stop (:attr:`~raytracer.tracing.paraxial.Paraxial.stop_radius`).
Real aiming solves for that ray exactly, by Newton's method on all rays at
once; paraxial aiming takes the ray paraxial optics sends there, which is
faster and inexact where the pupil aberrates.

Rays from infinity start on one plane wavefront, square to their
direction, so their optical paths compare; rays from a virtual object
start in front of the system, on their line back from it, their optical
path counted from the object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conditions import Configuration, Field, Pupil
from ..conditions.pupil import Sampling, hexapolar
from ..rays import Rays
from .paraxial import Paraxial
from .sequential import SequentialTracer


class _Launch:
    """Where one field's rays come from: an object point, or a direction from infinity."""

    def __init__(self, paraxial: Paraxial, object_side: tuple[float, float]) -> None:
        self.paraxial = paraxial
        conditions = paraxial.configuration.conditions
        z0 = paraxial.z[0]
        self.n0 = paraxial.n_object
        face = paraxial.steps[0].face
        reach = face.extent.extent if face.extent is not None else 10.0 * paraxial.stop_radius
        self.z_launch = z0 - 1.0 - 2.0 * reach  # in front of anything the light meets first
        if conditions.object.infinite:
            tx, ty = np.tan(np.radians(object_side))
            self.direction = np.array([tx, ty, 1.0]) / np.sqrt(1.0 + tx * tx + ty * ty)
            self.origin = None
            e1 = np.cross([0.0, 1.0, 0.0], self.direction)
            self.e1 = e1 / np.linalg.norm(e1)
            self.e2 = np.cross(self.direction, self.e1)
            self.anchor = np.array([0.0, 0.0, self.z_launch])
        else:
            if conditions.fields.kind == "angle":
                z_ep = paraxial.entrance_pupil.z_mm  # the chief ray crosses the axis there at the stated angle
                xy = -(z_ep - paraxial.object_z) * np.tan(np.radians(object_side))
            else:
                xy = np.asarray(object_side, dtype=float)
            self.origin = np.array([xy[0], xy[1], paraxial.object_z])

    @property
    def infinite(self) -> bool:
        return self.origin is None

    def rays(self, u: np.ndarray, wavelength_um: float, power) -> Rays:
        """The rays of launch parameters *u* ``(N, 2)``: wavefront coordinates from infinity, slopes from a point."""

        if self.infinite:
            start = self.anchor + u[:, :1] * self.e1 + u[:, 1:] * self.e2
            return Rays(start, np.tile(self.direction, (len(u), 1)), wavelength_um, power)
        d = np.column_stack([u, np.ones(len(u))])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        start, opl = np.tile(self.origin, (len(u), 1)), 0.0
        if self.origin[2] > self.z_launch:  # a virtual object: start in front, on the ray's line back from it
            back = (self.origin[2] - self.z_launch) / d[:, 2]
            start, opl = start - back[:, None] * d, -self.n0 * back
        return Rays(start, d, wavelength_um, power, opl)

    def paraxial_aim(self, targets: np.ndarray) -> np.ndarray:
        """The launch parameters paraxial optics sends to the stop points *targets*."""

        p = self.paraxial
        if self.infinite:
            A, B = p.stop_heights
            if abs(A) < 1e-12:
                raise ValueError("the entrance pupil is at infinity: a beam from infinity cannot be aimed through it")
            w0 = self.n0 * self.direction[:2] / self.direction[2]
            heights = (targets - B * w0) / A  # where each ray crosses the first vertex's plane
            points = np.column_stack([heights, np.full(len(targets), p.z[0])])
            return np.column_stack([(points - self.anchor) @ self.e1, (points - self.anchor) @ self.e2])
        P = p.trace(1.0, 0.0, p.object_z, upto=p.stop)[0][-1]
        Q = p.trace(0.0, 1.0, p.object_z, upto=p.stop)[0][-1]
        return (targets - P * self.origin[:2]) / Q / self.n0


def _stop_tracer(configuration: Configuration, wavelength_um: float) -> SequentialTracer:
    """A tracer that runs no further than the stop and clips nothing: for finding where rays cross it."""

    system = configuration.system
    steps = system.path
    to_stop = system.replace(path=list(steps[: system.stop_index + 1]) + [steps[-1]])
    return SequentialTracer(to_stop, wavelength_um=wavelength_um, clip_apertures=False)


def aim(configuration: Configuration, field, pupil_points, wavelength_um: float, power=1.0,
        *, tolerance: float = 1e-10, iterations: int = 30) -> Rays:
    """Rays of one *field* at one wavelength, crossing the stop at the normalized *pupil_points* ``(N, 2)``.

    *field* is a declared field's index, or a :class:`~raytracer.conditions.Field`. The rays are aimed as the
    conditions say, and are ready to trace.
    """

    paraxial = Paraxial(configuration)
    source = _Launch(paraxial, object_field(configuration, field, paraxial))
    points = np.atleast_2d(np.asarray(pupil_points, dtype=float))
    targets = points * paraxial.stop_radius
    u = source.paraxial_aim(targets)
    if configuration.conditions.aiming.kind == "real":
        u = _solve(configuration, source, u, targets, wavelength_um, tolerance, iterations)
    return source.rays(u, wavelength_um, power)


def _solve(configuration, source, u, targets, wavelength_um, tolerance, iterations) -> np.ndarray:
    """Newton on every ray at once: the launch parameters that cross the stop at *targets*."""

    tracer = _stop_tracer(configuration, wavelength_um)
    stop = configuration.system.stop_index
    pose = tracer.steps[stop].face.pose
    scale = max(source.paraxial.stop_radius, 1.0)
    h = 1e-6 * (scale if source.infinite else 1.0)  # a hair of wavefront, or of slope
    n = len(u)

    def crossings(params):
        traced = tracer.trace(source.rays(params, wavelength_um, 1.0))
        return pose.to_local(traced.trajectory[:, stop + 1])[:, :2]

    for _ in range(iterations):
        probes = np.vstack([u, u + [h, 0.0], u + [0.0, h]])
        at = crossings(probes)
        r = at[:n] - targets
        if np.nanmax(np.abs(r), initial=0.0) < tolerance * scale:
            break
        jacobian = np.stack([(at[n : 2 * n] - at[:n]) / h, (at[2 * n :] - at[:n]) / h], axis=2)
        ok = np.isfinite(jacobian).all(axis=(1, 2)) & np.isfinite(r).all(axis=1)
        ok[ok] &= np.linalg.det(jacobian[ok]) != 0.0  # rays that cannot reach the stop stay as they are
        step = np.zeros_like(u)
        step[ok] = np.linalg.solve(jacobian[ok], r[ok][:, :, None])[:, :, 0]
        u = u - step
    return u


def object_field(configuration: Configuration, field, paraxial: Paraxial | None = None) -> tuple[float, float]:
    """The object-side field — angles (degrees) from infinity, else object heights (mm) — that *field* names.

    Image heights are reached by solving the chief ray onto the detector: with real rays, unless the
    fields say paraxial.
    """

    conditions = configuration.conditions
    fields = conditions.fields
    xy = fields.absolute(field)
    if fields.kind != "image_height":
        return xy
    paraxial = paraxial or Paraxial(configuration)
    infinite = conditions.object.infinite
    # Paraxially the image height is linear in the object-side field (in tan of the angle from infinity).
    unit = (0.0, 45.0) if infinite else (0.0, 1.0)
    ys, ws = paraxial.chief_of(unit)
    k = len(paraxial.z) - 1
    slope = ys[k]  # image height on the detector per unit object field

    def to_field(t):
        return tuple(np.degrees(np.arctan(t))) if infinite else tuple(t)

    t = np.asarray(xy) / slope
    if not fields.real or not np.any(xy):
        return to_field(t)
    chief = _stated_object_side(configuration)
    for _ in range(20):
        landing = _chief_landing(chief, to_field(t))
        r = landing - np.asarray(xy)
        if np.max(np.abs(r)) < 1e-10 * max(1.0, np.max(np.abs(xy))):
            break
        d = 1e-6 * max(1.0, np.max(np.abs(t)))
        jx = (_chief_landing(chief, to_field(t + [d, 0.0])) - landing) / d
        jy = (_chief_landing(chief, to_field(t + [0.0, d])) - landing) / d
        t = t - np.linalg.solve(np.column_stack([jx, jy]), r)
    return to_field(t)


def _stated_object_side(configuration: Configuration) -> Configuration:
    """The same configuration with its fields restated object-side: what a chief ray is aimed from."""

    conditions = configuration.conditions
    fields = conditions.fields
    kind = "angle" if conditions.object.infinite else "object_height"
    restated = type(fields)(kind, ((0.0, 0.0),), (1.0,))
    return configuration.replace(conditions=conditions.replace(fields=restated))


def _chief_landing(configuration: Configuration, object_side: tuple[float, float]) -> np.ndarray:
    """Where the real chief ray of an object-side field lands on the detector, ``(x, y)`` in its frame."""

    primary = configuration.conditions.spectrum.primary
    ray = aim(configuration, Field(*object_side), [[0.0, 0.0]], primary)
    traced = SequentialTracer(configuration.system, wavelength_um=primary, clip_apertures=False).trace(ray)
    return configuration.system.path[-1].face.pose.to_local(traced.position)[0, :2]


@dataclass(frozen=True)
class Bundle:
    """Rays traced through a configuration, a row per ray, with what each samples.

    ``field`` ``(N, 2)`` is the field point, in the declared fields' unit; ``pupil`` ``(N, 2)`` the normalized
    pupil point. Each ray's wavelength is its own; its power is the share of the pupil it stands for, times
    its wavelength's share of the spectrum.
    """

    configuration: Configuration
    rays: Rays
    field: np.ndarray
    pupil: np.ndarray


def _fields(configuration: Configuration, field) -> list:
    if isinstance(field, str):
        if field != "all":
            raise ValueError(f"a field is 'all', an index or a Field, got {field!r}")
        return list(range(len(configuration.conditions.fields)))
    if isinstance(field, (list, tuple)) and not isinstance(field, Field):
        return [f for item in field for f in _fields(configuration, item)]
    return [field]


def _pupil(pupil) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pupil, Sampling):
        return pupil.points(), pupil.weights()
    if isinstance(pupil, Pupil):
        return np.array([[pupil.px, pupil.py]]), np.ones(1)
    points = np.atleast_2d(np.asarray(pupil, dtype=float))
    return points, np.full(len(points), 1.0 / len(points))


def trace(configuration: Configuration, *, field="all", pupil=None, wavelength="all") -> Bundle:
    """Rays of *configuration*: from each field, at each wavelength, through each pupil point — aimed, then traced.

    *field* is ``"all"`` (the declared fields), an index, a :class:`~raytracer.conditions.Field`, or a list;
    *wavelength* is ``"all"``, ``"primary"``, an index, a value (µm) or a list; *pupil* is a sampling
    (:mod:`raytracer.pupil`; hexapolar with 6 rings by default), a :class:`~raytracer.conditions.Pupil`, or
    points ``(N, 2)``.
    """

    spectrum = configuration.conditions.spectrum
    points, weights = _pupil(hexapolar(6) if pupil is None else pupil)
    wavelengths = spectrum.select(wavelength)
    shares = np.array([spectrum.weights[spectrum.wavelengths.index(w)] if w in spectrum.wavelengths else 1.0
                       for w in wavelengths])
    shares = shares / shares.sum()
    paraxial = Paraxial(configuration)
    batches, field_rows, pupil_rows = [], [], []
    for f in _fields(configuration, field):
        absolute = configuration.conditions.fields.absolute(f)
        for wavelength_um, share in zip(wavelengths, shares):
            batches.append(aim(configuration, f, points, wavelength_um, weights * share))
            field_rows.append(np.tile(absolute, (len(points), 1)))
            pupil_rows.append(points)
    rays = SequentialTracer(configuration.system, wavelength_um=paraxial.wavelength_um).trace(Rays.concatenate(batches))
    return Bundle(configuration, rays, np.vstack(field_rows), np.vstack(pupil_rows))


__all__ = ["aim", "object_field", "trace", "Bundle"]
