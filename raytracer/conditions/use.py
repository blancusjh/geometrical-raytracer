"""The conditions of use, each a value: object, aperture, fields, spectrum, polarisation, aiming."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .coordinates import Field

#: Fraunhofer lines (µm): F (hydrogen blue), d (helium yellow), C (hydrogen red).
FRAUNHOFER = {"F": 0.4861327, "d": 0.5875618, "C": 0.6562725}


def _positive(value, what: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{what} is positive, got {value}")
    return value


@dataclass(frozen=True)
class Object:
    """Where the object is.

    ``distance`` runs along the light from the object to the first vertex
    (mm): positive in front of it, negative for a virtual object behind it,
    infinite for an object at infinity.
    """

    distance: float

    def __post_init__(self) -> None:
        distance = float(self.distance)
        if np.isnan(distance) or distance == -np.inf:
            raise ValueError(f"an object distance is finite or +inf, got {distance}")
        object.__setattr__(self, "distance", distance)

    @classmethod
    def at_infinity(cls) -> "Object":
        return cls(np.inf)

    @classmethod
    def finite(cls, distance: float) -> "Object":
        if not np.isfinite(float(distance)):
            raise ValueError("a finite object has a finite distance; for infinity, Object.at_infinity()")
        return cls(distance)

    @property
    def infinite(self) -> bool:
        return bool(np.isinf(self.distance))


@dataclass(frozen=True)
class SystemAperture:
    """How much light the system takes in: the paraxial marginal ray it fixes.

    Stated paraxially, as lens-design programs do: an entrance-pupil
    diameter (mm), an image-space F-number (EFL / EPD), an object- or
    image-space numerical aperture (``n u`` of the paraxial marginal ray),
    or the opening of the stop itself. Real rays then fill the stop out to
    that ray's height there.
    """

    kind: str  # "epd", "f_number", "object_na", "image_na" or "stop"
    value: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("epd", "f_number", "object_na", "image_na", "stop"):
            raise ValueError(f"unknown kind of system aperture {self.kind!r}")
        if (self.kind == "stop") != (self.value is None):
            raise ValueError("every system aperture but the stop's own opening has a value")

    @classmethod
    def epd(cls, diameter: float) -> "SystemAperture":
        return cls("epd", _positive(diameter, "an entrance-pupil diameter"))

    @classmethod
    def f_number(cls, n: float) -> "SystemAperture":
        return cls("f_number", _positive(n, "an F-number"))

    @classmethod
    def object_na(cls, na: float) -> "SystemAperture":
        return cls("object_na", _positive(na, "a numerical aperture"))

    @classmethod
    def image_na(cls, na: float) -> "SystemAperture":
        return cls("image_na", _positive(na, "a numerical aperture"))

    @classmethod
    def from_stop(cls) -> "SystemAperture":
        return cls("stop")


def _field_points(values) -> tuple[tuple[float, float], ...]:
    """Each value ``y`` or ``(x, y)`` as an ``(x, y)`` pair."""

    points = []
    for value in values:
        pair = np.atleast_1d(np.asarray(value, dtype=float))
        if pair.shape == (1,):
            pair = np.array([0.0, pair[0]])
        if pair.shape != (2,):
            raise ValueError(f"a field point is y or (x, y), got {value!r}")
        points.append((float(pair[0]), float(pair[1])))
    return tuple(points)


@dataclass(frozen=True)
class Fields:
    """The declared field points, all of one kind.

    Angles (degrees) of the chief ray in object space, object heights (mm),
    or image heights (mm), reached by solving the chief ray — with real rays
    unless ``real=False``. Each point is ``(x, y)``; a bare number is
    ``(0, y)``. ``weights`` say how much each point matters (default 1).
    """

    kind: str  # "angle", "object_height" or "image_height"
    points: tuple[tuple[float, float], ...]
    weights: tuple[float, ...]
    real: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("angle", "object_height", "image_height"):
            raise ValueError(f"unknown kind of field {self.kind!r}")
        if not self.points:
            raise ValueError("at least one field point is declared")
        if len(self.weights) != len(self.points):
            raise ValueError(f"{len(self.points)} field points and {len(self.weights)} weights")
        if any(w < 0.0 for w in self.weights):
            raise ValueError("field weights are not negative")
        if self.kind == "angle" and any(abs(c) >= 90.0 for p in self.points for c in p):
            raise ValueError("field angles lie strictly between -90 and 90 degrees")

    @classmethod
    def _of(cls, kind, values, weights, real=True) -> "Fields":
        points = _field_points(values)
        weights = tuple(float(w) for w in (weights if weights is not None else [1.0] * len(points)))
        return cls(kind, points, weights, bool(real))

    @classmethod
    def angles(cls, values, weights=None) -> "Fields":
        return cls._of("angle", values, weights)

    @classmethod
    def object_heights(cls, values, weights=None) -> "Fields":
        return cls._of("object_height", values, weights)

    @classmethod
    def image_heights(cls, values, weights=None, real: bool = True) -> "Fields":
        return cls._of("image_height", values, weights, real)

    @property
    def unit(self) -> str:
        return "deg" if self.kind == "angle" else "mm"

    @property
    def max_radius(self) -> float:
        """The largest declared field, radially: what a normalized field of 1 is."""

        return float(max(np.hypot(*p) for p in self.points))

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> Field:
        return Field(*self.points[index])

    def absolute(self, field: Field | int) -> tuple[float, float]:
        """The ``(x, y)`` of *field* — a declared index, an absolute or a normalized field — in this kind's unit."""

        if isinstance(field, (int, np.integer)):
            return self.points[int(field)]
        if field.relative:
            return field.x * self.max_radius, field.y * self.max_radius
        return field.x, field.y


@dataclass(frozen=True, init=False)
class Spectrum:
    """The wavelengths of use (µm), how much each weighs, and the primary — where first-order optics is read."""

    wavelengths: tuple[float, ...]
    weights: tuple[float, ...]
    primary: float

    def __init__(self, wavelengths, weights=None, primary: float | None = None) -> None:
        wavelengths = tuple(_positive(w, "a wavelength") for w in np.atleast_1d(wavelengths))
        weights = tuple(float(w) for w in (weights if weights is not None else [1.0] * len(wavelengths)))
        if len(weights) != len(wavelengths):
            raise ValueError(f"{len(wavelengths)} wavelengths and {len(weights)} weights")
        if primary is None:
            if len(wavelengths) != 1:
                raise ValueError("a spectrum of several wavelengths names its primary (primary=...)")
            primary = wavelengths[0]
        primary = float(primary)
        if not any(np.isclose(primary, w, rtol=0.0, atol=1e-9) for w in wavelengths):
            raise ValueError(f"the primary wavelength {primary} is not one of {wavelengths}")
        object.__setattr__(self, "wavelengths", wavelengths)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "primary", primary)

    @classmethod
    def single(cls, wavelength: float) -> "Spectrum":
        return cls([wavelength])

    @classmethod
    def fdc(cls, weights=None) -> "Spectrum":
        """The F, d and C lines, d primary: the visible of classical lens design."""

        return cls([FRAUNHOFER["F"], FRAUNHOFER["d"], FRAUNHOFER["C"]], weights, primary=FRAUNHOFER["d"])

    def __len__(self) -> int:
        return len(self.wavelengths)

    def select(self, wavelength) -> tuple[float, ...]:
        """The wavelengths *wavelength* names: ``"all"``, ``"primary"``, an index, a value (µm) or a list of those."""

        if isinstance(wavelength, str):
            if wavelength == "all":
                return self.wavelengths
            if wavelength == "primary":
                return (self.primary,)
            raise ValueError(f"a wavelength is 'all', 'primary', an index or a value in µm, got {wavelength!r}")
        if isinstance(wavelength, (list, tuple)):
            return tuple(w for item in wavelength for w in self.select(item))
        if isinstance(wavelength, (int, np.integer)):
            return (self.wavelengths[int(wavelength)],)
        return (_positive(wavelength, "a wavelength"),)


@dataclass(frozen=True)
class Polarization:
    """The polarisation of the light entering: unpolarised, linear at ``angle`` degrees from x, or circular."""

    kind: str = "unpolarized"
    angle: float = 0.0
    handedness: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("unpolarized", "linear", "circular"):
            raise ValueError(f"unknown polarisation {self.kind!r}")
        if self.kind == "circular" and self.handedness not in ("right", "left"):
            raise ValueError("circular polarisation is right- or left-handed")

    @classmethod
    def unpolarized(cls) -> "Polarization":
        return cls()

    @classmethod
    def linear(cls, angle: float) -> "Polarization":
        return cls("linear", float(angle))

    @classmethod
    def circular(cls, handedness: str = "right") -> "Polarization":
        return cls("circular", 0.0, handedness)


@dataclass(frozen=True)
class Aiming:
    """How a pupil point becomes a ray.

    ``real`` solves each ray to its point on the stop, exactly; ``paraxial``
    aims it at the paraxial entrance pupil — faster, and inexact where the
    pupil aberrates.
    """

    kind: str = "real"

    def __post_init__(self) -> None:
        if self.kind not in ("real", "paraxial"):
            raise ValueError(f"ray aiming is 'real' or 'paraxial', got {self.kind!r}")

    @classmethod
    def real(cls) -> "Aiming":
        return cls("real")

    @classmethod
    def paraxial(cls) -> "Aiming":
        return cls("paraxial")


@dataclass(frozen=True)
class Conditions:
    """How a system is used: where the object is, how much light enters, which fields, in which colours."""

    object: Object
    aperture: SystemAperture
    fields: Fields
    spectrum: Spectrum
    polarization: Polarization = Polarization()
    aiming: Aiming = Aiming()

    def __post_init__(self) -> None:
        for name, kind in (("object", Object), ("aperture", SystemAperture), ("fields", Fields), ("spectrum", Spectrum),
                           ("polarization", Polarization), ("aiming", Aiming)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"conditions: {name} is a {kind.__name__}, got {getattr(self, name)!r}")
        if self.object.infinite and self.fields.kind == "object_height":
            raise ValueError("fields are object heights, but the object is at infinity: give angles or image heights")
        if self.object.infinite and self.aperture.kind == "object_na":
            raise ValueError("the aperture is an object-space NA, but the object is at infinity: "
                             "give an EPD, an F-number or an image-space NA")

    def replace(self, **changes) -> "Conditions":
        return dataclasses.replace(self, **changes)


__all__ = ["Object", "SystemAperture", "Fields", "Spectrum", "Polarization", "Aiming", "Conditions", "FRAUNHOFER"]
