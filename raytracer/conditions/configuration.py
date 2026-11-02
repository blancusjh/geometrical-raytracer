"""A system under conditions: the subject of every analysis."""

from __future__ import annotations

from typing import Any

from ..elements import Stop
from ..geometry.apertures import Circle
from ..system import DETECT, OpticalSystem
from .use import Conditions


class Configuration:
    """An optical system under conditions of use: the subject of every analysis.

    Built by ``system.under(conditions)``. What the two say must agree, and
    a contradiction between them is rejected here, naming both sides: an
    aperture taken from a stop the system lacks, a medium with no index at
    a wavelength of use. A value: :meth:`replace` returns another.
    """

    def __init__(self, system: OpticalSystem, conditions: Conditions) -> None:
        if not isinstance(system, OpticalSystem):
            raise TypeError(f"a configuration is of an OpticalSystem, got {system!r}")
        if not isinstance(conditions, Conditions):
            raise TypeError(f"a configuration is under Conditions, got {conditions!r}")
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "conditions", conditions)
        self._check()

    def _check(self) -> None:
        system, conditions = self.system, self.conditions
        path = system.path
        if not path or path[-1].event != DETECT:
            raise ValueError(f"{system!r}: the path ends on a detector, where the image is formed")
        stop = system.stop
        if stop is None:
            raise ValueError(f"{system!r} has no aperture stop, and the pupils and the aiming are defined by it: "
                             "add a Stop, or name one with stop=")
        if conditions.aperture.kind == "stop":
            element = system[stop]
            opening = getattr(element, "aperture", None)
            if not (isinstance(element, Stop) and isinstance(opening, Circle) and not any(opening.center)):
                raise ValueError(f"the aperture is the stop's own opening, but the stop {stop!r} has no centred "
                                 "circular opening")
        for face in system.faces:
            for medium in (face.medium_minus, face.medium_plus):
                for wavelength in conditions.spectrum.wavelengths:
                    try:
                        medium.index(wavelength)
                    except ValueError as exc:
                        raise ValueError(f"the spectrum includes {wavelength} µm, where {medium.name!r} "
                                         f"has no index: {exc}") from None

    def __setattr__(self, name, value):
        raise AttributeError("Configuration is immutable; use .replace(...)")

    def replace(self, **changes: Any) -> "Configuration":
        """Another configuration: this one with ``system=`` or ``conditions=`` changed."""

        unknown = set(changes) - {"system", "conditions"}
        if unknown:
            raise TypeError(f"Configuration has no argument(s) {sorted(unknown)}")
        return Configuration(changes.get("system", self.system), changes.get("conditions", self.conditions))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Configuration) and (self.system, self.conditions) == (other.system, other.conditions)

    def __hash__(self) -> int:
        return hash((self.system, self.conditions))

    def __repr__(self) -> str:
        c = self.conditions
        return (f"Configuration({self.system!r}, object at {c.object.distance:g} mm, {c.aperture.kind} "
                f"{c.aperture.value if c.aperture.value is not None else ''}, {len(c.fields)} fields, "
                f"{len(c.spectrum)} wavelengths)")


__all__ = ["Configuration"]
