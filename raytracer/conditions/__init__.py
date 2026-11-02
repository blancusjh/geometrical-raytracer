"""Conditions: how an optical system is used.

The hardware knows nothing of where its object is, how much light it takes
in, which field it sees or in which colours. Those are the *conditions*,
stated here; a system *under* conditions is a :class:`Configuration`, the
subject of every analysis. Inconsistent conditions are rejected when a
configuration is built, naming both sides.

- :class:`Object` — at infinity, or a finite distance in front of the first
  vertex (negative: a virtual object behind it);
- :class:`SystemAperture` — how much light enters, stated paraxially as in
  lens-design programs: an entrance-pupil diameter, an F-number, an object-
  or image-space NA, or the stop's own opening;
- :class:`Fields` — the declared field points: angles, object heights or
  image heights;
- :class:`Spectrum` — the wavelengths of use, their weights and the primary;
- :class:`Polarization` — the state of the light entering;
- :class:`Aiming` — how pupil coordinates become real rays.

:class:`Field` and :class:`Pupil` are coordinates in that space, and
:mod:`.pupil` holds the pupil samplings.
"""

from .configuration import Configuration
from .coordinates import Field, Pupil
from .use import Aiming, Conditions, Fields, Object, Polarization, Spectrum, SystemAperture
from . import pupil

__all__ = [
    "Object",
    "SystemAperture",
    "Fields",
    "Spectrum",
    "Polarization",
    "Aiming",
    "Conditions",
    "Configuration",
    "Field",
    "Pupil",
    "pupil",
]
