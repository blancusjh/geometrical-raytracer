"""Optical analysis, split by what is being measured.

- :mod:`raytracer.analysis.imaging` — what the image looks like: spot
  diagrams, PSF, aerial image, contrast.
- :mod:`raytracer.analysis.aberrations` — what the ray/wavefront error
  looks like and why: field metrics, distortion, Seidel coefficients,
  chromatic aberration, wavefront/Zernike, ray fans, stigmatism.

Every public name from both is re-exported here, so a call site that
doesn't care which side of that split a name lives on can just use
``from raytracer.analysis import ...``. The re-export is derived from the
subpackages' own ``__all__`` rather than hand-listed, so the two cannot
drift apart.
"""

from . import aberrations, imaging
from .aberrations import *  # noqa: F401,F403
from .imaging import *  # noqa: F401,F403

__all__ = [*aberrations.__all__, *imaging.__all__]
