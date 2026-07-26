"""Pure shape math: surface profiles and conic/implicit-form definitions.

Everything here is genuinely just geometry — curvature, sag, a conic's
quadratic form, a Cartesian oval's implicit form — expressed with no
notion of a ray tracer, a medium, or a traceable surface. Solving a ray
against one of these shapes is :mod:`raytracer.math.intersections`; turning
a shape into something an engine can hit is :mod:`raytracer.optics`.
"""

from .cartesian_oval import (
    CartesianOvalProfile,
    cartesian_oval_parametric_curve,
    cartesian_oval_sag_and_slope,
    gots_params,
    max_usable_height,
)
from .conic import quadratic_coeffs_from_ep
from .fermat_oval import fermat_oval_F, fermat_oval_grad, sigma_parametric
from .profile import AsphereProfile

__all__ = [
    "AsphereProfile",
    "quadratic_coeffs_from_ep",
    "gots_params",
    "cartesian_oval_parametric_curve",
    "cartesian_oval_sag_and_slope",
    "max_usable_height",
    "CartesianOvalProfile",
    "fermat_oval_F",
    "fermat_oval_grad",
    "sigma_parametric",
]
