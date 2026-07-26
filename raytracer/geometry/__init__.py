"""Engine-agnostic shape math: surface profiles and conic-section algebra.

Everything here is pure geometry — curvature, sag, conic/implicit-form
solving — expressed in terms of :mod:`raytracer.math` alone. It has no
notion of a ray tracer, a medium, or a traceable surface; turning a shape
into something the sequential or non-sequential engine can hit is what
:mod:`raytracer.sequential.surfaces` and the ``Surface2D`` adapters in
:mod:`raytracer.nonseq` (``conics2d``, ``segments2d``, ``ovoid2d``) are for.
"""

from .cartesian_oval import (
    CartesianOvalProfile,
    cartesian_oval_parametric_curve,
    cartesian_oval_sag_and_slope,
    gots_params,
    max_usable_height,
)
from .conics2d import ConicProfile, quadratic_coeffs_from_ep
from .ovoid2d import fermat_ovoid_F, fermat_ovoid_grad, sigma_parametric
from .sag import AsphereProfile

__all__ = [
    "AsphereProfile",
    "ConicProfile",
    "quadratic_coeffs_from_ep",
    "gots_params",
    "cartesian_oval_parametric_curve",
    "cartesian_oval_sag_and_slope",
    "max_usable_height",
    "CartesianOvalProfile",
    "fermat_ovoid_F",
    "fermat_ovoid_grad",
    "sigma_parametric",
]
