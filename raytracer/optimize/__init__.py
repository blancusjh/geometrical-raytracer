"""Design optimization: searching a family of exact systems for a property.

Everything below :mod:`raytracer.analysis` *measures*; this package *moves
parameters*. Its distinguishing trait is what is being varied: not surface
coefficients fitting a target shape, but the free conjugates of families
whose members are all exactly right in one sense (stigmatic) while the
optimizer buys a second property (aplanatism, a flat image surface) with
the remaining freedom.
"""

from .aplanat import (
    AplanatFit,
    aplanatism_residuals,
    constraint_residuals,
    optimize_aplanat,
    optimize_train,
)
from .constraints import (
    Aplanatism,
    AxialColor,
    Constraint,
    Distortion,
    EvaluationContext,
    FlatImageSurface,
    TargetMagnification,
)

__all__ = [
    "AplanatFit",
    "aplanatism_residuals",
    "constraint_residuals",
    "optimize_aplanat",
    "optimize_train",
    "Constraint",
    "EvaluationContext",
    "Aplanatism",
    "FlatImageSurface",
    "Distortion",
    "TargetMagnification",
    "AxialColor",
]
