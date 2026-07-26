"""Deprecated: moved to :mod:`raytracer.nonseq.segments2d`.

``LineSegment2D``/``ProfileFace2D`` are ``Surface2D`` adapters (they depend
on the non-sequential engine's ``Ray2D``/``Intersection2D``), not
engine-agnostic geometry, so they now live with the rest of the
non-sequential engine.
"""

from __future__ import annotations

import warnings

from ..nonseq.segments2d import LineSegment2D, ProfileFace2D  # noqa: F401

warnings.warn(
    "raytracer.geometry.segments2d is deprecated; import LineSegment2D/ProfileFace2D "
    "from raytracer.nonseq.segments2d instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LineSegment2D", "ProfileFace2D"]
