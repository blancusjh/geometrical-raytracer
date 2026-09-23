"""Visualization: analysis figures, neutral scene, and dual backends."""

from . import plots
from .protocol import GLBackend, show
from .scene import (
    LensBodyItem,
    MarkerItem,
    RaySegmentsItem,
    Scene,
    ScreenItem,
    SurfaceItem,
)
from .solid_viewer import SolidViewer

__all__ = [
    "plots",
    "SolidViewer",
    "show",
    "GLBackend",
    "Scene",
    "RaySegmentsItem",
    "SurfaceItem",
    "LensBodyItem",
    "ScreenItem",
    "MarkerItem",
]
