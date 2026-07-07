"""Backward-compatible import path for the OpenGL viewer.

The implementation now lives in :mod:`raytracer.viz.gl`.
"""

from .viz.gl.viewer import OpenGLViewer, RenderConfig

__all__ = ["OpenGLViewer", "RenderConfig"]
