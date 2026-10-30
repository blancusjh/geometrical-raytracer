"""``show(drawing, backend=...)``: one neutral :class:`~raytracer.viz.scene.Drawing`, two backends.

``"mpl"`` renders in matplotlib (few rays, publication diagrams); ``"gl"`` opens the OpenGL viewer (dense ray
fields, HDR caustics, colorimetry).
"""

from __future__ import annotations

from typing import Literal

from .scene import Drawing


def show(drawing: Drawing, backend: Literal["mpl", "gl"] = "mpl", *, interactive: bool = True, **config):
    """Render *drawing* with the chosen backend and return the backend object."""

    if backend == "mpl":
        from .mpl import MplBackend

        b = MplBackend(**config)
    elif backend == "gl":
        b = GLBackend(**config)
    else:
        raise ValueError(f"Unknown backend {backend!r} (use 'mpl' or 'gl')")
    b.render(drawing)
    if interactive:  # pragma: no cover - blocks
        b.show()
    return b


class GLBackend:
    """The OpenGL viewer behind the backend interface (``render``, ``update``, ``show``, ``save``)."""

    def __init__(self, **viewer_kwargs) -> None:
        self.viewer_kwargs = viewer_kwargs
        self.viewer = None

    def render(self, drawing: Drawing):
        from .gl.viewer import OpenGLViewer

        x_lims, y_lims = drawing.bounds()
        self.viewer = OpenGLViewer(**{"x_lims": x_lims, "y_lims": y_lims, **self.viewer_kwargs})
        self.viewer.draw(drawing)
        return self.viewer

    def update(self, drawing: Drawing):
        if self.viewer is None:
            return self.render(drawing)
        self.viewer.clear()
        self.viewer.draw(drawing)
        return self.viewer

    def show(self) -> None:  # pragma: no cover - interactive
        self.viewer.run()

    def save(self, path) -> None:
        import matplotlib.image as mpimg

        mpimg.imsave(path, self.viewer.snapshot())  # already top-row first


__all__ = ["GLBackend", "show"]
