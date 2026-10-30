"""Backend protocol and the `show()` dispatcher.

`show(drawing, backend="mpl")` renders in matplotlib (few rays, publication
diagrams); `show(drawing, backend="gl")` opens the OpenGL viewer (dense ray
fields, HDR caustics, colorimetry). Both consume the same neutral
:class:`~raytracer.viz.scene.Drawing`.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .scene import (
    Drawing,
    LensBodyItem,
    MarkerItem,
    RaySegmentsItem,
    SensorItem,
    SurfaceItem,
)


@runtime_checkable
class Backend(Protocol):
    def render(self, drawing: Drawing): ...

    def update(self, drawing: Drawing): ...

    def show(self) -> None: ...


def populate_viewer(viewer, drawing: Drawing) -> None:
    """Draw every item of *drawing* on an :class:`~raytracer.viz.gl.OpenGLViewer`."""

    for item in drawing.items:
        if isinstance(item, RaySegmentsItem):
            viewer.draw_ray_segments(
                item.segments,
                colors=item.colors,
                intensities=item.intensities,
                escaping=item.escaping,
                directions=item.directions,
            )
        elif isinstance(item, SurfaceItem):
            viewer.draw_polyline(item.polyline, color=item.color, width=item.width)
        elif isinstance(item, LensBodyItem):
            verts, faces = item.triangles()
            viewer.draw_filled_polygon(verts, faces, color=item.facecolor)
            outline = np.asarray(item.polygon, dtype=float)
            if not np.allclose(outline[0], outline[-1]):
                outline = np.vstack([outline, outline[:1]])
            viewer.draw_polyline(outline, color=item.edgecolor, width=1.5)
        elif isinstance(item, SensorItem):
            viewer.draw_polyline(item.polyline, color=item.color, width=item.width)
            if item.hits is not None and len(item.hits):
                viewer.draw_markers(item.hits, color=(0.86, 0.08, 0.24, 1.0), size=4.0)
        elif isinstance(item, MarkerItem):
            viewer.draw_markers(item.points, color=item.color, size=item.size)


class GLBackend:
    """Adapter mapping the neutral drawing onto the OpenGL viewer."""

    def __init__(self, **viewer_kwargs) -> None:
        self.viewer_kwargs = viewer_kwargs
        self.viewer = None

    def render(self, drawing: Drawing):
        from .gl.viewer import OpenGLViewer

        x_lims, y_lims = drawing.bounds()
        kwargs = dict(self.viewer_kwargs)
        kwargs.setdefault("x_lims", x_lims)
        kwargs.setdefault("y_lims", y_lims)
        self.viewer = OpenGLViewer(**kwargs)
        populate_viewer(self.viewer, drawing)
        return self.viewer

    def update(self, drawing: Drawing):
        if self.viewer is None:
            return self.render(drawing)
        self.viewer._surface_renderers.clear()
        self.viewer._fill_renderers.clear()
        self.viewer.clear_markers()
        self.viewer.clear_rays()  # ray bundles accumulate across draw calls
        populate_viewer(self.viewer, drawing)
        return self.viewer

    def show(self) -> None:  # pragma: no cover - interactive
        self.viewer.run()

    def save(self, path) -> None:
        import matplotlib.image as mpimg

        # snapshot() already returns standard top-row-first image order;
        # flipping here would mirror the drawing vertically.
        mpimg.imsave(path, self.viewer.snapshot())


def show(
    drawing: Drawing,
    backend: Literal["mpl", "gl"] = "mpl",
    *,
    interactive: bool = True,
    **config,
):
    """Render *drawing* with the chosen backend and return the backend object."""

    if backend == "mpl":
        from .mpl import MplBackend

        b = MplBackend(**config)
        b.render(drawing)
        if interactive:  # pragma: no cover - blocks
            b.show()
        return b
    if backend == "gl":
        b = GLBackend(**config)
        b.render(drawing)
        if interactive:  # pragma: no cover - blocks
            b.show()
        return b
    raise ValueError(f"Unknown backend {backend!r} (use 'mpl' or 'gl')")


__all__ = ["Backend", "GLBackend", "populate_viewer", "show"]
