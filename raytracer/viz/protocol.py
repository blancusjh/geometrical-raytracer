"""Backend protocol and the `show()` dispatcher.

`show(scene, backend="mpl")` renders in matplotlib (few rays, publication
diagrams); `show(scene, backend="gl")` opens the OpenGL viewer (dense ray
fields, HDR caustics, colorimetry). Both consume the same neutral Scene.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .scene import (
    LensBodyItem,
    MarkerItem,
    RaySegmentsItem,
    Scene,
    ScreenItem,
    SurfaceItem,
)


@runtime_checkable
class Backend(Protocol):
    def render(self, scene: Scene): ...

    def update(self, scene: Scene): ...

    def show(self) -> None: ...


def _lens_body_mesh(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strip-triangulate a lens body polygon into (verts, faces).

    ``Lens2D.body_polygon()`` lays out M front-face points (h: -semi -> +semi)
    followed by the M back-face points reversed, so vertex ``i`` and vertex
    ``2M-1-i`` sit at the same height — pairing them yields a quad strip
    (same construction as ``sag_drawing.lens_fill_mesh``).
    """

    polygon = np.asarray(polygon, dtype=float)
    m = polygon.shape[0] // 2
    faces = np.empty((2 * (m - 1), 3), dtype=np.uint32)
    i = np.arange(m - 1, dtype=np.uint32)
    back = np.uint32(2 * m - 1) - i
    faces[0::2] = np.column_stack([i, i + 1, back])
    faces[1::2] = np.column_stack([i + 1, back - 1, back])
    return polygon, faces


class GLBackend:
    """Adapter mapping the neutral scene onto the OpenGL viewer."""

    def __init__(self, **viewer_kwargs) -> None:
        self.viewer_kwargs = viewer_kwargs
        self.viewer = None

    def render(self, scene: Scene):
        from .gl.viewer import OpenGLViewer

        x_lims, y_lims = scene.bounds()
        kwargs = dict(self.viewer_kwargs)
        kwargs.setdefault("x_lims", x_lims)
        kwargs.setdefault("y_lims", y_lims)
        self.viewer = OpenGLViewer(**kwargs)
        self._populate(scene)
        return self.viewer

    def update(self, scene: Scene):
        if self.viewer is None:
            return self.render(scene)
        self.viewer._surface_renderers.clear()
        self.viewer._fill_renderers.clear()
        self.viewer.clear_markers()
        self.viewer.clear_rays()  # ray bundles accumulate across draw calls
        self._populate(scene)
        return self.viewer

    def _populate(self, scene: Scene) -> None:
        viewer = self.viewer
        for item in scene.items:
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
                verts, faces = _lens_body_mesh(item.polygon)
                viewer.draw_filled_polygon(verts, faces, color=item.facecolor)
                viewer.draw_polyline(
                    np.vstack([item.polygon, item.polygon[:1]]),
                    color=item.edgecolor,
                    width=1.5,
                )
            elif isinstance(item, ScreenItem):
                viewer.draw_polyline(
                    np.array([item.p0, item.p1]), color=item.color, width=item.width
                )
            elif isinstance(item, MarkerItem):
                viewer.draw_markers(item.points, color=item.color, size=item.size)

    def show(self) -> None:  # pragma: no cover - interactive
        self.viewer.run()

    def save(self, path) -> None:
        import matplotlib.image as mpimg

        # snapshot() already returns standard top-row-first image order;
        # flipping here would mirror the scene vertically.
        mpimg.imsave(path, self.viewer.snapshot())


def show(
    scene: Scene,
    backend: Literal["mpl", "gl"] = "mpl",
    *,
    interactive: bool = True,
    **config,
):
    """Render *scene* with the chosen backend and return the backend object."""

    if backend == "mpl":
        from .mpl import MplBackend

        b = MplBackend(**config)
        b.render(scene)
        if interactive:  # pragma: no cover - blocks
            b.show()
        return b
    if backend == "gl":
        b = GLBackend(**config)
        b.render(scene)
        if interactive:  # pragma: no cover - blocks
            b.show()
        return b
    raise ValueError(f"Unknown backend {backend!r} (use 'mpl' or 'gl')")


__all__ = ["Backend", "GLBackend", "show"]
