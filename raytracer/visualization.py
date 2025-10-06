"""VisPy helpers for inspecting ray tracing scenes."""

from __future__ import annotations

import numpy as np
from vispy import app, scene
from vispy.color import Color

from .geometry import Surface2D
from .rays import RayTree


class Scene2DViewer:
    """Simple plotting wrapper for 2D ray tracing results."""

    def __init__(
        self,
        x_lims: tuple[float, float] = (-6.0, 6.0),
        y_lims: tuple[float, float] = (-6.0, 6.0),
        show_axis: bool = True,
    ) -> None:
        self.canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black", size=(900, 700))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
        if show_axis:
            scene.visuals.XYZAxis(parent=self.view.scene)
        self.view.camera.set_range(x=x_lims, y=y_lims)

    def draw_surfaces(self, surfaces: list[Surface2D]) -> None:
        positions: list[np.ndarray] = []
        connections: list[np.ndarray] = []
        vertex_offset = 0

        for surface in surfaces:
            segments = None
            if hasattr(surface, "polyline_segments"):
                segments = surface.polyline_segments()  # type: ignore[attr-defined]
            if not segments:
                segments = [surface.polyline()]

            for pts in segments:
                pts = np.asarray(pts, dtype=np.float32)
                if pts.shape[0] < 2:
                    continue
                positions.append(pts)
                idx = np.arange(pts.shape[0], dtype=np.uint32) + vertex_offset
                connections.append(np.column_stack((idx[:-1], idx[1:])))
                vertex_offset += pts.shape[0]

        if not positions:
            return

        pos = np.vstack(positions)
        connect = np.vstack(connections)
        scene.visuals.Line(
            pos=pos,
            connect=connect,
            color=Color("white").rgba,
            width=1.8,
            antialias=True,
            method="gl",
            parent=self.view.scene,
        )

    def draw_rays(
        self,
        tree: RayTree,
        tail_length: float = 12.0,
        width: float = 1.2,
        extend_mode: str = "none",
        extend_length: float = 0.0,
        axis_y: float = 0.0,
        show_misses: bool = True,
    ) -> None:
        segments: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        connections: list[list[int]] = []
        hit_positions: list[np.ndarray] = []
        vertex_index = 0

        miss_color = Color("orange").rgba
        hit_color = Color("yellow").rgba

        for node in sorted(tree.nodes(), key=lambda n: n.label):
            start = np.asarray(node.ray.origin, dtype=np.float32)
            direction = np.asarray(node.ray.direction, dtype=np.float32)
            if node.intersection is None:
                if not show_misses:
                    continue
                end = start + direction * float(tail_length)
                color = miss_color
            else:
                hit_point = np.asarray(node.intersection.point, dtype=np.float32)
                color = hit_color
                hit_positions.append(hit_point)
                end = hit_point

                mode = extend_mode.lower()
                if mode == "length" and extend_length > 0.0:
                    end = hit_point + direction * float(extend_length)
                elif mode == "axis":
                    denom = direction[1]
                    if abs(denom) > 1e-9:
                        t_ext = (float(axis_y) - hit_point[1]) / denom
                        if t_ext > 0.0:
                            end = hit_point + direction * t_ext

            segments.extend((start, end))
            colors.extend((color, color))
            connections.append([vertex_index, vertex_index + 1])
            vertex_index += 2

        if segments:
            pos = np.vstack(segments)
            connect = np.asarray(connections, dtype=np.uint32)
            color_arr = np.vstack(colors)
            scene.visuals.Line(
                pos=pos,
                connect=connect,
                color=color_arr,
                width=width,
                antialias=True,
                method="gl",
                parent=self.view.scene,
            )

        if hit_positions:
            hits = np.vstack(hit_positions)
            scene.visuals.Markers(
                pos=hits,
                size=5,
                face_color=Color("crimson").rgba,
                parent=self.view.scene,
            )

    @staticmethod
    def run() -> None:
        app.run()
