"""VisPy helpers for inspecting ray tracing scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from vispy import app, scene
from vispy.color import Color
from vispy.io import write_png

from .geometry import Surface2D
from .rays import RayTree


def _stack_segments(segments: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack polyline segments and return positions/connect indices."""
    if not segments:
        raise ValueError("segments must be non-empty")
    pos_list: list[np.ndarray] = []
    connect_list: list[np.ndarray] = []
    cursor = 0
    for segment in segments:
        segment = np.asarray(segment, dtype=np.float32)
        points = segment.shape[0]
        if points < 2:
            continue
        pos_list.append(segment)
        if points > 2:
            idx = np.column_stack((np.arange(cursor, cursor + points - 1, dtype=np.uint32),
                                    np.arange(cursor + 1, cursor + points, dtype=np.uint32)))
        else:
            idx = np.array([[cursor, cursor + 1]], dtype=np.uint32)
        connect_list.append(idx)
        cursor += points
    if not pos_list:
        raise ValueError("segments only contained degenerate polylines")
    pos = np.vstack(pos_list)
    connect = np.vstack(connect_list)
    return pos.astype(np.float32), connect.astype(np.uint32)


class Scene2DViewer:
    """Simple plotting wrapper for 2D ray tracing results."""

    def __init__(
        self,
        x_lims: tuple[float, float] = (-6.0, 6.0),
        y_lims: tuple[float, float] = (-6.0, 6.0),
        show_axis: bool = True,
        *,
        show: bool = True,
        size: tuple[int, int] = (900, 700),
        bgcolor: str | tuple[float, float, float, float] = "black",
        line_method: str = "gl",
        antialias: bool = True,
    ) -> None:
        self.canvas = scene.SceneCanvas(keys="interactive", show=show, bgcolor=bgcolor, size=size)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
        if show_axis:
            scene.visuals.XYZAxis(parent=self.view.scene)
        self.view.camera.set_range(x=x_lims, y=y_lims)

        method = line_method.lower()
        if method not in {"gl", "agg"}:
            raise ValueError("line_method must be 'gl' or 'agg'")
        self._line_method = method
        self._antialias = bool(antialias)

        self._surface_visuals: list[scene.VisualNode] = []
        self._ray_visuals: list[scene.VisualNode] = []
        self._marker_visuals: list[scene.VisualNode] = []

    # ------------------------------------------------------------------ utils
    def _clear_visuals(self, attr: str) -> None:
        visuals: Iterable[scene.VisualNode] = getattr(self, attr, [])
        for visual in visuals:
            visual.parent = None
        setattr(self, attr, [])

    # ---------------------------------------------------------------- surfaces
    def draw_surfaces(self, surfaces: list[Surface2D]) -> None:
        self._clear_visuals("_surface_visuals")
        segments: list[np.ndarray] = []
        for surface in surfaces:
            if hasattr(surface, "polyline_segments"):
                raw_segments = surface.polyline_segments()  # type: ignore[attr-defined]
            else:
                raw_segments = [surface.polyline()]
            for seg in raw_segments:
                seg = np.asarray(seg, dtype=np.float32)
                if seg.ndim != 2 or seg.shape[0] < 2:
                    continue
                segments.append(seg)

        if not segments:
            return

        if self._line_method == "gl":
            pos, connect = _stack_segments(segments)
            line = scene.visuals.Line(
                pos=pos,
                connect=connect,
                color=Color("white").rgba,
                width=1.8,
                antialias=self._antialias,
                method="gl",
                parent=self.view.scene,
            )
            self._surface_visuals.append(line)
        else:
            color = Color("white").rgba
            for seg in segments:
                line = scene.visuals.Line(
                    pos=seg,
                    color=color,
                    width=1.8,
                    antialias=self._antialias,
                    method="agg",
                    parent=self.view.scene,
                )
                self._surface_visuals.append(line)

    # --------------------------------------------------------------------- rays
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
        self._clear_visuals("_ray_visuals")
        self._clear_visuals("_marker_visuals")

        miss_color = Color("orange").rgba.astype(np.float32)
        hit_color = Color("yellow").rgba.astype(np.float32)
        extend_mode = extend_mode.lower()

        segment_entries: list[tuple[np.ndarray, np.ndarray]] = []
        hit_positions: list[np.ndarray] = []

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

                if extend_mode == "length" and extend_length > 0.0:
                    end = hit_point + direction * float(extend_length)
                elif extend_mode == "axis":
                    denom = direction[1]
                    if abs(denom) > 1e-9:
                        t_ext = (float(axis_y) - hit_point[1]) / denom
                        if t_ext > 0.0:
                            end = hit_point + direction * t_ext

            segment = np.vstack([start, end])
            segment_entries.append((segment, color))

        if not segment_entries:
            return

        if self._line_method == "gl":
            positions: list[np.ndarray] = []
            colors: list[np.ndarray] = []
            connects: list[np.ndarray] = []
            cursor = 0
            for seg, color in segment_entries:
                positions.append(seg)
                colors.append(np.vstack([color, color]))
                connects.append(np.array([[cursor, cursor + 1]], dtype=np.uint32))
                cursor += 2
            pos = np.vstack(positions)
            color_arr = np.vstack(colors)
            connect = np.vstack(connects)
            line = scene.visuals.Line(
                pos=pos,
                connect=connect,
                color=color_arr,
                width=width,
                antialias=self._antialias,
                method="gl",
                parent=self.view.scene,
            )
            self._ray_visuals.append(line)
        else:
            for seg, color in segment_entries:
                line = scene.visuals.Line(
                    pos=seg,
                    color=color,
                    width=width,
                    antialias=self._antialias,
                    method="agg",
                    parent=self.view.scene,
                )
                self._ray_visuals.append(line)

        if hit_positions:
            markers = scene.visuals.Markers(
                pos=np.asarray(hit_positions, dtype=np.float32),
                size=5,
                face_color=Color("crimson").rgba,
                parent=self.view.scene,
            )
            self._marker_visuals.append(markers)

    # ----------------------------------------------------------------- export
    def save(
        self,
        path: str | Path,
        size: tuple[int, int] | None = None,
        *,
        flip_y: bool = True,
        reset_size: bool = True,
    ) -> Path:
        """Render the current canvas to an image file."""
        target = Path(path)
        orig_size = self.canvas.size
        if size is not None:
            self.canvas.size = size
        app.process_events()
        image = self.canvas.render()
        if flip_y:
            image = np.flipud(image)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_png(str(target), image)
        if reset_size and size is not None:
            self.canvas.size = orig_size
        return target

    # ----------------------------------------------------------------- cleanup
    def close(self) -> None:
        self.canvas.close()

    @staticmethod
    def run() -> None:
        app.run()
