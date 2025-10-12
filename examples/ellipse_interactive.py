
"""Interactive ellipse example with OpenGL visualization and parameter sliders."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.geometry import EllipseConic
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization_opengl import OpenGLViewer, RenderConfig
from raytracer.interactive import InteractiveController, SliderConfig
from vispy import scene
from vispy.color import Color


class InteractiveEllipseDemo:
    """Interactive ellipse ray tracing demo with sliders."""

    def __init__(self):
        # Fixed ellipse parameters
        self.semi_major = 4.0
        self.semi_minor = 2.5

        # Create ellipse (assumes EllipseConic defines .focus and .frame)
        self.ellipse = EllipseConic(
            semi_major=self.semi_major,
            semi_minor=self.semi_minor,
            focus=np.array([0.0, 0.0]),
            surface_id="mirror",
        )

        # Foci (for markers)
        major_axis_dir = self.ellipse.frame.direction_to_world(np.array([1.0, 0.0], dtype=float))
        major_axis_dir = major_axis_dir / np.linalg.norm(major_axis_dir)
        focal_offset = float(np.sqrt(self.semi_major**2 - self.semi_minor**2))
        self.primary_focus = self.ellipse.focus.copy()
        self.secondary_focus = self.primary_focus - 2.0 * focal_offset * major_axis_dir
        self.major_axis_dir = major_axis_dir

        # Parameters
        self.params = {
            'source_x': 0.0,
            'source_y': 0.11,
            'direction_angle': 180.0,   # degrees
            'aperture': 80.0,           # degrees
            'samples': 500,
            'max_generations': 3,
            'ray_width': 0.001,          # match slider min below
            'ray_intensity': 1.0,
            'sigma_factor': 0.5,
        }

        self.cached_tree = None
        self.show_markers = True

        # Viewer + render config
        render_config = RenderConfig(
            ray_width=self.params['ray_width'],
            sigma_factor=self.params['sigma_factor'],
            accumulation_mode='squared',     # additive sqrt accumulation
            default_intensity=self.params['ray_intensity'],
            weight_scale=1.0,                # alpha from color acts as weight
        )

        self.viewer = OpenGLViewer(
            mode="2d",
            x_lims=(-8.0, 1.0),
            y_lims=(-4.0, 4.0),
            show_axis=True,
            size=(1200, 800),
            bgcolor="black",
            render_config=render_config,
        )

        # Toggle hit markers
        @self.viewer.canvas.events.key_press.connect
        def on_key(event):
            if event.key == 'v':
                self.show_markers = not self.show_markers
                print(f"Markers: {'ON' if self.show_markers else 'OFF'}")
                if self.cached_tree is not None:
                    self._redraw_from_cache()

        # Controls
        self.controller = InteractiveController(self.viewer.canvas, self.on_parameter_update)

        self.controller.add_slider(SliderConfig('source_x', -3.0, 3.0, self.params['source_x'], step=0.1, format_str="{:.2f}"))
        self.controller.add_slider(SliderConfig('source_y', -2.0, 2.0, self.params['source_y'], step=0.05, format_str="{:.2f}"))
        self.controller.add_slider(SliderConfig('direction_angle', 0.0, 360.0, self.params['direction_angle'], step=5.0, format_str="{:.1f}°"))
        self.controller.add_slider(SliderConfig('aperture', 10.0, 180.0, self.params['aperture'], step=5.0, format_str="{:.1f}°"))
        self.controller.add_slider(SliderConfig('samples', 50, 2000, self.params['samples'], step=50, format_str="{:.0f}"))
        self.controller.add_slider(SliderConfig('max_generations', 1, 10, self.params['max_generations'], step=1, format_str="{:.0f}"))

        self.controller.add_slider(SliderConfig('ray_width', 0.001, 2.0, self.params['ray_width'], step=0.01, format_str="{:.4f}"))

        self.controller.add_slider(SliderConfig('ray_intensity', 0.1, 5.0, self.params['ray_intensity'], step=0.1, format_str="{:.2f}"))
        self.controller.add_slider(SliderConfig('sigma_factor', 0.1, 2.0, self.params['sigma_factor'], step=0.05, format_str="{:.2f}"))

        self.controller.enable_numeric_overlay(
            ['samples', 'ray_width', 'sigma_factor'],
            format_overrides={'samples': '{:.0f}', 'ray_width': '{:.4f}', 'sigma_factor': '{:.2f}'},
            anchor=(20.0, 35.0),
            line_height=18.0,
        )

        self.controller.print_help()
        self.update_visualization()

    # ------------------------------------------------------------------ Callbacks

    def on_parameter_update(self, params: dict) -> None:
        old = self.params.copy()
        self.params.update(params)

        sim_changed = any(old.get(k) != self.params.get(k)
                          for k in ['source_x', 'source_y', 'direction_angle', 'aperture', 'samples', 'max_generations'])
        vis_changed = any(old.get(k) != self.params.get(k)
                          for k in ['ray_width', 'ray_intensity', 'sigma_factor'])

        # Update render config values used by the shader
        self.viewer.render_config.ray_width = float(self.params['ray_width'])
        self.viewer.render_config.default_intensity = float(self.params['ray_intensity'])
        self.viewer.render_config.sigma_factor = float(self.params['sigma_factor'])

        if sim_changed:
            self.update_visualization(full_update=True)
        elif vis_changed:
            # Fast path: no retrace required
            self._redraw_from_cache()

    # ------------------------------------------------------------------ Drawing

    def _current_source(self):
        direction_rad = np.deg2rad(self.params['direction_angle'])
        direction = np.array([np.cos(direction_rad), np.sin(direction_rad)], dtype=float)
        return PointSource2D(
            origin=np.array([self.params['source_x'], self.params['source_y']], dtype=float),
            axis_direction=direction,
            aperture=np.deg2rad(self.params['aperture']),
            samples=int(self.params['samples']),
        )

    def _per_ray_intensity(self, node) -> float:
        # Example: simple generation falloff
        return 1.0 / (1.0 + 0.3 * float(node.generation))

    def _redraw_from_cache(self) -> None:
        if self.cached_tree is None:
            return

        if not self.show_markers:
            self.viewer.clear_markers()

        # Use color alpha as a *weight* so sample changes modify brightness under squared accumulation
        # Heuristic: keep total contribution roughly stable with samples.
        # If you want exact energy normalization, use weight_scale in RenderConfig instead.
        ray_weight = min(1.0, 50.0 / float(self.params['samples']))  # in [0,1]

        def color_resolver(node):
            if node.intersection is None:
                return (1.0, 0.5, 1.0, 0.5 * ray_weight)  # miss
            else:
                return (1.0, 1.0, 1.0, ray_weight)        # hit

        self.viewer.draw_rays(
            self.cached_tree,
            width=self.params['ray_width'],
            color_resolver=color_resolver,
            intensity_resolver=self._per_ray_intensity,
            show_misses=True,
            update_markers=self.show_markers,
        )
        self.viewer.canvas.update()

    def update_visualization(self, full_update: bool = True) -> None:
        if not full_update and self.cached_tree is not None:
            self._redraw_from_cache()
            return

        # Full update: retrace and redraw
        tracer = RayTracer2D(
            surfaces=[self.ellipse],
            config=TraceConfig(max_generations=int(self.params['max_generations']),
                               allow_reflection=True, allow_refraction=False)
        )
        self.cached_tree = tracer.trace([self._current_source()])

        # Draw surfaces (on full updates)
        self.viewer.draw_surfaces([self.ellipse], color="white", width=2.0)

        self._redraw_from_cache()

        # Add focus markers once
        if not hasattr(self, '_markers_added'):
            focus_markers = np.vstack([self.primary_focus, self.secondary_focus])
            colors = np.array([[0.3, 0.9, 0.3, 1.0], [0.1, 0.7, 1.0, 1.0]], dtype=np.float32)
            scene.visuals.Markers(
                pos=focus_markers.astype(np.float32),
                size=8,
                face_color=colors,
                parent=self.viewer.view.scene
            )
            self._markers_added = True

        self.viewer.canvas.update()

    def run(self):
        print("\n🎯 Interactive Ellipse Ray Tracing Demo")
        print("=" * 50)
        self.viewer.run()


def main():
    demo = InteractiveEllipseDemo()
    demo.run()


if __name__ == "__main__":
    main()
