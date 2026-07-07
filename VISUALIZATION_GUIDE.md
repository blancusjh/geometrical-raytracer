# OpenGL Visualisation Guide

The simplified toolkit ships with a single 2-D OpenGL viewer (`raytracer.visualization_opengl.OpenGLViewer`). It renders surfaces as polylines and rays as Gaussian tubes whose intensity is accumulated either quadratically or via alpha blending.

## Basic Usage

```python
from raytracer.visualization_opengl import OpenGLViewer, RenderConfig

viewer = OpenGLViewer(
    x_lims=(-8.0, 2.0),
    y_lims=(-4.0, 4.0),
    render_config=RenderConfig(
        ray_width=0.01,
        sigma_factor=0.01,
        accumulation_mode="squared",
    ),
)

viewer.draw_surfaces([surface])
viewer.draw_rays(ray_tree)
viewer.run()
```

- **Mouse drag** pans the view.
- **Scroll** zooms in/out.

## Render Configuration

| Field | Description |
| ----- | ----------- |
| `ray_width` | Base radius (in world units) used to compute a screen-space width. |
| `sigma_factor` | Scales the Gaussian falloff relative to the width. |
| `accumulation_mode` | Either `"squared"` (energy-like accumulation) or `"alpha"` (standard alpha blending). |
| `default_intensity` | Fallback intensity for rays when no resolver is provided. |
| `weight_scale` | Linear multiplier applied before accumulation. |
| `min_pixels` | Minimum on-screen thickness to keep very thin rays visible. |

## Customising Colours & Intensities

Both `draw_rays` and `draw_markers` accept callables to override colours or intensities:

```python
viewer.draw_rays(
    tree,
    color_resolver=lambda node: (1.0, 0.8, 0.2, 0.8),
    intensity_resolver=lambda node: 1.0 / (1.0 + node.generation),
)
```

Return either RGB or RGBA tuples. Intensities are positive scalars multiplied by the Gaussian profile.

## Surface Rendering

`draw_surfaces` expects surfaces providing a `polyline()` method (every `Surface2D` subclass does). Pass optional colour and width overrides:

```python
viewer.draw_surfaces([mirror, lens], color="white", width=2.0)
```

The viewer stores the latest ray/surface geometry and refreshes automatically when the canvas resizes. Call `update_visual_params_only()` after tweaking `RenderConfig` fields to rebuild ray widths without re-uploading geometry.
