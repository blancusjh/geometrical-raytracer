# OpenGL Visualization Guide

## Overview

The RayTracer now includes an advanced OpenGL-based visualization system with:
- **Intensity Accumulation**: Physically-based intensity rendering using √(I₁² + I₂² + ...) accumulation
- **Gaussian Ray Rendering**: Rays rendered as tubes with configurable width and gaussian intensity falloff
- **Interactive Controls**: Real-time parameter adjustment via keyboard-based sliders
- **2D/3D Support**: Unified viewer for both 2D and 3D ray tracing
- **Backward Compatibility**: Original visualization systems remain unchanged

## Quick Start

### Basic OpenGL Visualization

```python
from raytracer import OpenGLViewer, RenderConfig

# Configure rendering
config = RenderConfig(
    ray_width=0.5,           # Width of ray tubes
    sigma_factor=0.5,        # Gaussian falloff factor
    accumulation_mode='squared'  # 'squared' or 'alpha'
)

# Create viewer
viewer = OpenGLViewer(
    mode="2d",               # "2d" or "3d"
    x_lims=(-10, 10),
    y_lims=(-10, 10),
    render_config=config
)

# Draw surfaces and rays
viewer.draw_surfaces([surface1, surface2])
viewer.draw_rays(ray_tree)
viewer.run()
```

### Interactive Example

```python
from raytracer import InteractiveController, SliderConfig

# Create controller
controller = InteractiveController(viewer.canvas, update_callback)

# Add sliders
controller.add_slider(SliderConfig(
    name='source_x',
    min_val=-3.0,
    max_val=3.0,
    initial=0.0,
    step=0.1
))

# Print help and run
controller.print_help()
viewer.run()
```

See `examples/ellipse_interactive.py` for a complete example.

## Intensity Accumulation Modes

### Squared Mode (Physically Based)
For each pixel, accumulates squared intensities: `I = √(I₁² + I₂² + ...)`

This provides realistic intensity addition for overlapping rays.

```python
RenderConfig(accumulation_mode='squared')
```

### Alpha Mode (Linear Blending)
Traditional alpha blending for artistic effects.

```python
RenderConfig(accumulation_mode='alpha')
```

## Gaussian Ray Rendering

Rays are rendered as tubes with intensity falloff:

```
I(r) = I₀ × exp(-(r/σ)²)
```

Where:
- `r` = distance from ray centerline
- `σ` = `ray_width × sigma_factor`

Configure via:
```python
RenderConfig(
    ray_width=0.5,      # Base width
    sigma_factor=0.5    # Controls falloff steepness
)
```

## Interactive Controls

### Keyboard Controls
- **Tab**: Switch active parameter
- **Up/Right**: Increase value
- **Down/Left**: Decrease value
- **Space**: Force update/re-render

### Camera Controls (3D Mode)
- **C**: Toggle between Turntable and Fly cameras
- **Mouse**: Rotate (Turntable) or look around (Fly)
- **WASD**: Move camera (Fly mode)

## API Reference

### OpenGLViewer

```python
class OpenGLViewer(
    mode: Literal["2d", "3d"] = "2d",
    x_lims: tuple[float, float] = (-6.0, 6.0),
    y_lims: tuple[float, float] = (-6.0, 6.0),
    z_lims: tuple[float, float] | None = None,
    show_axis: bool = True,
    show: bool = True,
    size: tuple[int, int] = (900, 700),
    bgcolor: str | tuple = "black",
    render_config: RenderConfig | None = None,
)
```

#### Methods

**`draw_surfaces(surfaces, color="white", width=2.0)`**
Draw surface geometries as lines.

**`draw_rays(tree, width=None, intensity_resolver=None, ...)`**
Draw rays with intensity accumulation.
- `intensity_resolver`: Optional function `(RayNode) -> float` to customize intensity
- `color_resolver`: Optional function `(RayNode) -> RGBA` to customize color

### RenderConfig

```python
@dataclass
class RenderConfig:
    ray_width: float = 0.5
    sigma_factor: float = 0.5
    accumulation_mode: Literal["squared", "alpha"] = "squared"
    default_intensity: float = 1.0
```

### InteractiveController

```python
class InteractiveController(
    canvas: SceneCanvas,
    update_callback: Callable[[dict[str, Any]], None]
)
```

#### Methods

**`add_slider(config: SliderConfig)`**
Add an interactive parameter.

**`get_value(name: str) -> float`**
Get current parameter value.

**`print_help()`**
Display control instructions.

### SliderConfig

```python
@dataclass
class SliderConfig:
    name: str
    min_val: float
    max_val: float
    initial: float
    step: float | None = None
    format_str: str = "{:.2f}"
```

## Backward Compatibility

All existing visualization systems remain available:

```python
# Original 2D viewer (unchanged)
from raytracer.visualization import Scene2DViewer

# Original 3D viewer (unchanged)
from raytracer.visualization_vispy import visualize_axisymmetric_scene
```

Existing examples will continue to work without modification.

## Advanced Usage

### Custom Intensity Resolver

Control per-ray intensity:

```python
def intensity_resolver(node: RayNode) -> float:
    # Reduce intensity for later generations
    return 1.0 / (1.0 + node.generation * 0.3)

viewer.draw_rays(tree, intensity_resolver=intensity_resolver)
```

### Custom Color Resolver

Dynamic ray coloring:

```python
def color_resolver(node: RayNode) -> tuple:
    # Color by generation
    colors = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
    return colors[node.generation % len(colors)]

viewer.draw_rays(tree, color_resolver=color_resolver)
```

### Real-time Parameter Updates

```python
def update_callback(params: dict) -> None:
    # Update simulation
    source.origin = np.array([params['x'], params['y']])

    # Re-trace
    tree = tracer.trace([source])

    # Re-render
    viewer.draw_surfaces([surface])
    viewer.draw_rays(tree)
    viewer.canvas.update()
```

## Examples

### 1. Basic 2D Visualization
```bash
python examples/ellipse_depth5.py  # Original example
```

### 2. Interactive OpenGL
```bash
python examples/ellipse_interactive.py  # New interactive example
```

## Performance Optimizations

The viewer is highly optimized for real-time interaction:

### Built-in Optimizations
1. **Instanced Rendering**: All rays rendered in a single GPU draw call
2. **Ray Tree Caching**: Cached between visual-only updates
3. **Smart Updates**: Visual parameters (width, sigma) update without re-tracing
4. **GPU Fragment Culling**: Early discard for low-intensity pixels (gaussian < 0.01)
5. **Visual Reuse**: Updates existing visuals instead of recreating

### Performance Tips

1. **Reduce sample count** for real-time interaction (100-1000 rays is smooth)
2. **Use smaller sigma_factor** for sharper rays (faster fragment shader)
3. **Separate parameter types**: Visual params update ~100x faster than simulation params
4. **Disable hit markers** for many rays: Pass `update_markers=False`
5. **Small ray_width values** (0.001-0.01) work best with many rays

## Troubleshooting

**Issue**: Rays appear too thick/thin
- Adjust `ray_width` in `RenderConfig`

**Issue**: Rays have hard edges
- Increase `sigma_factor` for smoother falloff

**Issue**: Low intensity/dark visualization
- Increase `default_intensity` or use `intensity_resolver`
- Switch to `accumulation_mode='alpha'` for brighter rendering

**Issue**: Interactive controls not working
- Check that `InteractiveController` is created before `viewer.run()`
- Ensure `print_help()` is called to see active parameters
