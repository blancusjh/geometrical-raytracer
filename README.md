# RayTracer

A compact 2-D ray-tracing toolkit for exploring reflection and refraction on conic surfaces. The package keeps only the essentials: analytic conics, Snell's law, a breadth-first ray tracer, and an OpenGL viewer for visualising ray trees.

## Features

- Geometric conics (ellipse, parabola, circle, hyperbola) with optional rotation and refractive indices (`raytracer/geometry.py`).
- Basic optical laws (Snell's law, reflect, refract) implemented on top of unit-vector helpers (`raytracer/physics.py`, `raytracer/rays.py`).
- Straightforward 2-D tracer that branches into reflected and refracted rays while carrying the active medium index (`raytracer/tracer.py`).
- Lightweight VisPy/OpenGL viewer that draws surfaces and ray trees with adjustable accumulation (`raytracer/visualization_opengl.py`).
- Simple point and parallel sources for seeding ray fans (`raytracer/sources.py`).

## Requirements

The minimal runtime dependency is NumPy. The OpenGL viewer and example scripts additionally require VisPy:

```bash
pip install numpy vispy
```

## Quick Start

```python
import numpy as np

from raytracer.geometry import EllipseConic
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization_opengl import OpenGLViewer

mirror = EllipseConic(semi_major=4.0, semi_minor=2.5)
source = PointSource2D(
    origin=np.array([0.0, 0.1]),
    axis_direction=np.array([-1.0, 0.0]),
    aperture=np.deg2rad(60.0),
    samples=200,
)

tree = RayTracer2D([mirror], TraceConfig(max_generations=3)).trace([source])

viewer = OpenGLViewer(x_lims=(-8, 2), y_lims=(-4, 4))
viewer.draw_surfaces([mirror])
viewer.draw_rays(tree)
viewer.run()
```

A runnable version of this demo lives in `examples/clean_ellipse_opengl.py`.

## Repository Layout

```
raytracer/
  __init__.py             # Public-facing re-exports
  geometry.py             # Conic sections and local-frame helpers
  physics.py              # Snell's law, reflect, refract
  rays.py                 # Core ray primitives and genealogy helpers
  sources.py              # Point and parallel 2-D emitters
  surfaces.py             # Surface base class and local frame utilities
  tracer.py               # Breadth-first 2-D tracer
  visualization_opengl.py # VisPy-based OpenGL viewer
examples/
  clean_ellipse_opengl.py # Minimal reflective ellipse demo
  conic_showcase.py       # General/hyperbola/parabola/sphere showcase
  refractive_sphere.py    # Parallel rays refracting through a glass circle
```

## Notes

- The codebase intentionally avoids additional abstractions; extensions (e.g., custom sources or surfaces) can subclass `Surface2D` or emit bespoke `RaySeed` batches.
- When modelling refractive interfaces, set `n_exterior`/`n_interior` on the surface so the tracer knows which medium to enter.
- The repository does not ship with an automated test suite. Run the example or craft small numerical checks when adjusting the tracer logic.

## License

No explicit license file is provided; treat the project as private unless stated otherwise by the owner.
