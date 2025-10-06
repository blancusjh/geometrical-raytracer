# RayTracer

RayTracer is a small 2D ray-tracing toolkit for exploring classical dioptrics. It focuses on constructive ray geometry rather than photorealistic rendering, making it handy for experimenting with reflective conics, Cartesian ovoids, and stigmatic singlets.

## Highlights

- Breadth-first ray tree with explicit generation labels and branching for reflection/refraction (`raytracer/tracer.py`).
- Catalog of analytic surfaces: conics (ellipse, parabola, circle) plus parametric Cartesian dioptriques and matched singlets (`raytracer/geometry.py`, `raytracer/dioptrics.py`).
- Source primitives for point and parallel emitters with simple fan/aperture sampling (`raytracer/sources.py`).
- Lightweight VisPy viewer that batches all surfaces/rays for interactive inspection (`raytracer/visualization.py`).

## Getting Started

### Dependencies

The core library only requires NumPy. Running the interactive examples additionally needs VisPy. Install everything into your environment:

```bash
pip install -r requirements.txt  # if you have one
# or manually
pip install numpy vispy
```

### Running Examples

Ready-made demonstrations live in `examples/`:

- `ellipse_depth5.py` – launches rays from one focus of an elliptical mirror and shows five internal reflections converging on the conjugate focus.
- `cartesian_dioptrique_focus.py` – traces refraction through a single Cartesian dioptrique from object distance `z0` to image distance `zi`.
- `cartesian_singlet_focus.py` – combines two matched Cartesian dioptriques into a stigmatic singlet, highlighting the object, intermediate, and final focal points.
- `high_aperture_singlet.py` – fires a wide (≈40°) fan of rays into a thick Cartesian singlet so you can inspect non-paraxial behaviour.

Execute an example with:

```bash
python examples/ellipse_depth5.py
```

The VisPy window supports pan/zoom via mouse interactions. Pass `extend_mode="axis"` to `Scene2DViewer.draw_rays` when you want each refracted branch to continue until it crosses the optical axis.

### Programmatic Use

The typical workflow is:

1. Build surfaces (mirrors or dioptriques) and supply their refractive indices.
2. Configure sources that emit seed rays.
3. Instantiate `RayTracer2D` with a `TraceConfig` specifying max generations and whether to enable reflection/refraction.
4. Call `trace([sources...])` to obtain a `RayTree` describing every segment.
5. Feed the tree to your analyser/visualiser of choice.

The tracer automatically flips surface normals based on incident direction and carries medium indices through each branch (`Gamma_k0` for reflections, `Gamma_k1` for refractions).

### Working With Cartesian Surfaces

`raytracer/dioptrics.py` exposes two main helpers:

- `CartesianDioptrique` – constructs a single surface from conjugate distances `(z0, zi)` and refractive pair `(n0, ni)`. You can provide either `rho_max` or an `aperture_radius`; it will numerically determine the appropriate profile and expose a polyline for tracing/visualisation.
- `CartesianSinglet` – wraps two compatible Cartesian dioptriques separated by a specified thickness, producing a matched front/back pair for lens studies.

Both are discretised into polylines, so increasing `samples` tightens accuracy for large apertures.

## Repository Layout

```
raytracer/
  dioptrics.py        # Cartesian dioptriques, singlet helper, sigma utilities
  geometry.py         # Conics and generic surface base class
  physics.py          # Reflection/refraction laws
  sources.py          # Point and parallel ray emitters
  tracer.py           # Breadth-first tracer with media support
  visualization.py    # Batched VisPy viewer for 2D scenes
examples/
  ellipse_depth5.py
  cartesian_dioptrique_focus.py
  cartesian_singlet_focus.py
```

## Development Notes

- The project currently has no automated test suite; when modifying tracing logic, consider scripting quick numerical checks similar to the ones used during development (see inline comments in the examples).
- The repository runs without git metadata in the provided environment; if you plan to contribute, initialise a repository and add a dependency specification (`requirements.txt` or `pyproject.toml`).
- VisPy raises a `ModuleNotFoundError` if not installed. To avoid blocking environments without GUI support, gate imports or provide headless fallbacks when integrating into larger toolchains.

## License

No explicit license file is present. Treat the code as private unless the project owner specifies otherwise.
