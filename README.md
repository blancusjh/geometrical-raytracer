# raytracer

Toolkit de trazado de rayos óptico con dos motores que comparten primitivas:

- **`raytracer.sequential`** — motor secuencial 3D exacto para sistemas
  definidos por prescripción (radios, espesores firmados, cónicas + asféricas
  pares, espejos, stop), con recuperación de conjugados paraxiales, resolución
  de rayo principal, muestreo de pupila vectorizado y una suite de análisis
  profesional: spot diagrams, ray fans, métricas de aberración, expansión de
  Zernike ANSI/OSA, frente de onda por eikonal, PSF escalar e imagen aérea
  parcialmente coherente (método de Abbe).
- **`raytracer.nonseq`** — motor 2D no-secuencial que produce el árbol
  completo de reflexiones/refracciones (cáusticas, exploración), con OPL,
  intensidad y división de Fresnel opcional.

Visualización en `raytracer.viz`: figuras de análisis en matplotlib
(`viz.plots`) y un visor OpenGL (`viz.gl`) con acumulación HDR en float y
tone-mapping auto-expuesto — la intensidad de miles de rayos superpuestos se
suma linealmente sin saturar el framebuffer.

## Instalación

```bash
pip install -e .          # numpy, scipy, matplotlib
pip install -e ".[gl]"    # + vispy para el visor OpenGL
```

## Motor secuencial: cargar una prescripción y analizarla

```python
from importlib import resources
from raytracer.sequential import (
    OpticalSystem, SequentialTracer, FieldPoint, PupilSampling,
    solve_object_plane, trace_pupil,
)
from raytracer.analysis import spot_data, fit_transverse
from raytracer.viz import plots

csv = resources.files("raytracer") / "data" / "US7557996_Fig3_Table3_prescription.csv"
system = OpticalSystem.from_prescription(csv)   # 48 superficies, 12 asféricas
tracer = SequentialTracer(system)
solve_object_plane(tracer)                      # recupera el plano objeto (B=0)

pupil = trace_pupil(tracer, FieldPoint(y=62.0), na_object_sine=0.3,
                    sampling=PupilSampling(kind="rings", radial=12, azimuth=72))
print(spot_data(pupil).rms_radius_um)           # 0.086 µm

fig = plots.layout_figure(tracer, fields=[56.0, 62.0, 67.0])
fig.savefig("layout.png", dpi=180)
```

Los notebooks en `examples/notebooks/` reproducen dos objetivos de patente
completos y sirven como referencia de la API:

- `us7557996_objective.ipynb` — objetivo litográfico DUV catadióptrico de
  inmersión (NA 1.2, λ=193.368 nm): réplica dígito a dígito del análisis de
  referencia.
- `euv_six_mirror.ipynb` — objetivo EUV de seis espejos (US 7,151,592,
  NA 0.22, λ=13.4 nm): validación analítica, refit de asféricas por mínimos
  cuadrados, frente de onda por eikonal, imagen de Abbe y curva de contraste.

## Motor 2D no-secuencial + visor OpenGL

```python
import numpy as np
from raytracer import EllipseConic, PointSource2D, RayTracer2D, TraceConfig
from raytracer import OpenGLViewer, RenderConfig

mirror = EllipseConic(semi_major=4.0, semi_minor=2.5)
source = PointSource2D(origin=np.array([0.0, 0.1]),
                       axis_direction=np.array([-1.0, 0.0]),
                       aperture=np.deg2rad(60.0), samples=2000)
tree = RayTracer2D([mirror], TraceConfig(max_generations=3)).trace([source])

viewer = OpenGLViewer(x_lims=(-8, 2), y_lims=(-4, 4))
viewer.draw_surfaces([mirror])
viewer.draw_rays(tree)     # acumulación HDR + auto-exposición
viewer.run()
```

Demos ejecutables en `examples/`, organizadas por carpeta:

- `stigmatic_surfaces/` — superficies diseñadas para estigmatismo perfecto:
  `ellipse_mirror.py` (foco a foco), `cartesian_oval_refractor_2d.py` y
  `cartesian_oval_refractor_3d.py` (óvalo de Descartes, parametrización GOTS,
  2D implícito vs 3D con sag cerrado). Los tres verifican el estigmatismo
  cuantitativamente (RMS/OPD ~0), no solo lo ilustran.
- `telescopes/` — `galilean.py` y `keplerian.py` (motor secuencial, separación
  resuelta para condición afocal exacta, magnificación angular medida por
  trazado) y `newtonian.py` (motor 2D no-secuencial, primario parabólico +
  secundario plano a 45°, con obstrucción central).
- `imaging/` — `single_lens_relay.py` (misma escena en matplotlib y OpenGL),
  `interactive_doublet.py` (arrastra fuente/lente/pantalla, Tab cicla
  parámetros), `geometric_and_diffractive_imaging.py` (relevo geométrico de un
  objeto extendido vs. formación difractiva por método de Abbe).
- `lithography/` — `duv_objective_2d.py`/`duv_objective_3d.py`, el objetivo
  DUV US7557996 completo (48 superficies) en 2D (HDR) y 3D (mallas de
  revolución, cámara turntable).

## Estructura

```
raytracer/
  core/        # vectores, marcos, materiales, leyes (Snell, Fresnel)
  geometry/    # sag asférico compartido, cónicas 2D, ovoides, óvalo de
               # Descartes (GOTS, compartido entre el ovoide 2D y el
               # CartesianOvalProfile 3D)
  sequential/  # filas de superficie, sistema, prescripción CSV, trazador
               # vectorizado, capa paraxial, campos y pupilas
  analysis/    # spots, fans, métricas, Zernike, eikonal, imaging escalar,
               # verificación de estigmatismo (spot RMS + OPD)
  nonseq/      # rayos, superficies, fuentes (incl. ImageSource2D), lentes/
               # espejos 2D, pantallas de observación, trazador no-secuencial
  viz/         # plots matplotlib, muestreo de sag compartido (sag_drawing),
               # escena neutral dual-backend (scene/mpl/protocol), color
               # (CIE), visor OpenGL HDR con modo geométrico no-aditivo (gl/),
               # modo interactivo (interactive), visor 3D (gl3d)
  data/        # prescripción US7557996 empaquetada
examples/      # demos por categoría (stigmatic_surfaces/, telescopes/,
               # imaging/, lithography/) + notebooks de réplica de patentes
reference/     # notebooks y módulo de referencia originales
tests/         # verdades analíticas + regresiones de sistema completo
```

## Tests

```bash
pytest                 # suite completa (87 tests)
pytest -m "not gpu"    # sin los smoke tests de OpenGL
```

Las verdades de referencia incluyen: conjugados de parábola/elipsoide a
precisión de máquina, magnificación paraxial del objetivo DUV
(+0.2500000135), NA de imagen 1.1977, RMS de spots y Zernike (0.5089 nm tras
quitar tilt/defoco), primer cero de Airy, colapso de contraste en el límite
de coherencia parcial, y linealidad aditiva del framebuffer HDR.
