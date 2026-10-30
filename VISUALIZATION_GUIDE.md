# Guía del visor OpenGL

El visor (`raytracer.viz.gl.OpenGLViewer`) dibuja el plano meridional de un
sistema y sus rayos como tubos gaussianos acumulados en HDR:

1. Cada segmento de rayo se expande a un quad en espacio de pantalla (el
   ancho es isótropo en píxeles para cualquier orientación) y se mezcla
   **aditivamente** (`ONE, ONE`) en un framebuffer float32 en unidades
   radiométricas lineales — la superposición de miles de rayos nunca satura.
2. Un pase de tone-mapping a pantalla completa comprime la energía acumulada
   para el display (`exponential` por defecto, `reinhard` o `linear`
   opcionales) seguido de gamma 2.2. Con `auto_exposure=True` (defecto), la
   exposición se fija para que el percentil 99 de energía quede cerca del
   blanco.
3. Superficies y marcadores se dibujan encima como overlays (no radiometría).

## RenderConfig

| Campo | Defecto | Significado |
|---|---|---|
| `ray_width` | 0.5 | ancho del rayo en unidades de mundo |
| `sigma_factor` | 0.5 | σ del perfil gaussiano = ancho_px × factor |
| `min_pixels` | 1.0 | ancho mínimo en píxeles al alejar el zoom |
| `use_solid_rays` | False | perfil sólido con borde suave en vez de gaussiano |
| `default_intensity` | 1.0 | intensidad base (multiplica la potencia de cada segmento) |
| `weight_scale` | 1.0 | escala global de energía (raramente necesaria) |
| `exposure` | 1.0 | exposición manual (multiplica la automática) |
| `tone_map` | "exponential" | `exponential` \| `reinhard` \| `linear` |
| `auto_exposure` | True | exposición por percentil 99 del buffer HDR |
| `background` | "black" | color de fondo |

## API

- `OpenGLViewer(x_lims, y_lims, size=(900, 700), bgcolor=None, render_config=None)`
  — la vista inicial contiene el rectángulo dado, con aspecto bloqueado. El
  plano del dibujo es `(z, y)`: el corte meridional del modelo 3-D.
- `draw(drawing)` — dibuja un `raytracer.viz.drawing.Drawing` completo: el
  mismo dibujo neutral que consume el backend de matplotlib.
- `draw_system(system, wavelength_um=None, tree=None)` — todos los
  elementos de un `OpticalSystem`, con lo que registró cada sensor en `tree`.
- `draw_elements(elements)` — elementos sueltos, en su pose o como pares
  `(elemento, pose)`.
- `draw_rays(tree, colors=None, intensities=None, select=None, markers=False)`
  — un `RayTree` del trazador no secuencial. `colors` e `intensities` son
  arreglos (uno por segmento) o funciones del árbol; las intensidades valen
  por defecto `default_intensity × tree.power`. Los rayos que escapan se
  extienden hasta el borde de la vista (se recalculan al hacer pan/zoom).
- `draw_ray_segments(segments, colors=None, intensities=None)` — segmentos
  `(N, 2, 2)` ya en el plano del dibujo; los haces se acumulan hasta
  `clear_rays()`.
- `draw_polyline`, `draw_filled_polygon`, `draw_markers`; `clear()` borra todo.
- `snapshot()` — render offscreen a RGBA uint8 (para guardar imágenes o
  tests headless). `read_accumulation()` devuelve el buffer HDR float.
- `run()`.

## Interacción

- **Arrastrar (botón izquierdo):** paneo (el contenido sigue al cursor).
- **Rueda:** zoom anclado al cursor (el punto bajo el cursor no se mueve).
- El pan/zoom solo actualiza uniforms — no se reconstruyen VBOs.

## Colorimetría

`raytracer.viz.color.wavelength_to_rgb(λ_nm)` convierte longitudes de onda a
sRGB lineal (ajustes CIE 1931). Colores espectrales, uno por segmento:

```python
import numpy as np
from raytracer.viz.color import wavelength_to_rgb

colors = np.array([(*wavelength_to_rgb(wl * 1e3), 1.0) for wl in tree.wavelength_um])
viewer.draw_rays(tree, colors=colors)
```

Como la acumulación es lineal en RGB, las fuentes de distintas longitudes de
onda se suman físicamente (mezcla aditiva de espectros) antes del tone-mapping.
