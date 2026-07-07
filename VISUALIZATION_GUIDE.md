# Guía del visor OpenGL

El visor (`raytracer.viz.gl.OpenGLViewer`) dibuja árboles de rayos 2D como
tubos gaussianos acumulados en HDR:

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
| `default_intensity` | 1.0 | intensidad base (multiplica `node.intensity`) |
| `weight_scale` | 1.0 | escala global de energía (raramente necesaria) |
| `exposure` | 1.0 | exposición manual (multiplica la automática) |
| `tone_map` | "exponential" | `exponential` \| `reinhard` \| `linear` |
| `auto_exposure` | True | exposición por percentil 99 del buffer HDR |
| `background` | "black" | color de fondo |

Los campos legacy (`accumulation_mode="squared"/"alpha"`) se aceptan y se
mapean al pipeline nuevo; los hacks de peso por número de muestras
(`weight_scale=min(1, 80/samples)`) ya no son necesarios.

## API

- `OpenGLViewer(x_lims, y_lims, size=(900,700), bgcolor=None, render_config=None)`
  — la vista inicial contiene el rectángulo dado con aspecto bloqueado.
- `draw_surfaces(surfaces, color="white", width=2.0)` — polilíneas de
  superficie como quads por segmento (el grosor funciona en cualquier GPU,
  sin depender de `glLineWidth`).
- `draw_rays(tree, color_resolver=None, intensity_resolver=None,
  show_misses=True, marker_color=None, marker_size=6.0)` — los resolvers
  reciben cada `RayNode`; el alfa del color actúa como peso por rayo. Los
  rayos que escapan se extienden hasta el borde de la vista actual (se
  recalculan solos al hacer pan/zoom).
- `draw_markers(points, color, size)` / `clear_markers()`.
- `update_visual_params_only(**campos_de_config)` — ajusta parámetros sin
  retesela geometría.
- `snapshot()` — render offscreen a RGBA uint8 (para guardar imágenes o
  tests headless). `read_accumulation()` devuelve el buffer HDR float.
- `run()` / `close()`.

## Interacción

- **Arrastrar (botón izquierdo):** paneo (el contenido sigue al cursor).
- **Rueda:** zoom anclado al cursor (el punto bajo el cursor no se mueve).
- El pan/zoom solo actualiza uniforms — no se reconstruyen VBOs.

## Colorimetría

`raytracer.viz.color.wavelength_to_rgb(λ_nm)` convierte longitudes de onda a
sRGB lineal (ajustes CIE 1931). Ejemplo de resolver espectral:

```python
from raytracer.viz.color import wavelength_to_rgb

def color_resolver(node):
    rgb = wavelength_to_rgb(node.wavelength_um * 1e3)
    return (*rgb, 1.0)
```

Como la acumulación es lineal en RGB, las fuentes de distintas longitudes de
onda se suman físicamente (mezcla aditiva de espectros) antes del tone-mapping.
