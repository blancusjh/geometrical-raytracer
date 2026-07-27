# raytracer

Trazador de rayos óptico en Python. Propaga luz **exactamente** —sin
aproximación paraxial— a través de sistemas ópticos reales, y mide lo que
sale: aberraciones, frente de onda, PSF, formación de imagen.

![Objetivo DUV US7557996, tres longitudes de onda](docs/img/duv_3d_spectrum.png)

## Qué hace

Dado un sistema óptico —una prescripción de superficies, o un puñado de
lentes y espejos colocados en el espacio— el paquete responde a tres tipos
de pregunta:

**¿Por dónde va la luz?** Cada rayo se intersecta con cada superficie
resolviendo la ecuación de la superficie, no una aproximación de ella: para
una cónica hay forma cerrada, para un perfil asférico Newton sobre
`f_Σ(A + λu) = 0`, para una superficie implícita cualquiera un solve
robusto. La refracción y la reflexión se aplican en forma vectorial
completa, con reflexión total interna detectada, no asumida ausente.

**¿Qué tan bien forma imagen?** Spot diagrams ponderados por área de pupila,
ray fans transversales, métricas de distorsión por rayo principal (la
convención de Zemax/OpticStudio), coeficientes de Seidel deducidos del
frente de onda real, expansión de Zernike ANSI/OSA, aberración cromática
axial y lateral, frente de onda por eikonal, PSF escalar e imagen aérea
parcialmente coherente por el método de Abbe.

**¿Cómo se ve?** Un visor OpenGL que acumula los rayos aditivamente en un
framebuffer de coma flotante: donde se superponen miles de rayos la
intensidad *suma*, y un tone-mapping auto-expuesto revela la estructura del
haz sin saturar. Es lo que produce la imagen de arriba — el objetivo
litográfico DUV de la patente US 7,557,996 (48 superficies, 12 asféricas,
NA 1.2) recorrido por tres haces de distinta longitud de onda que se separan
en el camino plegado y vuelven a sumarse a blanco en la imagen.

## Dos motores de propagación

La diferencia no es 2-D contra 3-D: es qué hace un rayo al llegar a una
superficie.

**Secuencial** — cada rayo atraviesa cada superficie en orden fijo, un solo
camino determinista. Es el modelo de diseño óptico clásico y el que permite
vectorizar: los N rayos avanzan juntos en numpy. Con él vienen la capa
paraxial (recuperación de conjugados, matrices ABCD), la resolución de rayo
principal y el muestreo de pupila.

**Ramificado** — en cada colisión el rayo se divide en su hijo reflejado
*y* su hijo refractado, generando el árbol completo. Sirve para lo que la
secuencia deliberadamente ignora: cáusticas, luz parásita, reflexiones
múltiples. Lleva OPL e intensidad, con división de Fresnel opcional.

```python
from pathlib import Path
from raytracer.design import OpticalSystem
from raytracer.propagation import (
    SequentialTracer, FieldPoint, PupilSampling, solve_object_plane, trace_pupil,
)
from raytracer.analysis import spot_data

system = OpticalSystem.from_prescription(
    Path("data/US7557996_Fig3_Table3_prescription.csv")
)
tracer = SequentialTracer(system)
solve_object_plane(tracer)          # recupera el plano objeto imponiendo B = 0

pupil = trace_pupil(tracer, FieldPoint(y=62.0), na_object_sine=0.3,
                    sampling=PupilSampling(kind="rings", radial=12, azimuth=72))
print(spot_data(pupil).rms_radius_um)        # 0.087 µm
```

![Layout del objetivo DUV](docs/img/duv_layout.png)

El trazado sigue el camino catadióptrico plegado: los dos espejos (M1, M2)
devuelven el haz sobre sí mismo antes del grupo de inmersión final.

![Spot diagrams del objetivo DUV](docs/img/duv_spots.png)

Spots recentrados en el centroide ponderado por área de pupila, con el primer
cero de Airy como referencia: los tres campos quedan por debajo del límite de
difracción.

Y el corte meridional del mismo trazado, con acumulación HDR: 363 rayos
exactos llenando la apertura, terminando sobre una pantalla de observación
dibujada en el plano imagen.

![Objetivo DUV en 2-D](docs/img/duv_2d.png)

## Superficies

Una superficie declara su propia geometría de una de dos formas —
**implícita** (`f_Σ(x) = 0`) o **paramétrica** (`x = P(t)`) — y eso es todo
lo que necesita para ser intersectable. Ninguna resuelve su propia
intersección: eso se hace una sola vez, genéricamente.

Vienen definidas la esfera/cónica/asférica de revolución, el segmento recto,
las cónicas en forma polar (elipse, círculo, parábola, hipérbola) y el óvalo
de Descartes. Añadir una superficie nueva es escribir su `f_Σ` y su
gradiente; la intersección, la normal y el recorte de apertura salen gratis.

### El caso que lo pone a prueba

El **óvalo de Descartes** es la superficie refractante construida para ser
perfectamente estigmática entre dos puntos conjugados. No es un ideal
inalcanzable: es una superficie concreta, y el trazador la resuelve exacta.

![Óvalo de Descartes: cono lleno convergiendo](docs/img/cartesian_oval_2d.png)

Comparada con una esfera de la **misma curvatura de vértice**, entre los
**mismos conjugados** y a la misma apertura:

![Spot diagrams: esfera vs óvalo](docs/img/stigmatic_spots.png)

La esfera deja 625.8 µm RMS de aberración esférica. El óvalo deja 6.8e-13 µm
— que no es un spot pequeño, es el suelo de la precisión de doble: 2e-17 de
la distancia objeto. El paquete no ilustra el estigmatismo, lo *verifica*, y
los ejemplos imprimen el número.

En 3-D, el mismo óvalo revuelto y recorrido por un haz denso:

![Óvalo de Descartes en 3-D](docs/img/cartesian_oval_3d.png)

## Cómo está organizado

Por la **naturaleza** de cada pieza, no por qué motor la necesitó primero.
Las dependencias forman un DAG estrictamente descendente, verificado por
análisis AST de los imports:

```
math  →  surfaces  →  optics  →  design  →  propagation  →  io  →  analysis  →  viz
```

| paquete | qué contiene |
|---|---|
| **`math`** | Vectores, transformaciones rígidas, y el solver de intersección rayo/superficie. No sabe nada de óptica. |
| **`surfaces`** | Las superficies y su contrato de descripción (implícita o paramétrica). |
| **`optics`** | La luz y sus leyes: `Ray`, `laws` (reflexión/refracción), `radiometry` (Fresnel), materiales con dispersión real (Sellmeier), emisores, instrumentos, elementos. |
| **`propagation`** | Los dos algoritmos, `sequential` y `branching`, más la capa paraxial y de pupilas. |
| **`design`** | `SurfaceRow`/`OpticalSystem`: el modelo de datos de un sistema, independiente de cualquier motor. |
| **`io`** | Lectura/escritura de prescripciones. |
| **`analysis`** | Todo lo que se mide sobre un sistema ya trazado. |
| **`viz`** | Figuras matplotlib y el visor OpenGL HDR. |

Tres decisiones que explican el resto:

**Una sola entrada para intersectar.** Un rayo es `A + λu`. Una superficie
se describe implícita o paramétricamente, y cada forma da un método:
resolver `f_Σ(A + λu) = 0`, o resolver `A + λu = P(t)`, en ambos casos para
la menor λ positiva. `intersect_ray_with_surface(rayo, superficie)` pregunta
a la superficie qué ofrece y aplica el método. Dentro del implícito el
solver toma la ruta exacta más barata que la superficie permita — tres
formas de resolver *la misma ecuación*, no tres métodos.

**El gradiente implícito hace doble trabajo:** es a la vez la derivada de
Newton y la normal de la superficie. Por eso `Surface.hit()` es una única
plantilla genérica y no código repetido en cada superficie.

**Un rayo no detecta sus propias intersecciones.** `Ray` es origen,
dirección y `point_at()`. Detectar la colisión es la descripción de la
superficie; decidir qué pasa después es del algoritmo de propagación. La
genealogía de rayos pertenece al algoritmo que la crea, no al rayo.

## Instalación

```bash
pip install -e .          # numpy, scipy, matplotlib
pip install -e ".[gl]"    # + vispy y un backend de ventana, para el visor OpenGL
```

## Ejemplos

```bash
python -m examples.lithography.duv_objective_3d --spectrum   # la imagen de arriba
python -m examples.lithography.duv_objective_2d              # el mismo objetivo, HDR 2-D
python -m examples.stigmatic_surfaces.ellipse_mirror
python -m examples.stigmatic_surfaces.cartesian_oval_refractor_2d
python -m examples.stigmatic_surfaces.cartesian_oval_refractor_3d --beam
python -m examples.telescopes.keplerian                      # y galilean, newtonian
python -m examples.imaging.single_lens_relay                 # matplotlib y OpenGL
python -m examples.imaging.interactive_doublet               # arrastra fuente/lente/pantalla
python -m examples.imaging.geometric_and_diffractive_imaging
```

Todos aceptan `--save salida.png` para renderizar sin abrir ventana;
`docs/render_images.py` regenera con eso las imágenes de este README.

Los ejemplos no solo dibujan: cada uno verifica su propia afirmación y
imprime el número. El espejo elíptico mide la distancia de cada rayo
reflejado al foco lejano (5.1e-13 µm RMS); el newtoniano comprueba que el
plegado exacto a 90° conserva el foco (1.5e-15); los telescopios *resuelven*
la separación que hace la focal efectiva infinita en el sistema real de
espesor finito, y luego confirman por trazado que dos rayos paralelos salen
paralelos.

| | |
|---|---|
| ![Espejo elíptico](docs/img/ellipse_mirror.png) | ![Newtoniano](docs/img/newtonian.png) |
| **Espejo elíptico** — todo rayo que sale de un foco pasa por el otro. | **Newtoniano** — primario parabólico, secundario plano a 45°, haz anular con obstrucción central. |
| ![Kepleriano](docs/img/keplerian.png) | ![Galileano](docs/img/galilean.png) |
| **Kepleriano** — imagen intermedia real (punto verde). **−3.989×**, invertida. | **Galileano** — ocular negativo, tubo corto. **+4.069×**, derecha. |

Los notebooks en `examples/notebooks/` reproducen dos objetivos de patente
completos: `us7557996_objective.ipynb` (el DUV de inmersión, réplica dígito a
dígito del análisis de referencia) y `euv_six_mirror.ipynb` (objetivo EUV de
seis espejos, US 7,151,592, NA 0.22, λ=13.4 nm — con refit de asféricas por
mínimos cuadrados, frente de onda por eikonal e imagen de Abbe).

## Tests

```bash
pytest                                    # 144 tests
xvfb-run -a pytest                        # 148, incluidos los de contexto OpenGL
```

Las verdades de referencia son analíticas donde existen: conjugados de
parábola y elipsoide a precisión de máquina, magnificación paraxial del
objetivo DUV (+0.2500000135), NA de imagen 1.1977, RMS de Zernike (0.5089 nm
tras quitar tilt y desenfoque), primer cero de Airy, colapso de contraste en
el límite de coherencia parcial, y linealidad aditiva del framebuffer HDR.
Los dos métodos de intersección se contrastan entre sí sobre el óvalo de
Descartes —la única superficie que ofrece ambas descripciones— exigiendo que
caigan en el mismo punto para el mismo rayo.

## Estructura del repositorio

```
raytracer/     el paquete
data/          prescripciones de ejemplo en CSV -- son datos, no código
examples/      demos ejecutables por categoría + notebooks de réplica de patentes
docs/          imágenes del README y el script que las regenera
reference/     notebooks y módulo de referencia originales
tests/         verdades analíticas + regresiones de sistema completo
```
