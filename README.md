# raytracer

Trazador de rayos óptico en Python. Propaga luz **exactamente** —resolviendo
la ecuación de cada superficie, sin aproximación paraxial— a través de
sistemas ópticos reales, y mide lo que sale: aberraciones, distorsión,
frente de onda, PSF, formación de imagen.

**10.000 líneas · 8 paquetes en cadena estricta · 163 tests contra verdades
analíticas · dos objetivos de patente replicados dígito a dígito**

![Objetivo DUV US7557996, tres longitudes de onda](docs/img/duv_3d_spectrum.png)

Arriba: el objetivo litográfico DUV de la patente US 7,557,996 —48
superficies, 12 asféricas, NA 1.2 en inmersión— recorrido por tres haces de
distinta longitud de onda. Se separan en el camino catadióptrico plegado y
vuelven a sumarse a blanco en la imagen. Lo dibuja el visor OpenGL, que
acumula los rayos aditivamente en un framebuffer de coma flotante: donde se
superponen miles de rayos la intensidad *suma*, y un tone-mapping
auto-expuesto revela la estructura del haz dentro del rango dinámico.

## Qué hace

Dado un sistema óptico —una prescripción de superficies, o un puñado de
lentes y espejos colocados en el espacio— el paquete responde a tres tipos
de pregunta.

**¿Por dónde va la luz?** Cada rayo se intersecta con cada superficie
resolviendo la ecuación de esa superficie. Para una cónica hay forma
cerrada; para un perfil asférico, Newton sobre `f_Σ(A + λu) = 0` desde una
semilla que aporta la propia superficie; para el resto, un barrido con
bracket-and-bisect. La refracción y la reflexión se aplican en forma
vectorial completa, y la reflexión total interna se detecta rayo a rayo.

**¿Qué tan bien forma imagen?** Spot diagrams ponderados por área de pupila,
ray fans transversales, distorsión por rayo principal (la convención de
Zemax/OpticStudio), coeficientes de Seidel deducidos del frente de onda
real, expansión de Zernike ANSI/OSA, aberración cromática axial y lateral,
frente de onda por eikonal, PSF escalar e imagen aérea parcialmente
coherente por el método de Abbe.

**¿Cómo se ve?** Dos backends sobre una misma escena: figuras matplotlib
para el análisis, y el visor OpenGL HDR para el trazado. La escena es una
lista de ítems neutrales —segmentos de rayo, perfiles de superficie,
cuerpos de lente, pantallas, marcadores— y cada backend la consume entera.

## Los dos motores de propagación

Los dos motores se distinguen por lo que hace un rayo al llegar a una
superficie.

**Secuencial** — cada rayo atraviesa cada superficie en orden fijo, un solo
camino determinista. Es el modelo de diseño óptico clásico, y el orden fijo
es lo que permite vectorizar: los N rayos avanzan juntos en numpy, con un
`TraceStatus` por rayo que marca fallo de intersección, TIR o viñeteo. Sobre
él se apoyan la capa paraxial (matrices ABCD, recuperación de conjugados),
la resolución del rayo principal y el muestreo de pupila.

**Ramificado** — en cada colisión el rayo se divide en su hijo reflejado *y*
su hijo refractado, y el recorrido en anchura genera el árbol completo. Con
él se ven las cáusticas, la luz parásita y las reflexiones múltiples. Lleva
OPL e intensidad acumuladas, con división de Fresnel opcional.

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

## El mapa objeto → imagen

Un sistema ideal manda cada punto objeto a `magnificación × objeto`. La
distorsión es la desviación del mapa real respecto de esa recta.
`distortion_grid` la mide trazando una malla cuadrada de puntos de campo,
uno por rayo principal —el rayo que pasa por el centro del diafragma, la
referencia estándar, inmune al coma y al viñeteo que arrastran el centroide
de un spot—, y devuelve dónde aterriza cada punto en el plano imagen.

```python
from pathlib import Path

import numpy as np
from raytracer.analysis import distortion_grid
from raytracer.design import OpticalSystem
from raytracer.propagation import SequentialTracer, solve_object_plane
from raytracer.viz import plots

tracer = SequentialTracer(
    OpticalSystem.from_prescription(Path("data/cooke_triplet_prescription.csv"))
)
conjugate = solve_object_plane(tracer)
half = abs(conjugate.object_z) * np.tan(np.deg2rad(10.0))   # ±10° de semicampo

grid = distortion_grid(tracer, magnification=conjugate.magnification,
                       half_field=half, n=11)

print(grid.valid_fraction)                     # 1.0  -- 121/121 sin viñetear
print(grid.max_distortion_um)                  # 25.81  (µm)
print(grid.max_relative_distortion_percent)    # 0.2074 (% de la altura de imagen)

plots.distortion_grid_figure(grid, exaggeration=25.0)
```

![Mallas de distorsión de tres sistemas](docs/img/distortion_grids.png)

Cada panel superpone dos mallas: la ideal, en gris discontinuo, y la
trazada, en azul. El dial `exaggeration` amplifica la desviación de cada
punto alrededor de su posición ideal, y cada panel usa el suyo: entre el
Doble Gauss y el objetivo DUV hay un factor 4700 en desviación máxima, y un
multiplicador común dejaría dos paneles ilegibles. El título de cada panel
lleva la magnitud verdadera, sin amplificar.

El triplete Cooke y el Doble Gauss dibujan corsé (**pincushion**): la
desviación radial es positiva en todo el campo y crece con él, hasta 25,8 µm
y 206 µm en la esquina — 0,207 % y 0,388 % de la altura de imagen.

El objetivo DUV, a ×2000, muestra el residuo de un diseño ya corregido: la
desviación llega a +43,9 nm hacia r ≈ 5,6 mm (7,8 partes por millón, el
máximo relativo), cruza el cero cerca de r ≈ 11,5 mm y termina en −2,2 nm en
las esquinas del campo. El signo cambia dentro del campo, que es la firma de
una distorsión balanceada en el diseño, y todo ello sobre un campo imagen de
25 mm de diámetro.

Las cuatro esquinas marcadas con una `x` roja son puntos cuyo rayo principal
vieta: la malla llega a 50 mm de semicampo objeto, y sus esquinas quedan a
50·√2 = 70,7 mm, más allá del radio útil de ~68 mm del diseño. Esos puntos
quedan como `NaN`, rompen la línea que atravesarían y quedan contados en
`grid.valid_fraction`.

Para la curva clásica de distorsión contra campo, `chief_ray_distortion`
hace el barrido 1-D con un solo rayo principal por punto, en mm de altura
objeto o en grados de campo.

## Superficies

Una superficie declara su propia geometría de una de dos formas —
**implícita** (`f_Σ(x) = 0` y `∇f_Σ`) o **paramétrica** (`x = P(t)` y una
semilla) — y con eso ya es intersectable. La intersección se resuelve una
sola vez, genéricamente, en `math.intersections`.

Vienen definidas la esfera/cónica/asférica de revolución, el segmento recto,
las cónicas en forma polar (elipse, círculo, parábola, hipérbola) y el óvalo
de Descartes. Añadir una superficie nueva es escribir su `f_Σ` y su
gradiente; la intersección, la normal y el recorte de apertura salen gratis.

### El caso que lo pone a prueba

El **óvalo de Descartes** es la superficie refractante construida para ser
perfectamente estigmática entre dos puntos conjugados. Es una superficie
concreta, con forma cerrada, y el trazador la resuelve exacta.

![Óvalo de Descartes: cono lleno convergiendo](docs/img/cartesian_oval_2d.png)

Comparada con una esfera de la **misma curvatura de vértice**, entre los
**mismos conjugados** y a la misma apertura:

![Spot diagrams: esfera vs óvalo](docs/img/stigmatic_spots.png)

La esfera deja 625,8 µm RMS de aberración esférica. El óvalo deja
6,8e-13 µm, que es el suelo de la precisión de doble: 2e-17 de la distancia
objeto. El paquete *verifica* el estigmatismo, y los ejemplos imprimen el
número.

En 3-D, el mismo óvalo revuelto y recorrido por un haz denso:

![Óvalo de Descartes en 3-D](docs/img/cartesian_oval_3d.png)

## La ontología del paquete

Los paquetes están cortados por la **naturaleza** de cada pieza, y las
dependencias forman una cadena estrictamente descendente. Cada import a
nivel de módulo apunta a un paquete anterior en el orden:

```
math  →  surfaces  →  optics  →  design  →  propagation  →  io  →  analysis  →  viz
```

Esas son las aristas reales del grafo, leídas del AST de cada módulo:

| paquete | importa a nivel de módulo | líneas |
|---|---|---|
| **`math`** | — | 463 |
| **`surfaces`** | `math` | 1136 |
| **`optics`** | `math`, `surfaces` | 947 |
| **`design`** | `surfaces`, `optics` | 378 |
| **`propagation`** | `math`, `surfaces`, `optics`, `design` | 1094 |
| **`io`** | `surfaces`, `optics`, `design` | 201 |
| **`analysis`** | `design`, `propagation` | 1928 |
| **`viz`** | `surfaces`, `optics`, `design`, `propagation`, `analysis` | 3863 |

`tests/test_architecture.py` recorre el AST de cada módulo y lo comprueba, de
modo que la cadena es una afirmación verificada en cada `pytest`.

### Qué es cada parte

**`math` — geometría sin óptica.** Vectores y normalización,
transformaciones rígidas (`RigidTransform`), y los solvers de intersección
rayo/superficie. Aquí vive la aritmética que valdría igual para cualquier
trazado geométrico: el paquete entero desconoce qué es un índice de
refracción.

**`surfaces` — las formas que un rayo puede encontrar.** Una superficie es
la frontera entre dos medios: posee su forma, su marco local, su apertura
libre y los índices a cada lado. Declara cómo describirse a sí misma
(implícita o paramétrica) y ahí termina su responsabilidad. `Surface.hit()`
es una plantilla genérica que sirve a todas.

**`optics` — la luz y las leyes que obedece.** `Ray` (origen, dirección,
longitud de onda, `point_at`), `laws` (reflexión y refracción vectoriales,
en versión escalar y por lotes), `radiometry` (coeficientes de Fresnel s y
p), `materials` (índice constante, Abbe, Sellmeier medido, y un catálogo de
vidrios), `sources` (puntual, paralela, de imagen), `instruments` (una
pantalla es un instrumento, que es una superficie) y `elements` (`Lens`,
`Mirror` ya montados).

**`design` — el modelo de datos de un sistema.** `SurfaceRow` es una fila de
prescripción: radio, espesor al siguiente, material, semidiámetro, constante
cónica, coeficientes asféricos, tipo de superficie. `OpticalSystem` es la
secuencia completa, con los vértices acumulados, los índices a cada lado de
cada superficie, el índice del diafragma y los planos objeto e imagen.
Describe **qué es** un sistema y contiene cero lógica de propagación: un
diseño existe antes de que exista un motor que lo trace.

**`propagation` — los algoritmos.** El único lugar que gobierna qué hace un
rayo después de chocar. `sequential` vectoriza N rayos por M superficies en
orden fijo; `branching` construye el árbol de reflejados y refractados;
`paraxial` aporta `ParaxialModel`, `solve_object_plane` (recupera el plano
objeto imponiendo B = 0 en la matriz del sistema) y `differential_conjugates`;
`fields` aporta `FieldPoint`, el solver 2-D del rayo principal
(`chief_ray_slopes`), el muestreo de pupila y `trace_pupil`. Estas dos
últimas viven aquí porque resolver un rayo principal es trazar.

**`io` — serialización.** Lectura y escritura de prescripciones, con los
formatos en un registro (`FORMAT_READERS`, `FORMAT_WRITERS`). El formato de
archivo es una naturaleza distinta de lo que un sistema es.

**`analysis` — lo que se mide sobre un sistema ya trazado.** Partido en dos
por el objeto de la medida: `imaging` mira la imagen (spots, PSF escalar,
imagen aérea de Abbe, contraste) y `aberrations` mira el error de rayo y de
frente de onda (métricas de campo, distorsión, Seidel, cromática, eikonal,
Zernike, ray fans, estigmatismo). Los dos re-exportan desde
`raytracer.analysis`, derivando la lista de sus propios `__all__`.

**`viz` — presentación.** `plots` para las figuras matplotlib, `scene` para
la escena neutral de ítems, `gl/` para el visor OpenGL (cámara, shaders,
renderers, ventana) e `interactive` para arrastrar fuente, lente y pantalla
en vivo. Es el paquete más grande y el único del que nada depende.

### Tres decisiones que explican el resto

**Una sola entrada para intersectar.** Un rayo es `A + λu`. Una superficie
se describe implícita o paramétricamente, y cada forma da un método:
resolver `f_Σ(A + λu) = 0`, o resolver `A + λu = P(t)`, en ambos casos para
la menor λ positiva. `intersect_ray_with_surface(rayo, superficie)` pregunta
a la superficie qué ofrece y aplica el método. Dentro del implícito, el
solver toma la ruta exacta más barata que la superficie permita: forma
cerrada si `f_Σ` es cuadrática, Newton si la superficie da semilla,
bracket-and-bisect en el caso general. Las tres resuelven *la misma
ecuación*.

**El gradiente implícito hace doble trabajo.** `∇f_Σ` es a la vez la
derivada que necesita Newton y la normal de la superficie. Declarar `f_Σ` y
su gradiente compra intersección y normales de una vez, y por eso
`Surface.hit()` es una única plantilla genérica.

**Un rayo lleva su estado y nada más.** `Ray` es origen, dirección,
longitud de onda y `point_at()`. Detectar la colisión corresponde a la
descripción de la superficie; decidir qué pasa después corresponde al
algoritmo de propagación; la genealogía de rayos pertenece al algoritmo que
la crea.

Las dos únicas aristas que cruzan la cadena hacia arriba están diferidas a
propósito, y el test las fija: `OpticalSystem.from_prescription` alcanza el
registro de lectores de `io` desde dentro del método, de modo que el modelo
de datos sigue sin depender de ningún formato de archivo; y
`surfaces/surface.py` anota un `Ray` bajo `if TYPE_CHECKING`, que se borra
en tiempo de ejecución, de modo que una superficie es intersectable sin
saber qué es un rayo.

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

Cada ejemplo verifica su propia afirmación e imprime el número. El espejo
elíptico mide la distancia de cada rayo reflejado al foco lejano
(5,1e-13 µm RMS); el newtoniano comprueba que el plegado exacto a 90°
conserva el foco (1,5e-15); los telescopios *resuelven* la separación que
hace infinita la focal efectiva en el sistema real de espesor finito, y
luego confirman por trazado que dos rayos paralelos salen paralelos.

| | |
|---|---|
| ![Espejo elíptico](docs/img/ellipse_mirror.png) | ![Newtoniano](docs/img/newtonian.png) |
| **Espejo elíptico** — todo rayo que sale de un foco pasa por el otro. | **Newtoniano** — primario parabólico, secundario plano a 45°, haz anular con obstrucción central. |
| ![Kepleriano](docs/img/keplerian.png) | ![Galileano](docs/img/galilean.png) |
| **Kepleriano** — imagen intermedia real (punto verde). **−3,989×**, invertida. | **Galileano** — ocular negativo, tubo corto. **+4,069×**, derecha. |

Los notebooks en `examples/notebooks/` reproducen dos objetivos de patente
completos: `us7557996_objective.ipynb` (el DUV de inmersión, réplica dígito a
dígito del análisis de referencia) y `euv_six_mirror.ipynb` (objetivo EUV de
seis espejos, US 7,151,592, NA 0.22, λ=13.4 nm — con refit de asféricas por
mínimos cuadrados, frente de onda por eikonal e imagen de Abbe). Los dos
incluyen su malla de distorsión, la del EUV centrada en su campo anular de
trabajo, a 120 mm del eje.

## Tests

```bash
pytest                                    # 159 tests
xvfb-run -a pytest                        # 163, incluidos los de contexto OpenGL
```

Las verdades de referencia son analíticas donde existen: conjugados de
parábola y elipsoide a precisión de máquina, magnificación paraxial del
objetivo DUV (+0.2500000135), NA de imagen 1.1977, RMS de Zernike (0.5089 nm
tras quitar tilt y desenfoque), primer cero de Airy, colapso de contraste en
el límite de coherencia parcial, y linealidad aditiva del framebuffer HDR.
Los dos métodos de intersección se contrastan entre sí sobre el óvalo de
Descartes —la única superficie que ofrece ambas descripciones— exigiendo que
caigan en el mismo punto para el mismo rayo. La cadena de dependencias entre
paquetes se comprueba leyendo el AST de cada módulo.

## Estructura del repositorio

```
raytracer/     el paquete
data/          prescripciones de ejemplo en CSV -- datos puros
examples/      demos ejecutables por categoría + notebooks de réplica de patentes
docs/          imágenes del README y el script que las regenera
reference/     notebooks y módulo de referencia originales
tests/         verdades analíticas + regresiones de sistema completo
```
