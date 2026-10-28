# Tercer orden, focos parabasales y distorsión (0.6)

## Qué cambió

`seidel_coefficients` calcula ahora las cinco sumas clásicas por superficie,
a partir de dos rayos paraxiales. La implementación anterior ajustaba modos de
Zernike a haces de apertura finita, mezclaba órdenes superiores y no extraía
la distorsión. Se conserva con el nombre explícito
`fitted_low_order_coefficients` en `fitted_aberrations.py`, fuera de la
validación de Seidel. Los cuatro notebooks que la utilizaban se han marcado
como históricos y sus llamadas se han migrado; sus salidas anteriores no se
presentan como resultados recalculados.

Se distinguen tres objetos físicos:

1. Las sumas de Seidel son coeficientes asintóticos alrededor del eje.
2. Los focos parabasales proceden de haces infinitesimales alrededor del rayo
   principal de cada campo; admiten un rayo principal no paraxial.
3. El mejor foco RMS minimiza el tamaño de un haz de apertura finita y depende
   del muestreo, los pesos y el viñeteo.

No son intercambiables. `field_metrics` conserva sus claves históricas de foco
RMS e incorpora claves `parabasal_*`; `aberrations_figure` utiliza estas últimas
para sus curvas de campo y astigmatismo.

## Rayos de referencia y unidades

Todas las longitudes geométricas y las sumas se expresan en mm. Los índices
físicos son positivos; para el cálculo paraxial se utiliza un índice algebraico
con signo que cambia en cada reflexión. Las coordenadas z y los espesores son
los del sistema, también después de reflejarse. Esta convención permite tratar
espejos sin alterar artificialmente el signo del eje. Una colocación rígida global
está permitida; los análisis requieren superficies y detector localmente centrados.

En la superficie i, y_i y u_i son la altura y pendiente dy/dz del rayo marginal
axial incidente; ȳ_i y ū_i corresponden al rayo principal paraxial. El marginal
llega a la altura a en el stop, donde a es el radio de normalización declarado;
el principal llega al centro del stop. Se construyen con matrices ABCD, sin
apuntado de rayos exactos ni ajuste sobre una apertura finita. El estado es
(y, n u), con n el índice firmado. Un stop conjugado con el objeto produce una
normalización singular y se rechaza.

Para campo finito, el dato es la altura objeto. Para objeto en infinito, el dato
es el ángulo θ y la pendiente de entrada es tan θ. Escalar el campo por h significa
escalar la altura o tan θ; no significa escalar exactamente θ. El radio a puede
especificarse explícitamente o tomarse del radio exterior de un stop circular.
Las sumas no integran una pupila ni predicen la transmisión de una apertura.

## Sumandos de Seidel

Sean n_i y n'_i los índices firmados antes y después de la superficie, c_i su
curvatura de vértice y u'_i, ū'_i las pendientes transmitidas o reflejadas.
Definimos las incidencias reducidas A_i, Ā_i, el invariante de Lagrange H,
el salto de pendiente reducida D_i y la cantidad P_i:

\[
A_i=n_i(u_i+c_i y_i),\qquad
\bar A_i=n_i(\bar u_i+c_i\bar y_i),
\]
\[
H=n_i(y_i\bar u_i-\bar y_i u_i),\qquad
D_i=\frac{u'_i}{n'_i}-\frac{u_i}{n_i},\qquad
P_i=c_i\left(\frac1{n'_i}-\frac1{n_i}\right).
\]

H es constante a lo largo del sistema. Para la esfera de referencia con la misma
curvatura de vértice, las contribuciones a esférica, coma, astigmatismo, Petzval
y distorsión, respectivamente, son

\[
S_{I,i}=-A_i^2y_iD_i,\qquad
S_{II,i}=-A_i\bar A_i y_iD_i,\qquad
S_{III,i}=-\bar A_i^2y_iD_i,
\]
\[
S_{IV,i}=-H^2P_i,
\]
\[
S_{V,i}=-\bar A_i\left[
\bar A_i^2y_i\left(\frac1{n_i'^2}-\frac1{n_i^2}\right)
-(H+\bar A_i y_i)\bar y_iP_i\right].
\]

Esta forma de S_V evita dividir por A_i, que puede anularse sin que el sistema
sea singular. La suma total de cada aberración es la suma de sus contribuciones.
El informe conserva tanto los sumandos como los datos de los dos rayos para
poder auditar las convenciones.

Para una cónica de constante K_i con término asférico a_4,i r⁴, la diferencia
cuártica de sag respecto de la esfera es G_i r⁴, con

\[
G_i=K_i c_i^3/8+a_{4,i}.
\]

La corrección al vector de cinco sumandos es

\[
8G_i(n'_i-n_i)
\left(y_i^4,\;y_i^3\bar y_i,\;y_i^2\bar y_i^2,\;0,\;y_i\bar y_i^3\right).
\]

No se divide por y_i; un marginal que cruza el eje no genera una singularidad
artificial. Los términos r⁶ y superiores no contribuyen a tercer orden.
Los óvalos de Descartes se rechazan aquí hasta incorporar y validar su expansión
de vértice. El trazado exacto de esos perfiles sigue disponible.

## Predicción transversal y curvaturas geométricas

Sean X e Y las coordenadas de pupila divididas por a, h la escala de campo,
ρ²=X²+Y², y n', u' el índice firmado y pendiente marginal a la salida del
último elemento. Los errores transversales cúbicos respecto de la imagen
Gaussiana, en el plano Gaussiano, son

\[
\Delta x=\frac{S_I X\rho^2+2S_{II}hXY+(S_{III}+S_{IV})h^2X}{2n'u'},
\]
\[
\Delta y=\frac{S_I Y\rho^2+S_{II}h(X^2+3Y^2)
 +(3S_{III}+S_{IV})h^2Y+S_Vh^3}{2n'u'}.
\]

Por ello la distorsión de tercer orden es S_V h³/(2n'u'). No se resta el rayo
principal antes de comparar: hacerlo eliminaría precisamente ese término.
Un detector desenfocado no cambia las sumas, pero estas expresiones se refieren
al plano Gaussiano. El informe devuelve ambos z para evitar una comparación
entre planos distintos. Los sistemas con imagen afocal se rechazan.

Las curvaturas C_T, C_S y C_P se definen **geométricamente** mediante
z(y_G)=z_G+C y_G²/2, donde y_G es la altura de imagen Gaussiana. En nuestra
convención de z firmado:

\[
C_T=-\frac{n'}{H^2}(3S_{III}+S_{IV}),\qquad
C_S=-\frac{n'}{H^2}(S_{III}+S_{IV}),\qquad
C_P=n'\sum_i P_i.
\]

El signo debe compararse con esta definición de sag, no con nombres de columnas
de otro programa: el convertidor de curvaturas de RayOptics 0.9.8 usa el signo
opuesto. La validación externa compara sus **sumas**, y los signos geométricos
se contrastan además con trazados exactos. C_P se obtiene directamente incluso
para campo nulo; C_T y C_S se marcan indefinidos si H=0.

## Foco parabasal y distorsión exacta

`parabasal_focus` apunta rayos a ±δ y ±δ/2 alrededor del centro del stop,
en las direcciones tangencial y sagital. Para cada dirección, p es la coordenada
transversal sobre el detector y q la pendiente de salida en el mismo marco.
La intersección diferencial con el rayo principal tiene desplazamiento

\[
\Delta z=-\frac{\partial p/\partial s}{\partial q/\partial s},
\]

donde s es la coordenada física del objetivo en el stop. Se devuelve el resultado
con δ/2 y la diferencia entre pasos. Esta diferencia diagnostica convergencia,
no certifica una cota de error. Se rechazan apuntados fallidos, rayos que no llegan
al detector y haces diferenciales paralelos. Los focos son prolongaciones en el
medio final; no representan propagación a través de nuevas interfaces.

Las sondas geométricas ignoran aperturas; `chief_transmitted` informa por separado
si el principal atravesaría físicamente las aperturas. Esto permite estudiar una
pupila anular sin confundir un rayo de referencia obstruido con luz transmitida.
Para campos 2D se usan la dirección radial y su perpendicular; los ejes x/y del
sistema no se etiquetan automáticamente como sagital/tangencial para todo campo.

`distortion_map` usa por defecto el mapa paraxial del principal al detector actual.
En infinito es rectilíneo, proporcional a las tangentes de los ángulos componentes.
La referencia `f_theta` usa en cambio el ángulo polar total, con la misma escala
axial. En un sistema en aire al foco Gaussiano son las referencias usuales f tan θ
y f θ. Fuera de ese caso la escala la define ABCD, no se introduce f por suposición.
La distorsión relativa axial queda indefinida (`NaN`/`null`), mientras que el
error absoluto puede ser cero. El centroide ponderado por área transmitida es
opcional y se conserva como una medida diferente, sensible a coma y viñeteo.

## Referencias independientes y criterios

`reference/generate_third_order_reference.py` utiliza RayOptics 0.9.8 y **no
importa raytracer**. Construye cinco sistemas de índices constantes explícitos:
lente esférica en infinito; lente con objeto finito y stop posterior; lente
asférica con objeto finito y stop posterior; espejo esférico; espejo parabólico.
Calcula sus rayos paraxiales con RayOptics, las contribuciones a las cinco sumas
y los focos parabasales mediante apuntado y trazado exacto independientes.
Los datos se guardan en `tests/reference/rayoptics_third_order.json`.

| Comparación | Resultado máximo observado | Tolerancia de prueba |
|---|---:|---:|
| 40 coeficientes por superficie (8 superficies × 5) | 3.47e-18 mm | 2e-14 mm absoluta |
| Focos parabasales tangencial/sagital | 2.98e-10 mm | 2e-8 mm absoluta |
| Posiciones del principal | 4.45e-15 mm | 2e-10 mm absoluta |
| Reducción del residuo transversal al dividir escala por 2 | 32.006–32.137 | 32 con rtol=0.015 |
| Reducción del residuo de foco de campo | 15.97–16.14 | 16 con rtol=0.05 |

La escala conjunta de campo y pupila toma 1/2, 1/4 y 1/8. Una predicción cúbica
correcta deja un residuo O(ε⁵), que disminuye 2⁵=32 veces al dividir ε por dos.
Las curvaturas predicen un desplazamiento cuadrático con residuo O(ε⁴), que
se reduce 2⁴=16 veces. Las tolerancias admiten órdenes superiores y la
cancelación numérica de diferencias pequeñas; no se interpretan como precisión
física del vidrio ni como equivalencia universal entre programas.

El contrato 8 añade una solución analítica independiente: espejo esférico de
radio −R, R>0, stop en el vértice, incidencia paralela a ángulo θ y detector
z=−R/2. Las ecuaciones de Coddington dan

\[
z_T=-\frac R2\cos^2\theta,\qquad z_S=-\frac R2,
\qquad \Delta z_T=\frac R2\sin^2\theta.
\]

Con R=100 mm y θ=10°, Δz_T=1.507684480 mm y Δz_S=0. El rayo principal llega
a y=(R/2)tan θ: su distorsión rectilínea es cero, pero respecto de f θ vale
(R/2)(tan θ−θ), con θ expresado en radianes. Así se verifica que cambiar de
referencia modifica la distorsión sin cambiar el trazado.

Se conservan **exactamente diez pruebas**. Los contratos 8 y 10 incorporan estos
casos; no se introducen pruebas ocultas mediante parametrización.

## Reproducción y API

```bash
python -m examples.aberrations.third_order_validation
```

El ejemplo usa el doblete analítico N-BK7/N-F2 de la etapa 0.5 a 0.58756 µm,
stop de radio 5 mm y objeto en infinito. Genera una figura y
`docs/third_order_results.json`, con entradas, curvas, contribuciones y residuos.

```python
from examples.aberrations.chromatic_sensitivity import doublet
from raytracer.analysis.aberrations import (
    distortion_map, parabasal_focus, seidel_coefficients,
)
from raytracer.propagation import FieldPoint, SequentialTracer

system = doublet()
field = FieldPoint.angle(y_deg=3)
ledger = seidel_coefficients(system, field)
print(ledger.named_sums_mm)
print(ledger.field_curvatures_per_mm)
focus = parabasal_focus(SequentialTracer(system), field)
mapping = distortion_map(SequentialTracer(system), [field])
```

Para regenerar los datos externos, en el entorno separado descrito en la guía
de fundamentos:

```bash
.venv-reference/bin/python reference/generate_third_order_reference.py
```

En el doblete a 3°, (S_I,S_II,S_III,S_IV,S_V) en µm son aproximadamente
(0.2929775, 0.6616569, 0.6892686, 0.4905885, 0.0107631).
Las curvaturas (C_T,C_S,C_P) son (−0.03725941, −0.01718296, −0.00714473) mm⁻¹.
A 8°, la desviación del principal es −2.04072 µm y la del centroide −12.88744 µm:
la diferencia no puede atribuirse toda a distorsión. La fracción geométrica
transmitida del barrido está entre 0.96523 y 1. El centroide usa 8×32 puntos de
cuadratura; cerca de un borde de viñeteo sus cambios dependen del muestreo.

## Fuentes y alcance pendiente

- W. T. Welford, *Aberrations of Optical Systems*, Adam Hilger, 1986:
  teoría clásica de sumas por superficie y rayos paraxiales.
- M. J. Hayford, [RayOptics](https://github.com/mjhoptics/ray-optics), versión
  0.9.8: módulos `parax.firstorder`, `parax.thirdorder` y `raytr.raytrace`.
  Se fija esta versión; no se sustituye silenciosamente por la más reciente.
- Ansys, [Seidel Coefficients](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v242/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Seidel_Coefficients.html):
  alcance de las sumas clásicas y distinción entre coeficientes y conversiones.
- Ansys, [Field Curvature and Distortion](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v26102/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Field_Curvature_and_Distortion.html):
  definición parabasal de focos y referencias rectilínea y angular de distorsión.

Este desarrollo no calcula propagación ondulatoria ni optimiza prescripciones.
Quedan fuera las aberraciones nodales de sistemas descentrados, óptica afocal,
perfiles sin expansión admitida, referencias calibradas por ajuste de escala y
los modelos industriales de distorsión de imagen televisiva. El dominio útil de
tercer orden depende del sistema: se debe comparar con el trazado exacto al
campo y apertura de interés. También siguen pendientes pupilas de entrada/salida,
telecentricidad y una caracterización más amplia de convergencia de muestreo.
