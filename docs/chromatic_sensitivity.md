# Cromática, sensibilidad y trazabilidad (0.5)

## Modelo y convenciones

La geometría usa mm, las longitudes de onda µm y las inclinaciones grados.
Los errores transversales se miden en los ejes locales del detector. Todas las
longitudes de onda comparten su plano físico; enfocar cada color por separado
ocultaría parte de la aberración cromática. El color axial usa el conjugado
ABCD con objeto fijo, incluyendo infinito, y requiere un sistema centrado.
No mezcla aberración esférica de apertura finita con color paraxial.

La referencia espectral debe aparecer una vez en las longitudes de onda
(tolerancia absoluta 1e-12 µm). No se escoge silenciosamente el vecino más cercano.
`chromatic_spots` admite sistemas colocados en 3D y un `SequentialTracer` para
conservar sus opciones de caminos virtuales y tolerancias numéricas.

## Pesos y estadísticas

Sean s_j los pesos espectrales normalizados, a_ij los pesos de área originales
de pupila y v_ij una máscara de supervivencia. La fracción geométrica transmitida es

\[
T=\sum_{j,i}s_j a_{ij}v_{ij},\qquad
w_{ij}=s_j a_{ij}v_{ij}/T.
\]

Para posiciones p_ij sobre el detector, el centroide y el radio RMS son

\[
\bar p=\sum w_{ij}p_{ij},\qquad
\mathrm{RMS}_{centroide}=\sqrt{\sum w_{ij}\|p_{ij}-\bar p\|^2}.
\]

El RMS respecto de una referencia p_ref sustituye el centroide por ese punto.
Son medidas diferentes: un haz perfectamente concentrado pero desplazado tiene
RMS de centroide nulo y error respecto de la referencia no nulo.
`ChromaticSpots.rms_radius_um` y `polychromatic_rms_um` usan el rayo principal
de la longitud de onda de referencia; `centroid_rms_um` usa el centroide conjunto.
`GeometricEvaluator.reference_rms_mm` usa el rayo principal nominal de la
**primera longitud de onda listada**, cuya coordenada transversal se conserva
incluso en el diagnóstico de reenfoque. Esta referencia se registra en el informe.

Los pesos espectrales son potencias relativas de bandas discretas. Si se parte
de una densidad espectral, hay que incluir los anchos de integración y, si
procede, la respuesta del detector. No son números de rayos. La medida de pupila
es área del stop nominal, no un modelo radiométrico universal de la fuente.
Fresnel, absorción, dispersión por rugosidad y coherencia quedan fuera de T.

## Materiales con procedencia

`MaterialMetadata` conserva fuente, intervalo espectral declarado, temperatura
de referencia, convención del índice y notas. Los intervalos se comprueban en
cada evaluación; su ausencia significa validez desconocida. La temperatura es
informativa: no implementa dn/dT ni cambia la dispersión automáticamente.

Se contrastan ocho índices tabulados de dos fichas primarias SCHOTT:

| Vidrio | n(F), 486.13 nm | n(d), 587.56 nm | n(C), 656.27 nm | n(t), 1014 nm |
|---|---:|---:|---:|---:|
| N-BK7 | 1.52238 | 1.51680 | 1.51432 | 1.50731 |
| N-F2 | 1.63208 | 1.62005 | 1.61506 | 1.60261 |

Fuentes: [ficha N-BK7](https://media.schott.com/api/public/content/41e799d0bf874807a0bb8e702fbb75b5?v=54856406)
y [ficha N-F2](https://media.schott.com/api/public/content/061f3156c83a44ed9220770b0f65a869?v=d69b35e0).
Las fichas redondean índices a cinco decimales y algunas longitudes de onda a
0.1 nm. La tolerancia absoluta de contraste es 6e-6; el máximo residuo observado
es 3.515e-6. Esto verifica consistencia con esas tablas, no caracteriza una colada.
Los intervalos declarados N-BK7 [0.3126, 2.3254] µm y N-F2 [0.4047, 2.3254] µm
están acotados por sus índices tabulados; no certifican cada longitud intermedia
ni una transmisión útil en todo el intervalo.

Los demás vidrios indican la revisión de fuente primaria pendiente. Los nombres
históricos SK16/SK4 conservan sustitutos N-SK16/N-SK4 identificados como tales;
F4 es una aproximación Abbe. Los índices DUV constantes se restringen a
0.193368 µm. El CSV de resinas conserva coeficientes históricos con procedencia
no verificada explícita. No se presenta como catálogo de mediciones certificado.
`IndexOffsetMaterial` añade δn constante a una dispersión base para sensibilidad;
no representa por sí solo una variación de composición o de número de Abbe.

## Desalineación y sensibilidad

`Perturbation.apply` devuelve una copia; no modifica el nominal. Los índices de
superficie empiezan en cero. Se admiten traslaciones e inclinaciones de grupos,
espesor de una fila, radio de una superficie curva e índice del medio transmitido.
Las rotaciones usan ejes fijos del sistema. El pivote predeterminado es el vértice
actual de la primera superficie seleccionada: para secuencias combinadas conviene
especificar `pivot_mm`. Las transformaciones se aplican en el orden declarado.

El evaluador apunta al stop **una vez en el nominal para cada campo y color**.
Conserva después los mismos orígenes, direcciones y pesos incidentes: reapuntar
tras un descentramiento cambiaría la iluminación y podría esconder viñeteo.
El detector conserva su posición y orientación nominales, incluso al modificar
el último espesor. En cambio, `chromatic_spots` apunta de nuevo al sistema que
se le entrega; sus funciones responden a preguntas distintas.

La sensibilidad de una métrica M se calcula por diferencias centrales:

\[
D_h=\frac{M(+h)-M(-h)}{2h},\qquad
D_{h/2}=\frac{M(+h/2)-M(-h/2)}{h}.
\]

Se devuelve D_(h/2), los valores extremos, |D_(h/2)-D_h|, la discrepancia entre
pendientes laterales y si cambian los rayos supervivientes. La diferencia entre
pasos diagnostica convergencia; no es una cota certificada. Un RMS puede tener
una cúspide y el viñeteo puede introducir discontinuidades: una derivada central
nula no demuestra insensibilidad. Las unidades son las de la métrica divididas
por `parameter_unit` (mm, deg o índice adimensional).

El oráculo de sensibilidad es un espejo plano con detector a distancia L:

\[
x(\alpha)=-L\tan(2\alpha),\qquad
\left.\frac{dx}{d\alpha_{deg}}\right|_0=-2L\pi/180.
\]

Para L=100 mm, la pendiente es −3.490658504 mm/deg. La prueba exige error
relativo ≤2e-9 y también contrasta las realizaciones aleatorias con la tangente
exacta. Los errores numéricos de trazado quedan separados del viñeteo y la TIR.
Un haz completamente perdido es fallo registrado, no un spot perfecto.

`focus_policy="refocus"` minimiza analíticamente el RMS policromático al trasladar
el plano a lo largo de su normal, **por campo**. La fórmula se deriva en la guía
de fundamentos y se contrasta con un nuevo trazado al plano desplazado. No es
optimización de diseño ni una compensación física simultánea de todos los campos.
No añade interfaces ni aperturas entre planos. En un haz paralelo el mejor foco
queda indefinido y el evaluador lo rechaza.

## Monte Carlo e informe

`Tolerance(..., distribution="uniform")` interpreta `scale` como semianchura;
`"normal"` lo interpreta como desviación típica sin truncamiento. Las variables
son independientes. Se conservan semilla, todas las realizaciones, métricas y
fallos. `successful_fraction` es fracción calculada sin fallo, no rendimiento de
fabricación: este requiere distribuciones de proceso y criterios de aceptación.
No se implementan correlaciones, compensadores de montaje ni sensibilidad térmica.

`write_analysis` guarda JSON estricto con versión de formato, prescripción
completa, huella SHA-256 del modelo, ajustes, resultados, versiones del entorno
y huella de los módulos Python. Valores no finitos se convierten a `null`, que
nunca significa cero; el productor debe acompañarlos con estados o fallos.
La huella de código y la referencia CSV normalizan saltos de línea para ser
independientes del checkout de Windows.

## Ejemplo reproducible

```bash
python -m examples.aberrations.chromatic_sensitivity
```

El doblete se construye mediante las dos ecuaciones del límite delgado en contacto:
potencia en d = 0.01 mm⁻¹ y diferencia de potencia F−C = 0. Sus espesores reales
de 4 y 2 mm dejan aberraciones residuales; no es un diseño optimizado. El stop
es de radio 5 mm, el objeto está en infinito y el detector en el foco paraxial d.
Se usan campos 0°/3°, líneas F/d/C con pesos 1:2:1 y cuadratura 8×32 por color.

| Magnitud del ejemplo | Resultado |
|---|---:|
| Separación focal paraxial F−C (valor absoluto) | 73.44983 µm |
| RMS policromático de centroide, campo 3°, detector fijo | 15.64307 µm |
| RMS tras reenfoque diagnóstico del campo 3° | 7.38711 µm |
| Sensibilidad del centroide x al descentramiento x, campo 3° | 1.002688 mm/mm |
| Realizaciones Monte Carlo calculadas | 128/128 |
| Percentiles 5/50/95 del RMS respecto de referencia nominal | 16.52867 / 19.69874 / 25.69571 µm |

Monte Carlo usa semilla 20260923, distribuciones uniformes independientes de
±0.02 mm (traslación x del doblete), ±0.02° (inclinación y alrededor del origen)
y ±1e-4 (índice del crown). Son entradas ilustrativas, no tolerancias recomendadas.
El archivo `chromatic_sensitivity_results.json` conserva las realizaciones y
los ajustes; no hay que interpretar estos resultados como umbrales de aceptación.

## Layout y límites pendientes

`raytracer.viz.geometric.layout_3d_figure` representa las superficies y aperturas
en sus marcos reales y los caminos del trazador. Incluye vista completa y detalle;
los ejes se muestran en orden z/x/y, con etiquetas y coordenadas globales.
El contorno del detector es un parche de visualización, no una apertura física.
No infiere monturas, sólidos mecánicos ni interferencias. Los visores históricos
con hipótesis axiales siguen siendo herramientas diferentes.

Quedan por validar con casos independientes la extracción completa de Seidel,
curvatura de campo tangencial/sagital, distorsión y pupilas de entrada/salida.
También faltan un catálogo más amplio de fuentes primarias, convergencia de
muestreo para cada familia de sistemas y una interfaz integrada de edición.
La batería de diez contratos valida los casos declarados, no todo ese alcance.
