# Fundamentos y validación geométrica (0.4)

## Qué se considera validado

La batería contiene diez funciones de prueba en
`tests/test_geometrical_optics.py`, sin parametrización, marcadores de exclusión
ni dependencia del visor. Se eliminaron los 33 archivos de prueba anteriores.
Los casos relacionados se agrupan por propiedad física o integridad del proyecto;
esto concentra la validación, pero reduce deliberadamente la cobertura de
visualización, optimización y óptica ondulatoria.

Pasar estas pruebas acredita estos contratos y sus casos concretos. No acredita
cada prescripción existente, cualquier apertura numérica, todas las asferas ni
la equivalencia general con un programa comercial.

## Convenciones

- Geometría, radios, espesores, OPL y errores transversales: milímetros.
- Longitud de onda para materiales: micrómetros. El generador RayOptics convierte
  explícitamente a nanómetros al llamar a su API.
- Rayos: dirección unitaria; posiciones y direcciones devueltas en el marco global.
- Superficie: `z = sag(sqrt(x²+y²))` en su marco local, con normal local calculada
  a partir de su geometría. Las cónicas se restringen a la rama del vértice.
- Los espesores acumulan vértices nominales sobre el eje z del sistema.
  `row.placement` añade traslación y rotación respecto de ese vértice;
  `system.frame` lleva todo el sistema a coordenadas globales.
- `RigidTransform.from_euler_xyz` recibe grados: rotaciones activas sucesivas
  alrededor de los ejes fijos x, y, z (`Rz @ Ry @ Rx`).
- `image_placement` se aplica respecto del plano imagen nominal acumulado.
  `transform_surfaces(indices, transform)` mueve un grupo en coordenadas del
  sistema; para girar alrededor de c, la traslación es `c - R @ c`.
- `FieldPoint(x,y)` identifica alturas objeto. `FieldPoint.angle(x_deg,y_deg)`
  especifica dos ángulos cuyas tangentes son las pendientes del haz paralelo.
- OPL acumula distancias firmadas multiplicadas por el índice. El trazado físico
  rechaza transferencias internas negativas salvo autorización explícita mediante
  `allow_virtual_segments=True`. La primera transferencia virtual requiere
  `allow_virtual_object=True`; la última puede ser una prolongación virtual
  (`allow_virtual_image=True`, valor predeterminado).
- No se desplaza artificialmente el origen entre superficies. Se ha retirado
  `restart_offset` del trazador secuencial; debe eliminarse de llamadas antiguas.
- `TraceStatus.NO_INTERSECTION` identifica geometría sin intersección admisible;
  `DIVERGED`, fallo numérico de convergencia; `BACKWARD`, transferencia negativa
  no permitida; `PARALLEL`, paralelismo con un plano que se necesita alcanzar.

## Los diez contratos

| Nº | Propiedad | Oráculo y criterio principal |
|---|---|---|
| 1 | Leyes de interfaz y OPL | Snell, reflexión, reciprocidad, ángulo crítico, Brewster, R+T=1; OPL analítico de lámina con error absoluto ≤2e-12 mm |
| 2 | Intersecciones y dominio | Esfera de R=10 mm: sag analítica, ecuador, rayo horizontal; altura 11 mm rechazada; paralelo, retroceso y vector nulo diagnosticados |
| 3 | Primer orden y dispersión | Ecuación de lente gruesa, conjugados ABCD frente a rayos diferenciales, foco en inmersión; signo del conjugado virtual y dispersión N-BK7 |
| 4 | Superficies con solución exacta | Parábola/elipse: foco ≤2e-10 mm; óvalo de Descartes: error transversal y dispersión OPL <1e-9 mm |
| 5 | Pupila y aperturas | Momentos exactos del disco, medio conservado por un stop, máscaras circular/anular/rectangular, transmisión de medio disco=0.5 y stop en puente 2D |
| 6 | Marcos 3D | Covariancia de posiciones, direcciones y OPL bajo transformación rígida; movimiento de grupo; espejo a 45° que pliega el eje 90° |
| 7 | Apuntado | Error de coordenada objetivo en el diafragma <1e-9 mm para campo finito y angular; isotropía del cono de senos |
| 8 | Aberración esférica | Leyes LSA∝h² y TSA∝h³ con coeficientes analíticos; reducción del RMS en el mejor foco |
| 9 | Integridad del proyecto | JSON conserva geometría, materiales, aperturas, marcos y trazas; CSV rechaza datos no representables; versión desconocida rechazada |
| 10 | Referencia externa | 45 rayos de RayOptics: todas las posiciones y OPL con atol=1e-9 mm, componentes de dirección con atol=1e-11; rtol=0 |

Los umbrales son criterios de regresión para estas escalas geométricas. No
constituyen estimaciones universales del error del algoritmo. Para las leyes
asintóticas de tercer orden se admiten errores relativos explícitos, porque el
trazado exacto contiene términos de orden superior.

## Ecuaciones de referencia

### Intersección cónica

Con curvatura c y constante cónica K, la superficie implícita es

\[
c(x^2+y^2+(1+K)z^2)-2z=0.
\]

La sustitución `p+t d` produce una ecuación cuadrática. Se calculan sus raíces
con una forma que evita cancelación innecesaria, se descarta la segunda rama
algebraica y se escoge la primera intersección hacia delante. La normal
implícita permite tratar el ecuador de una esfera sin usar una derivada sag
infinita. Para asferas generales se usa Newton con comprobación final del
residuo; sigue pendiente un método global que encuentre todas las raíces.

### Primer orden

El estado paraxial es `(y, n_firmado·u)`. Un espejo invierte el signo del índice
axial efectivo para la matriz, sin cambiar el medio físico del rayo. Si C es
el elemento inferior izquierdo de la matriz de transferencia, la focal efectiva
firmada sobre el eje del sistema es `-n_imagen_firmado/C`.

Para una lente gruesa en aire:

\[
\Phi=(n-1)\left[\frac1{R_1}-\frac1{R_2}
 +\frac{(n-1)t}{nR_1R_2}\right],\qquad f'=1/\Phi.
\]

No se cambia el signo de un objeto virtual para convertirlo en objeto real.
La capa diferencial conserva el plano de partida usado para recuperar el
conjugado. ABCD escalar requiere superficies centradas; una transformación
rígida global puede manejarse convirtiendo los rayos al marco del sistema.

### Cuadratura de pupila y viñeteo

El área normalizada del disco es `dA/π = 2ρ dρ dφ/(2π)`. Usando `u=ρ²`, una
cuadratura de Gauss–Legendre en u y una malla azimutal uniforme proporciona
pesos por área. Dos identidades de referencia son:

\[
\langle\rho^2\rangle=1/2,\qquad\langle\rho^4\rangle=1/3.
\]

Los anillos existentes usan pesos trapezoidales en `u`, repartidos entre los
puntos de cada anillo. Son exactos para el primer momento anterior, sin
prometer la misma precisión de órdenes altos que Gauss. Una pupila anular se
mapea uniformemente en área entre el radio interior y el exterior.

`PupilTrace.weights` normaliza los rayos supervivientes para medir su spot.
`sample_weights` conserva el peso de todos los rayos originales. La fracción
geométrica transmitida usa estos últimos, de modo que el viñeteo permanece en
el denominador. No incluye pérdidas de Fresnel ni absorción. Un fallo de
apuntado también resta peso; sus residuos y estados permiten distinguirlo del
viñeteo físico. El rayo principal es una referencia geométrica sin recorte,
incluso cuando una obstrucción central lo bloquearía físicamente.

### Aberración esférica de un espejo

Se toma un espejo cóncavo de radio firmado `-R`, R positivo, vértice z=0 y rayos
incidentes paralelos a +z a altura h. La superficie tiene
`z=-R+sqrt(R²-h²)` y el foco paraxial está en `z=-R/2`.
La reflexión vectorial, desarrollada para `h/R` pequeño, da

\[
\mathrm{LSA}=z_{\rm corte}+R/2=\frac{h^2}{4R}+O(h^4/R^3),
\]

\[
\mathrm{TSA}=y(z=-R/2)=-\frac{h^3}{2R^2}+O(h^5/R^4).
\]

Así, duplicar h multiplica aproximadamente LSA por 4 y TSA por 8. Esta
comparación valida signos y coeficientes, además de la apariencia de la curva.
`axial_intercepts` devuelve el z de máxima proximidad al eje y la distancia
residual: un rayo sesgado no tiene por qué cruzar el eje.

### Mejor foco geométrico

En el marco del detector, se usan posiciones transversales p y pendientes q.
Una traslación Δ del detector en su z local da `p(Δ)=p+Δq`. Restando los
centroides ponderados, el mínimo del radio RMS se obtiene con

\[
\Delta_*=-\frac{\langle(p-\bar p)\cdot(q-\bar q)\rangle}
                 {\langle\lVert q-\bar q\rVert^2\rangle}.
\]

Esto es una medida analítica del foco de un haz ya trazado. No modifica la
prescripción ni realiza optimización de diseño. El resultado presupone que no
se añaden interfaces o aperturas entre ambos planos. Para un haz paralelo sin
variación de pendientes, el desplazamiento queda indefinido (`NaN`).

## Referencia externa reproducible

El archivo `tests/reference/rayoptics_cooke.json` contiene rayos calculados por
**RayOptics 0.9.8**, usando **opticalglass 1.2.0** en un entorno separado.
El generador registra ambas versiones, las entradas y el SHA-256 de la
prescripción. La prueba normal lee el archivo; no lo regenera con el motor que
está verificando.

- Prescripción: `data/optical_systems/photographic/cooke_triplet_prescription.csv`.
- Objeto: z=-100 mm; campos (0,0), (0,10), (7,7) mm.
- Longitudes de onda: 0.48613, 0.58756, 0.65627 µm.
- Cinco direcciones por campo, definidas hacia (0,0), (±2,0), (0,±2) en z=0.
- Aperturas deshabilitadas en la comparación: se contrasta el trazado, no el
  algoritmo de apuntado o viñeteo de RayOptics.
- La prescripción incluye un plano stop auxiliar en el vértice de una cara
  curva; algunas transferencias son algebraicamente negativas. Se habilita
  explícitamente `allow_virtual_segments=True` en ambos usos de la referencia.
- Las superficies, intersecciones, refracciones y longitudes geométricas se
  evalúan en RayOptics. Los índices se suministran como números resueltos por
  nuestro catálogo: **esta comparación no valida independientemente el catálogo**.
- El OPL absoluto se reconstruye con las distancias firmadas de RayOptics y los
  índices. No se confunde con su OPD por cuerdas igualmente inclinadas.

Regeneración explícita (requiere acceso a los paquetes):

```bash
python -m venv .venv-reference
.venv-reference/bin/python -m pip install -e ".[reference]"
.venv-reference/bin/python reference/generate_rayoptics_reference.py
```

En Windows, el ejecutable está en `.venv-reference\Scripts\python.exe`.
Para regenerar únicamente la figura y el informe con los datos ya versionados:

```bash
python -m examples.aberrations.geometrical_validation
```

Los resultados medidos se guardan en `docs/geometrical_validation_results.json`.
En el entorno de desarrollo: máximo error de coordenada 1.1541e-13 mm, máximo
error de dirección 9.9920e-16 y máximo error de OPL 1.4211e-13 mm. Son diferencias
numéricas entre implementaciones, no una precisión física de fabricación.

## Referencias y trabajo pendiente

La documentación primaria de [RayOptics: Optical Calculations and Analysis](https://ray-optics.readthedocs.io/en/stable/optical/index.html)
describe su motor secuencial y su base en el procedimiento general de Spencer
y Murty. La [implementación de RayOptics](https://github.com/mjhoptics/ray-optics)
permite inspeccionar la referencia usada. Las identidades analíticas anteriores
se explicitan aquí para que las pruebas puedan revisarse sin depender de una
salida numérica externa.

Esta etapa no certifica la extracción existente de Seidel ni su separación por
superficie. Sigue siendo prioritario contrastar sus convenciones y coeficientes
con teoría independiente, junto con curvatura de campo, astigmatismo, coma y
distorsión para varios campos y aperturas. También faltan pupilas de entrada y
salida completas, sensibilidad geométrica a desalineaciones y validación del
catálogo en sus rangos espectrales.

Los layouts y visores de sólidos conservan sus hipótesis axiales. El trazador
acepta aperturas rectangulares, pero el apuntado automático de pupila se limita
a diafragmas circulares/anulares. El CSV es una tabla geométrica: no conserva el plano objeto, el medio objeto,
los comentarios ni las leyes de dispersión completas; debe usarse JSON para
guardar un proyecto. El JSON v1 soporta los perfiles, aperturas y
materiales incorporados; tipos personalizados necesitan un serializador
explícito. El alcance de estas restricciones debe ampliarse con referencias
concretas, manteniendo claros los diez contratos de validación.
