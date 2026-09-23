# Cambios

## 0.6.0 — tercer orden y geometría de campo (propuesta)

- Cinco sumas Seidel por superficie desde rayos paraxiales; corrección cónica y
  cuártica explícita, predicción transversal y curvaturas con signo documentado.
- Focos parabasales por diferencias a dos pasos, separados del foco RMS.
- Distorsión para campos finitos e infinitos con mapa paraxial o f-theta;
  centroide opcional y estado de transmisión del principal explícito.
- Cinco referencias RayOptics independientes: 40 coeficientes de superficie,
  focos diferenciales y posiciones del principal. Convergencia residual de
  quinto orden transversal y cuarto orden en foco de campo. Diez pruebas.
- Guía matemática, informe reproducible y figura del doblete N-BK7/N-F2.

Migración: `seidel_coefficients(system, field, ...)` calcula sumas clásicas.
El ajuste anterior cambia a `fitted_low_order_coefficients`; se migran cuatro
notebooks históricos. En distorsión, grados definen haces realmente paralelos;
no se acepta una magnificación finita junto a campos angulares. La distorsión
relativa axial es indefinida, no cero. `aberrations_figure` requiere las nuevas
claves `parabasal_*` que devuelve `field_metrics`.

## 0.5.0 — cromática y sensibilidad (propuesta)

- Cromática con pesos espectrales y de pupila; detector común y RMS de centroide
  diferenciado del RMS respecto del rayo principal. Color axial analítico ABCD.
- Materiales con procedencia e intervalos declarados; contraste de ocho índices
  SCHOTT N-BK7/N-F2; aproximaciones históricas identificadas explícitamente.
- Perturbaciones de grupos, espesor, radio e índice sin modificar el nominal;
  sensibilidad a dos pasos y Monte Carlo con semilla y fallos conservados.
- Iluminación y detector fijos para sensibilidad; reenfoque diagnóstico por campo.
- JSON de análisis con prescripción, ajustes y huellas de modelo/código; layout 3D
  con vista completa y detalle; ejemplo reproducible de doblete analítico.
- Se mantienen diez pruebas, ampliadas con oráculos cromáticos y de sensibilidad.
  Referencia CSV independiente de CRLF; CI comprueba el paquete instalado.

Migración: una referencia espectral ausente ahora produce error. El muestreo
cromático predeterminado apunta al stop físico; solicitar `na_object_sine` para
usar explícitamente el cono anterior. Los materiales con intervalo declarado
rechazan extrapolaciones. Guardar en JSON para conservar metadatos y perturbación
de índice; el CSV de prescripción no representa un proyecto completo.

## 0.4.0 — fundamentos geométricos (propuesta)

- Intersección cónica estable, dominio de sag explícito, OPL firmado y segmentos
  virtuales bajo opciones explícitas; retirada de `restart_offset`.
- Marcos y aperturas 3D, apuntado al stop, cuadratura por área, conjugados firmados,
  aberraciones geométricas y persistencia JSON versionada.
- Sustitución de la batería previa por diez contratos; referencia externa de
  45 rayos RayOptics y comparación analítica de aberración esférica.
