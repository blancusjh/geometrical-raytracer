# Desarrollo y revisión

El objetivo actual es óptica geométrica y teoría de aberraciones. Todo análisis
nuevo debe declarar sus coordenadas, unidades, ponderación, referencia y alcance.

## Código

- Preferir funciones breves con nombres físicos y responsabilidades explícitas.
- Separar geometría, propagación, análisis, persistencia y visualización.
- Mantener espacios y formato uniformes con `python -m ruff format`.
- Revisar imports y errores comunes con `python -m ruff check` sobre los archivos modificados.
- Documentar decisiones numéricas y convenciones; evitar comentarios que repitan el código.
- Conservar entradas nominales al calcular variantes. Rechazar datos incompatibles.
- No sustituir fallos, referencias ausentes o modelos desconocidos por resultados plausibles.

## Diez contratos de validación

Se mantienen exactamente diez funciones en `tests/test_geometrical_optics.py`.
No aumentar el conteo mediante parametrización. Incorporar un caso dentro del
contrato físico que le corresponde o proponer una sustitución justificada.
Los oráculos deben proceder de teoría explícita, datos primarios o una
implementación independiente. No recalcular una referencia con el motor probado.
Explicar tolerancias: separar redondeo de fuentes, aproximación asintótica y error
numérico. Conservar generadores, entradas, versiones y convenciones de referencias.

```bash
python -m pip install . pytest
python -I -m pytest --import-mode=importlib -q
```

Este comando comprueba el paquete instalado. Para iterar sobre el checkout:
`python -m pip install -e ".[dev]"` y `python -m pytest -q`.
La matriz de CI cubre Python 3.11–3.13 en Linux, macOS y Windows.

## Cambios revisables

Cada propuesta debe explicar problema, comportamiento resultante, validación y
límites pendientes. Mantener ejemplos reproducibles y regenerar sus informes
cuando cambien las fórmulas o convenciones. Verificar visualmente figuras nuevas.
No afirmar equivalencia con otro software por el solo parecido de una gráfica.
