# TP3 Parte C — IO2 (RedBaires) — Experimentación, análisis y conclusiones

Repositorio de código, datos y resultados de la entrega de parte C: Parameter
Variation completo sobre las variables de decisión, análisis estadístico de sus
resultados y recomendación final al comité de RedBaires. El informe (procedimiento
de experimentación, dificultades y cambios vs. Parte A/B, resultados con intervalo
de confianza, comparación estadística y recomendación) se entrega por afuera de
este repositorio, como documento aparte — mismo criterio que la entrega de Parte B.

El motor de eventos discretos (`motor.py`, `parametros.py`, `costos.py`) es el
mismo de la entrega de Parte B, sin ninguna modificación — esta etapa no cambia el
simulador, solo agrega la capa de experimentación formal (Parameter Variation,
precisión objetivo, warm-up) y el análisis estadístico por encima.

## Estructura

```
IO2-TP3c-Entrega/
├── README.md                      # este archivo
├── codigo/
│   ├── motor.py                   # motor de eventos discretos (idéntico a Parte B)
│   ├── parametros.py              # carga de datos, calibración, escenarios
│   ├── costos.py                  # cálculo de la VR COSTO
│   ├── experimento_pv.py          # Parameter Variation: grilla de 81 combinaciones,
│   │                                 precisión objetivo relativa+piso, warm-up (MSER-5)
│   ├── analisis_estadistico.py    # regresión WLS (efectos principales + interacciones)
│   │                                 y frontera de Pareto
│   └── graficos_pv.py             # figuras de frontera de Pareto e interacción VD1×VD2
├── Datos/                          # parámetros, dataset limpio y coeficientes GLM calibrados
└── resultados/                     # CSV/TXT/PNG generados por el código (reproducibles)
```

Paquete autocontenido: todo lo que el código necesita está en `Datos/` (no depende
de ninguna otra carpeta del repositorio de trabajo de la cátedra).

## Cómo correr

Desde `codigo/`, con Python 3 + numpy/pandas/scipy/statsmodels/matplotlib
instalados:

```bash
python3 parametros.py           # chequeo de carga/calibración (opcional, standalone)
python3 motor.py                # 1 réplica suelta de prueba (opcional, standalone)
python3 costos.py               # chequeo de costeo (opcional, standalone)
python3 experimento_pv.py       # Parameter Variation completo (81 combinaciones) -> resultados/
python3 analisis_estadistico.py # regresión WLS + frontera de Pareto -> resultados/
python3 graficos_pv.py          # figuras -> resultados/
```

`experimento_pv.py` acepta `--smoke` (3 combinaciones, M chico) para verificar que
el pipeline corre antes de lanzar la grilla completa — tarda bastante más que las
otras etapas (del orden de media hora con el `M_MAX` real).

`analisis_estadistico.py` y `graficos_pv.py` dependen de que
`resultados/parameter_variation.csv` ya exista (lo genera `experimento_pv.py`) — no
hay un orden estricto entre ellos dos, salvo ese.

## Alcance de esta entrega

Implementa el Parameter Variation completo (factorial, no one-at-a-time) sobre las
4 variables de decisión (VD1 desacoplada en dotación de moderadores + paquete de
clasificador, VD2 fricción, VD3 degradación — 3 niveles cada una, 81 combinaciones),
con precisión objetivo relativa+piso y warm-up (MSER-5) para la ocupación de
moderadores. Incluye el análisis estadístico (regresión ponderada WLS con efectos
principales e interacciones) y la identificación de la frontera de Pareto
costo-beneficio entre las 81 combinaciones. **No incluye** el texto del informe
final (procedimiento completo, dificultades de experimentación, comparación formal
narrada y recomendación al comité en prosa) — eso se entrega por afuera de este
repositorio, como documento aparte.
