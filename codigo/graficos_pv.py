"""Graficos del Parameter Variation - RedBaires (TP3 parte C, IO2).

Dos figuras, pensadas para los hallazgos de analisis_estadistico.py (no para
reexplorar los datos desde cero):

1. Frontera de Pareto: las 81 combinaciones (COSTO vs. ALCANCE_NOC), con las 11
   no dominadas resaltadas y conectadas - el hallazgo de que las 11 tienen todas
   VD2 en nivel alto se ve coloreando por nivel_vd2 (misma paleta de 3 colores
   que ya uso parte b/codigo/graficos.py para los 3 escenarios, reusada aca para
   mantener continuidad visual entre ambas partes del informe).
2. Interaccion moderadores x friccion sobre ALCANCE_NOC: una linea por nivel de
   VD2 (friccion), moderadores en el eje x - si las lineas no son paralelas, esa
   es la interaccion que detecto el WLS (p<0.0001). Los puntos son promedios
   ponderados por M sobre las 9 combinaciones de clasificador x VD3 que quedan
   fuera de este grafico (se marginalizan, no se fijan en un valor arbitrario).

Colores por nivel bajo/medio/alto: son categorias ORDINALES (intensidad
creciente de la palanca), no identidades sin orden - por eso se mantiene la
misma rampa fija de 3 colores de parte b en vez de inventar una paleta nueva
para este grafico puntual (bajo=azul, medio=naranja, alto=rojo, exactamente
como conservador/intermedio/pesimista en graficos.py de parte b).

Uso: python3 graficos_pv.py (guarda los 2 PNG en resultados/)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "..", "resultados")
ARCHIVO_ENTRADA = os.path.join(RESULTADOS_DIR, "parameter_variation.csv")

COLOR_NIVEL = {"bajo": "#4C72B0", "medio": "#DD8452", "alto": "#C44E52"}
ORDEN_NIVELES = ["bajo", "medio", "alto"]


def _guardar(fig, nombre):
    ruta = os.path.join(RESULTADOS_DIR, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"guardado: {ruta}")


def _cargar():
    return pd.read_csv(ARCHIVO_ENTRADA)


def _frontera_pareto(df):
    d = df.sort_values("costo_media").reset_index(drop=True)
    eficientes, mejor_alcance = [], np.inf
    for i, fila in d.iterrows():
        if fila["alcance_noc_media"] < mejor_alcance:
            eficientes.append(i)
            mejor_alcance = fila["alcance_noc_media"]
    return d.loc[eficientes].sort_values("costo_media")


def grafico_frontera_pareto():
    df = _cargar()
    eficientes = _frontera_pareto(df)
    dominadas = df.drop(eficientes.index)

    fig, eje = plt.subplots(figsize=(8.5, 6))

    for nivel in ORDEN_NIVELES:
        sub = dominadas[dominadas["nivel_vd2"] == nivel]
        eje.scatter(sub["costo_media"], sub["alcance_noc_media"],
                    color=COLOR_NIVEL[nivel], alpha=0.35, s=35, linewidths=0,
                    label=f"VD2 {nivel} (dominada)")

    eje.plot(eficientes["costo_media"], eficientes["alcance_noc_media"],
              color="#55A868", linewidth=1.5, zorder=2)
    for nivel in ORDEN_NIVELES:
        sub = eficientes[eficientes["nivel_vd2"] == nivel]
        eje.scatter(sub["costo_media"], sub["alcance_noc_media"],
                    color=COLOR_NIVEL[nivel], s=110, edgecolor="black",
                    linewidths=1.3, zorder=3,
                    label=f"VD2 {nivel} (eficiente)" if nivel == "alto" else None)

    eje.set_xlabel("COSTO total (USD)")
    eje.set_ylabel("ALCANCE_NOC (impresiones)")
    eje.set_title("Frontera de Pareto: 81 combinaciones del Parameter Variation\n"
                  "(las 11 combinaciones eficientes tienen VD2 en nivel alto, sin excepcion)")
    eje.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _guardar(fig, "frontera_pareto.png")


def grafico_interaccion_moderadores_friccion():
    df = _cargar()
    fig, eje = plt.subplots(figsize=(7.5, 5.5))

    for nivel in ORDEN_NIVELES:
        sub = df[df["nivel_vd2"] == nivel]
        agg = sub.groupby("moderadores_por_turno").apply(
            lambda g: np.average(g["alcance_noc_media"], weights=g["m"]),
            include_groups=False,
        ).sort_index()
        eje.plot(agg.index, agg.values, marker="o", markersize=7,
                  color=COLOR_NIVEL[nivel], linewidth=2, label=f"VD2 {nivel}")

    eje.set_xlabel("moderadores_por_turno (VD1)")
    eje.set_ylabel("ALCANCE_NOC promedio (marginalizado sobre clasificador y VD3)")
    eje.set_xticks([2, 6, 12])
    eje.set_title("Interaccion VD1 (moderadores) x VD2 (friccion) sobre ALCANCE_NOC\n"
                  "lineas no paralelas = el beneficio de moderadores depende del nivel de friccion")
    eje.legend(title="nivel de friccion")
    fig.tight_layout()
    _guardar(fig, "interaccion_moderadores_friccion.png")


def main():
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    grafico_frontera_pareto()
    grafico_interaccion_moderadores_friccion()


if __name__ == "__main__":
    main()
