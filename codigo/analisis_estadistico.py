"""Analisis estadistico del Parameter Variation - RedBaires (TP3 parte C, IO2).

Lee resultados/parameter_variation.csv (81 combinaciones, ver experimento_pv.py) y
responde las dos preguntas del memo que el barrido factorial completo (no
one-at-a-time) esta pensado para contestar:

    1. "Conviene cada palanca, en que intensidad?" -> efecto principal de cada
       palanca sobre ALCANCE_NOC y COSTO.
    2. "Bajo que condiciones una palanca cara se vuelve imprescindible?" ->
       interacciones de a pares entre palancas (si una interaccion es
       significativa, el efecto de una palanca depende del nivel de la otra -
       eso es justo la nocion de "imprescindible bajo ciertas condiciones").

Metodo: WLS (weighted least squares), no OLS plano ni ANOVA clasico de
varianza homogenea - el rango de ALCANCE_NOC/COSTO entre combinaciones es
demasiado amplio (COSTO entre $33k y $245k, ALCANCE_NOC entre 3,7M y 21,1M)
para asumir que todas las celdas tienen la misma varianza. Peso de cada fila
= m / std^2 (inverso de la varianza de la media de esa celda, calculada a
partir de las M replicas que ya corrio experimento_pv.py) - asi las celdas
mas precisas (M alto, std bajo) pesan mas en el ajuste, que es exactamente lo
que hace una WLS bien especificada.

Predictores: se usan las UNIDADES DE INGENIERIA reales de cada palanca
(moderadores_por_turno, recall_clasificador, reduccion_alcance_rate_limit,
intensidad_degradacion), no un codigo generico 0/1/2 - asi el coeficiente es
interpretable directamente ("por cada moderador adicional...") y no asume que
los 3 niveles estan igual de espaciados (no lo estan: ej. intensidad_degradacion
va 0 / 0.5 / 0.9, no 0 / 0.33 / 0.66). `reduccion_alcance_rate_limit` no esta en
el CSV (se guardo `umbral_reenvios_rate_limit`, que va al reves: mas friccion =
umbral MAS BAJO) - se reconstruye aca desde `nivel_vd2` con el mapeo fijo del
diseno (ver EJE_VD2 en experimento_pv.py), para que las 4 palancas queden todas
orientadas igual (valor mas alto = mas intervencion).

Se reporta ademas el "efecto bajo->alto" de cada palanca (coeficiente * rango
real entre su nivel bajo y su nivel alto) - mas comparable entre palancas que
el coeficiente crudo, porque normaliza por la unidad de cada una.

Uso: python3 analisis_estadistico.py
"""

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "..", "resultados")
ARCHIVO_ENTRADA = os.path.join(RESULTADOS_DIR, "parameter_variation.csv")
ARCHIVO_SALIDA = os.path.join(RESULTADOS_DIR, "analisis_estadistico.txt")

# Mapeo fijo nivel_vd2 -> reduccion_alcance_rate_limit, igual al EJE_VD2 de
# experimento_pv.py (no se reimporta el modulo para no re-ejecutar nada).
REDUCCION_POR_NIVEL_VD2 = {"bajo": 0.0, "medio": 0.45, "alto": 0.90}

PALANCAS = {
    "moderadores_por_turno": "VD1 - moderadores",
    "recall_clasificador": "VD1 - clasificador",
    "reduccion_alcance_rate_limit": "VD2 - friccion",
    "intensidad_degradacion": "VD3 - degradacion",
}


def cargar_datos():
    df = pd.read_csv(ARCHIVO_ENTRADA)
    df["reduccion_alcance_rate_limit"] = df["nivel_vd2"].map(REDUCCION_POR_NIVEL_VD2)
    return df


def _formula(variable):
    p = list(PALANCAS.keys())
    principales = " + ".join(p)
    interacciones = " + ".join(f"{a}:{b}" for i, a in enumerate(p) for b in p[i + 1:])
    return f"{variable} ~ {principales} + {interacciones}"


def ajustar_wls(df, variable, columna_std):
    """WLS con peso = m / std^2 (inverso de la varianza de la media de cada
    celda) - ver docstring del modulo."""
    pesos = df["m"] / df[columna_std] ** 2
    modelo = smf.wls(_formula(variable), data=df, weights=pesos)
    return modelo.fit()


def _efecto_bajo_alto(resultado, df, variable_col):
    rango = df[variable_col].max() - df[variable_col].min()
    coef = resultado.params.get(variable_col, np.nan)
    pval = resultado.pvalues.get(variable_col, np.nan)
    return coef * rango, pval


def resumen_principales(resultado, df, titulo):
    lineas = [f"\n--- Efectos principales ({titulo}) ---",
              f"{'Palanca':28}{'coef (unidad real)':>20}{'efecto bajo->alto':>20}{'p-valor':>12}"]
    for col, nombre in PALANCAS.items():
        coef = resultado.params.get(col, np.nan)
        pval = resultado.pvalues.get(col, np.nan)
        efecto, _ = _efecto_bajo_alto(resultado, df, col)
        marca = "*" if pval < 0.05 else " "
        lineas.append(f"{nombre:28}{coef:>20,.4f}{efecto:>20,.0f}{pval:>11.4f}{marca}")
    return "\n".join(lineas)


def resumen_interacciones(resultado, titulo):
    cols = list(PALANCAS.keys())
    lineas = [f"\n--- Interacciones de a pares ({titulo}) ---",
              f"{'Par de palancas':45}{'coef':>18}{'p-valor':>12}"]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            clave = f"{a}:{b}"
            if clave not in resultado.params.index:
                clave = f"{b}:{a}"
            coef = resultado.params.get(clave, np.nan)
            pval = resultado.pvalues.get(clave, np.nan)
            marca = "*" if pval < 0.05 else " "
            nombre = f"{PALANCAS[a]} x {PALANCAS[b]}"
            lineas.append(f"{nombre:45}{coef:>18,.6f}{pval:>11.4f}{marca}")
    return "\n".join(lineas)


def frontera_pareto(df):
    """Combinaciones no dominadas: ninguna otra combinacion tiene a la vez
    menor ALCANCE_NOC y menor COSTO (son los 2 objetivos que el comite quiere
    minimizar a la vez - no existe una sola combinacion optima, existe un
    conjunto de alternativas eficientes entre las que el comite elige segun
    cuanto este dispuesto a pagar por cada impresion nociva evitada)."""
    d = df[["nombre", "nivel_moderadores", "nivel_clasificador", "nivel_vd2", "nivel_vd3",
            "alcance_noc_media", "costo_media"]].copy()
    d = d.sort_values("costo_media").reset_index(drop=True)
    eficientes = []
    mejor_alcance = np.inf
    for _, fila in d.iterrows():
        if fila["alcance_noc_media"] < mejor_alcance:
            eficientes.append(fila)
            mejor_alcance = fila["alcance_noc_media"]
    return pd.DataFrame(eficientes)


def main():
    df = cargar_datos()

    salida = []
    res_alcance = ajustar_wls(df, "alcance_noc_media", "alcance_noc_std")
    res_costo = ajustar_wls(df, "costo_media", "costo_std")

    salida.append("=" * 70)
    salida.append("ALCANCE_NOC (impresiones nocivas totales, promedio por combinacion)")
    salida.append("=" * 70)
    salida.append(resumen_principales(res_alcance, df, "ALCANCE_NOC"))
    salida.append(resumen_interacciones(res_alcance, "ALCANCE_NOC"))
    salida.append(f"\nR^2 (WLS) = {res_alcance.rsquared:.4f}")

    salida.append("\n" + "=" * 70)
    salida.append("COSTO (USD totales, promedio por combinacion)")
    salida.append("=" * 70)
    salida.append(resumen_principales(res_costo, df, "COSTO"))
    salida.append(resumen_interacciones(res_costo, "COSTO"))
    salida.append(f"\nR^2 (WLS) = {res_costo.rsquared:.4f}")

    pareto = frontera_pareto(df)
    salida.append("\n" + "=" * 70)
    salida.append(f"FRONTERA DE PARETO (costo-beneficio): {len(pareto)} de 81 combinaciones "
                   "no dominadas")
    salida.append("=" * 70)
    salida.append(pareto.to_string(index=False))

    texto = "\n".join(salida)
    print(texto)
    with open(ARCHIVO_SALIDA, "w") as f:
        f.write(texto + "\n")
    print(f"\nGuardado en {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
