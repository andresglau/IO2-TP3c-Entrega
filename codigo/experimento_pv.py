"""Parameter Variation - RedBaires (TP3 parte C, IO2).

Barrido factorial completo (no one-at-a-time) sobre las 3 palancas del memo,
con VD1 desacoplada en sus dos componentes de inversion (dotacion de
moderadores vs. paquete de clasificador - ver parte c/CLAUDE.md y la
discusion con el usuario sobre por que VD1 se desacopla y VD2/VD3 no: son
dos presupuestos de negocio distintos, mientras que en VD2/VD3 el umbral y
la magnitud son dos perillas de una sola politica). Motivo de ir a full
factorial y no secuencial-por-palanca: la pregunta del memo "bajo que
condiciones una palanca cara se vuelve imprescindible" es una pregunta de
INTERACCION entre palancas, que one-at-a-time no puede detectar (asume que
no hay interaccion).

Ejes de la grilla (3 niveles cada uno, reusando los ya calibrados en
parte b/codigo/parametros.py - ESCENARIOS):
    - moderadores_por_turno: 2 / 6 / 12
    - paquete clasificador (precision, recall, capex, opex atados): 3 tiers
    - VD2 (umbral_reenvios_rate_limit, reduccion_alcance_rate_limit): 3 tiers
    - VD3 (segmento_degradado, intensidad_degradacion): 3 tiers
    -> 3^4 = 81 combinaciones.

Precision objetivo RELATIVA + piso (decision del usuario, 2026-11-13): a
diferencia de monte_carlo.py (parte b), que usa un half-width (HW) absoluto
fijo calibrado solo sobre 3 escenarios, aca el rango de ALCANCE_NOC/COSTO es
mucho mas amplio (de "casi sin intervencion" a "las 3 palancas al maximo a
la vez"), asi que exigir el mismo HW absoluto en todos lados es o demasiado
laxo en combinaciones grandes o demasiado caro en combinaciones chicas. El
criterio queda: HW objetivo = max(piso_absoluto, R_RELATIVO * media).

Warm-up (decision del usuario, 2026-11-13): se aplica SOLO a
ocupacion_moderadores_promedio (unica VE con sesgo de arranque real: parte
en 0 porque no hay ninguna cascada activa todavia, que es la condicion
inicial real del incidente, no un artefacto - pero el promedio temporal de
ocupacion si arrastra ese tramo de "sistema vacio" si no se trunca). NO se
aplica a ALCANCE_NOC/COSTO (son VR acumulados del incidente completo, se
miden siempre desde t=0) ni a espera_cola (motor.py de parte b no guarda el
instante de encolado de cada item, y no se toca parte b - ver CLAUDE.md).
Metodo: MSER-5 sobre la curva de ocupacion promediada entre replicas piloto
(no Welch grafico - no escala a 81 corridas automaticas sin inspeccion
visual por combinacion).

Uso: python3 experimento_pv.py            (corre la grilla completa, escribe
                                            resultados/parameter_variation.csv)
     python3 experimento_pv.py --smoke    (3 combinaciones, M chico, para
                                            verificar que el pipeline corre
                                            antes de lanzar la corrida larga)
"""

import csv
import itertools
import os
import sys
import zlib

import numpy as np
from scipy import stats

import costos
import motor
import parametros as p

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "..", "resultados")
ARCHIVO_SALIDA = os.path.join(RESULTADOS_DIR, "parameter_variation.csv")

D = 30  # dias de incidente, igual que las corridas ya verificadas de parte b

# ---------------------------------------------------------------------------
# Ejes de la grilla (ver docstring del modulo)
# ---------------------------------------------------------------------------

EJE_MODERADORES = [2, 6, 12]

EJE_CLASIFICADOR = [
    dict(precision_clasificador=0.72, recall_clasificador=0.63,
         capex_clasificador=0.0, opex_anual_clasificador=0.0),
    dict(precision_clasificador=0.84, recall_clasificador=0.77,
         capex_clasificador=485_000.0, opex_anual_clasificador=100_000.0),
    dict(precision_clasificador=0.95, recall_clasificador=0.90,
         capex_clasificador=900_000.0, opex_anual_clasificador=180_000.0),
]

EJE_VD2 = [
    dict(umbral_reenvios_rate_limit=20, reduccion_alcance_rate_limit=0.0),
    dict(umbral_reenvios_rate_limit=5, reduccion_alcance_rate_limit=0.45),
    dict(umbral_reenvios_rate_limit=1, reduccion_alcance_rate_limit=0.90),
]

EJE_VD3 = [
    dict(segmento_degradado=frozenset(), intensidad_degradacion=0.0),
    dict(segmento_degradado=frozenset({"mega"}), intensidad_degradacion=0.50),
    dict(segmento_degradado=frozenset({"macro", "mega"}), intensidad_degradacion=0.90),
]

NIVELES = ["bajo", "medio", "alto"]

# ---------------------------------------------------------------------------
# Precision objetivo (relativa + piso) y parametros del procedimiento secuencial
# ---------------------------------------------------------------------------

R_RELATIVO = 0.05
PISO_ALCANCE = 300_000.0   # impresiones
PISO_COSTO = 500.0         # USD
NIVEL_CONFIANZA = 0.95

M0_PILOTO = 30
DELTA_M = 20
M_MAX = 500  # mas chico que el de parte b (2000): acotado porque son 81 combinaciones,
             # no 3 - si alguna no converge, mejor que corte rapido y quede marcada

M_PILOTO_WARMUP = 10          # replicas con trace para detectar el warm-up por combinacion
ANCHO_BIN_OCUPACION_MIN = 120  # bins de 2 horas para la curva de ocupacion
BATCH_MSER = 5                 # MSER-5: bloques de 5 bins (10 horas) para buscar el corte

SEMILLA_BASE = 7_000_000  # distinta de la de parte b (3_000_000) para no pisar semillas
                           # si alguna vez se corren los dos scripts comparando numeros


def armar_grilla():
    """Devuelve la lista de 81 combinaciones, cada una con sus metadatos de
    nivel (para el CSV/analisis estadistico posterior) y el dict de overrides
    listo para pasarle a parametros.armar_params_custom."""
    combos = []
    ejes = itertools.product(
        enumerate(EJE_MODERADORES),
        enumerate(EJE_CLASIFICADOR),
        enumerate(EJE_VD2),
        enumerate(EJE_VD3),
    )
    for (i_m, mod), (i_c, clasif), (i_v2, vd2), (i_v3, vd3) in ejes:
        overrides = dict(moderadores_por_turno=mod, **clasif, **vd2, **vd3)
        nombre = f"pv_m{i_m}_c{i_c}_vd2_{i_v2}_vd3_{i_v3}"
        combos.append(dict(
            nombre=nombre, overrides=overrides,
            nivel_moderadores=NIVELES[i_m], moderadores_por_turno=mod,
            nivel_clasificador=NIVELES[i_c], recall_clasificador=clasif["recall_clasificador"],
            nivel_vd2=NIVELES[i_v2], umbral_reenvios_rate_limit=vd2["umbral_reenvios_rate_limit"],
            nivel_vd3=NIVELES[i_v3], intensidad_degradacion=vd3["intensidad_degradacion"],
        ))
    return combos


def _semilla(nombre, m):
    """Mismo idioma que monte_carlo.py (parte b): zlib.crc32 en vez de
    hash() built-in para que la semilla de cada replica sea reproducible
    entre procesos."""
    offset = zlib.crc32(nombre.encode()) % 100_000
    return SEMILLA_BASE + offset + m


# ---------------------------------------------------------------------------
# Warm-up: MSER-5 sobre la curva de ocupacion promediada entre replicas piloto
# ---------------------------------------------------------------------------

def _curva_ocupacion_binada(trace_ocupacion, d_minutos, ancho_bin):
    """Integra el escalon (t, ocupados) del trace en bins de ancho fijo,
    devolviendo el promedio de ocupados (sin normalizar por capacidad - eso
    se hace afuera) en cada bin. Mismo principio que
    motor._registrar_cambio_ocupacion (area bajo la curva), aplicado por
    bin en vez de a la corrida completa."""
    n_bins = int(np.ceil(d_minutos / ancho_bin))
    areas = np.zeros(n_bins)
    t_prev, ocup_prev = 0.0, 0
    for t_evento, ocupados in trace_ocupacion:
        t_evento = min(t_evento, d_minutos)
        if t_evento > t_prev:
            _acumular_area(areas, t_prev, t_evento, ocup_prev, ancho_bin)
        t_prev, ocup_prev = t_evento, ocupados
    if t_prev < d_minutos:
        _acumular_area(areas, t_prev, d_minutos, ocup_prev, ancho_bin)
    return areas / ancho_bin


def _acumular_area(areas, t_ini, t_fin, valor, ancho_bin):
    """Reparte el area `valor * (t_fin - t_ini)` entre los bins que ese
    tramo [t_ini, t_fin) atraviesa (puede cruzar mas de uno)."""
    bin_ini = int(t_ini // ancho_bin)
    bin_fin = int((t_fin - 1e-9) // ancho_bin)
    for b in range(bin_ini, bin_fin + 1):
        a = max(t_ini, b * ancho_bin)
        z = min(t_fin, (b + 1) * ancho_bin)
        if z > a:
            areas[b] += valor * (z - a)


def mser5(serie, tam_batch=BATCH_MSER):
    """MSER-5 (Marginal Standard Error Rule, batch=5): agrupa `serie` en
    bloques de `tam_batch`, y devuelve el indice (en unidades de `serie`,
    no de bloque) del punto de corte que minimiza la varianza de la media de
    los bloques restantes. Metodo programatico (no requiere inspeccion
    visual por combinacion, a diferencia de Welch grafico) - necesario para
    automatizar la deteccion en 81 combinaciones sin intervencion manual."""
    n_bloques = len(serie) // tam_batch
    if n_bloques < 3:
        return 0
    medias_bloque = np.array([
        serie[i * tam_batch:(i + 1) * tam_batch].mean() for i in range(n_bloques)
    ])
    mejor_d, mejor_mse = 0, np.inf
    for d in range(n_bloques - 2):  # al menos 2 bloques remanentes
        resto = medias_bloque[d:]
        mse = resto.var(ddof=1) / len(resto)
        if mse < mejor_mse:
            mejor_mse, mejor_d = mse, d
    return mejor_d * tam_batch


def detectar_warmup(params, nombre, d_dias, m_piloto):
    """Corre `m_piloto` replicas con trace, promedia la curva de ocupacion
    bin a bin entre ellas (mismo espiritu que el metodo de Welch: promediar
    varias replicas para suavizar el ruido antes de buscar el corte), y le
    aplica MSER-5 al resultado. Devuelve el instante de corte en minutos."""
    d_minutos = d_dias * 24 * 60
    curvas = []
    for m in range(m_piloto):
        r = motor.correr_replica(nombre, d_dias, seed=_semilla(nombre, -1000 - m),
                                  devolver_trace=True, params=params)
        curvas.append(_curva_ocupacion_binada(r["trace"]["ocupacion"], d_minutos,
                                               ANCHO_BIN_OCUPACION_MIN))
    curva_promedio = np.mean(curvas, axis=0)
    bin_corte = mser5(curva_promedio)
    return bin_corte * ANCHO_BIN_OCUPACION_MIN


def _ocupacion_truncada(trace_ocupacion, t_warmup, d_minutos, capacidad):
    """Promedio de ocupacion (0-1) solo sobre [t_warmup, D), reconstruido
    desde el trace de la replica - analogo a
    motor._EstadoReplica.ve_area_ocupacion pero integrando desde t_warmup en
    vez de desde 0."""
    ventana = max(d_minutos - t_warmup, 1e-9)
    area = 0.0
    t_prev, ocup_prev = 0.0, 0
    for t_evento, ocupados in trace_ocupacion:
        t_evento = min(t_evento, d_minutos)
        a, z = max(t_prev, t_warmup), max(t_evento, t_warmup)
        if z > a:
            area += ocup_prev * (z - a)
        t_prev, ocup_prev = t_evento, ocupados
    if t_prev < d_minutos:
        a, z = max(t_prev, t_warmup), d_minutos
        if z > a:
            area += ocup_prev * (z - a)
    return area / (ventana * capacidad)


# ---------------------------------------------------------------------------
# Procedimiento secuencial (Law, 2015) con precision relativa + piso
# ---------------------------------------------------------------------------

def _hw(valores, confianza=NIVEL_CONFIANZA):
    n = len(valores)
    if n < 2:
        return float("inf")
    z = stats.norm.ppf(1 - (1 - confianza) / 2)
    return z * valores.std(ddof=1) / np.sqrt(n)


def _objetivo_relativo(valores, piso, r=R_RELATIVO):
    media = abs(valores.mean()) if len(valores) else 0.0
    return max(piso, r * media)


def correr_combo(meta, d_dias=D, confianza=NIVEL_CONFIANZA, m0_piloto=M0_PILOTO,
                  delta_m=DELTA_M, m_max=M_MAX, m_piloto_warmup=M_PILOTO_WARMUP):
    """Arma los parametros de la combinacion, detecta su warm-up de
    ocupacion, y corre el procedimiento secuencial sobre ALCANCE_NOC y COSTO
    hasta cumplir HW <= max(piso, R_RELATIVO*media) en ambas a la vez (o
    hasta m_max). Devuelve una fila lista para el CSV."""
    nombre = meta["nombre"]
    params = p.armar_params_custom(nombre, base_escenario="conservador", **meta["overrides"])
    d_minutos = d_dias * 24 * 60

    t_warmup = detectar_warmup(params, nombre, d_dias, m_piloto_warmup)

    muestras_alcance, muestras_costo, muestras_ocup_trunc = [], [], []
    m = 0
    while True:
        faltan = m0_piloto if m == 0 else delta_m
        for _ in range(faltan):
            r = motor.correr_replica(nombre, d_dias, seed=_semilla(nombre, m),
                                      devolver_trace=True, params=params)
            muestras_alcance.append(r["alcance_noc"])
            muestras_costo.append(costos.costo_total_replica(r, params, d_dias))
            muestras_ocup_trunc.append(_ocupacion_truncada(
                r["trace"]["ocupacion"], t_warmup, d_minutos, params.moderadores_por_turno))
            m += 1

        arr_alcance = np.array(muestras_alcance)
        arr_costo = np.array(muestras_costo)
        hw_alcance, hw_costo = _hw(arr_alcance, confianza), _hw(arr_costo, confianza)
        obj_alcance = _objetivo_relativo(arr_alcance, PISO_ALCANCE)
        obj_costo = _objetivo_relativo(arr_costo, PISO_COSTO)

        if hw_alcance <= obj_alcance and hw_costo <= obj_costo:
            break
        if m >= m_max:
            print(f"ADVERTENCIA [{nombre}]: no alcanzo precision objetivo "
                  f"(HW alcance={hw_alcance:,.0f} > {obj_alcance:,.0f} o "
                  f"HW costo={hw_costo:,.2f} > {obj_costo:,.2f}) - corte por m_max={m_max}.")
            break

    z = stats.norm.ppf(1 - (1 - confianza) / 2)
    arr_ocup = np.array(muestras_ocup_trunc)
    media_alcance, std_alcance = arr_alcance.mean(), arr_alcance.std(ddof=1)
    media_costo, std_costo = arr_costo.mean(), arr_costo.std(ddof=1)

    return dict(
        nombre=nombre,
        nivel_moderadores=meta["nivel_moderadores"], moderadores_por_turno=meta["moderadores_por_turno"],
        nivel_clasificador=meta["nivel_clasificador"], recall_clasificador=meta["recall_clasificador"],
        nivel_vd2=meta["nivel_vd2"], umbral_reenvios_rate_limit=meta["umbral_reenvios_rate_limit"],
        nivel_vd3=meta["nivel_vd3"], intensidad_degradacion=meta["intensidad_degradacion"],
        D=d_dias, m=m,
        hw_alcance_objetivo=obj_alcance, hw_alcance_final=hw_alcance,
        alcance_noc_media=media_alcance, alcance_noc_std=std_alcance,
        alcance_noc_ic_inf=media_alcance - z * std_alcance / np.sqrt(m),
        alcance_noc_ic_sup=media_alcance + z * std_alcance / np.sqrt(m),
        hw_costo_objetivo=obj_costo, hw_costo_final=hw_costo,
        costo_media=media_costo, costo_std=std_costo,
        costo_ic_inf=media_costo - z * std_costo / np.sqrt(m),
        costo_ic_sup=media_costo + z * std_costo / np.sqrt(m),
        t_warmup_ocupacion_min=t_warmup,
        ocupacion_promedio_truncada=arr_ocup.mean(),
    )


def main(smoke=False):
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    grilla = armar_grilla()
    kwargs = dict(m0_piloto=10, delta_m=10, m_max=40, m_piloto_warmup=5) if smoke else {}
    if smoke:
        grilla = grilla[:3]

    filas = []
    for i, meta in enumerate(grilla, 1):
        print(f"[{i}/{len(grilla)}] {meta['nombre']} "
              f"(mod={meta['nivel_moderadores']}, clasif={meta['nivel_clasificador']}, "
              f"vd2={meta['nivel_vd2']}, vd3={meta['nivel_vd3']})...", flush=True)
        fila = correr_combo(meta, **kwargs)
        filas.append(fila)
        print(f"    M={fila['m']}  ALCANCE_NOC={fila['alcance_noc_media']:,.0f}  "
              f"COSTO=${fila['costo_media']:,.2f}  "
              f"ocup.trunc={fila['ocupacion_promedio_truncada']:.2%} "
              f"(warmup={fila['t_warmup_ocupacion_min']:,.0f} min)", flush=True)

    with open(ARCHIVO_SALIDA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado en {ARCHIVO_SALIDA} ({len(filas)} combinaciones)")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
