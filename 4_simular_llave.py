"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 4: Simular Llave
==============================================================
Simula TODAS las rondas del Mundial 2026 en orden obligatorio
(nomenclatura oficial en español, sin saltarse ninguna ronda):
  Dieciseisavos de Final → Octavos de Final → Cuartos de Final →
  Semifinales → Tercer y Cuarto Puesto → Final

Para cada partido muestra probabilidades por modelo (RF, XGB,
SVM, Stacking, Poisson), score probabilities, y ganador.

Combinación final: 65% ensemble + 35% ELO
Goles: 50% Poisson + 50% goles esperados ajustados por ELO
==============================================================
"""

import os
import sys
import pickle
import warnings
import time
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  Códigos ANSI y utilidades visuales
# ──────────────────────────────────────────────────────────────
R = "\033[0m"          # Reset
B = "\033[1m"          # Bold
D = "\033[38;5;220m"   # Dorado
V = "\033[38;5;82m"    # Verde
W = "\033[97m"         # Blanco
G = "\033[90m"         # Gris
C = "\033[38;5;51m"    # Cian
AZ = "\033[38;5;39m"   # Azul
L = "\033[38;5;135m"   # Lila
RO = "\033[38;5;196m"  # Rojo

FLAG_MAP = {}

# Nomenclatura oficial de fútbol en español (sin saltarse ninguna ronda):
#   32 equipos / 16 partidos  → Dieciseisavos de Final
#   16 equipos /  8 partidos  → Octavos de Final
#    8 equipos /  4 partidos  → Cuartos de Final
#    4 equipos /  2 partidos  → Semifinales
#    2 equipos /  1 partido   → Tercer y Cuarto Puesto (paralelo a la Final)
#    2 equipos /  1 partido   → Final
ROUND_ORDER = [
    "Dieciseisavos de Final",
    "Octavos de Final",
    "Cuartos de Final",
    "Semifinales",
    "Tercer y Cuarto Puesto",
    "Final",
]


def ct(texto, *codigos):
    """Colorea texto con códigos ANSI."""
    return "".join(codigos) + str(texto) + R


def flag(equipo):
    """Retorna prefijo visual sin emojis."""
    return FLAG_MAP.get(str(equipo).strip().lower(), "")


def barra(valor, ancho=20):
    """Barra de progreso Unicode."""
    valor = max(0.0, min(1.0, float(valor)))
    lleno = int(round(valor * ancho))
    return ct("█" * lleno, V) + ct("░" * (ancho - lleno), G) + f" {int(valor*100):>3d}%"


def equipo_texto(equipo, ganador=False, ancho=22):
    """Formatea nombre de equipo sin emojis."""
    f = flag(equipo)
    texto = f"{f} {equipo}" if f else str(equipo)
    texto = texto[:ancho].ljust(ancho)
    return ct(texto, D, B) if ganador else ct(texto, G)


def encabezado_ronda(titulo):
    """Imprime encabezado estilizado para una ronda."""
    linea = "═" * 68
    print()
    print(ct(linea, V))
    print(ct(f"  {titulo.center(66)}  ", D, B))
    print(ct(linea, V))
    print()


def prob_texto(valor):
    """Formatea probabilidad como porcentaje alineado."""
    return f"{valor*100:6.1f}%"


# ──────────────────────────────────────────────────────────────
#  Configuración de rutas
# ──────────────────────────────────────────────────────────────
RAIZ = os.path.dirname(os.path.abspath(__file__))
PROCESADOS = os.path.join(RAIZ, "procesados")
SALIDA = os.path.join(RAIZ, "salida")
os.makedirs(SALIDA, exist_ok=True)


def cargar_pickle(nombre):
    """Carga un pickle desde procesados/."""
    ruta = os.path.join(PROCESADOS, f"{nombre}.pkl")
    if not os.path.exists(ruta):
        print(f"  ⚠ {nombre}.pkl no encontrado.")
        print(f"    → Ejecute primero el script que lo genera.")
        return None
    with open(ruta, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
#  INICIO
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print(ct("  PREDICCIÓN MUNDIALISTA 2026 - SIMULACIÓN DE LLAVE", D, B))
print("═"*68)

# ─── [1/6] Cargar modelos y datos ─────────────────────────────
print("\n[1/6] Cargando modelos y datos...")

modelo_rf      = cargar_pickle("modelo_rf")
modelo_xgb     = cargar_pickle("modelo_xgb")
modelo_svm     = cargar_pickle("modelo_svm")
modelo_stack   = cargar_pickle("modelo_stack")
modelo_poisson = cargar_pickle("modelo_poisson")
features_df    = cargar_pickle("features")
stats_ind      = cargar_pickle("stats_individuales")
features_lista = cargar_pickle("features_modelo")
le             = cargar_pickle("label_encoder")
schedule_2026  = cargar_pickle("schedule_2026")

modelos_faltan = []
if modelo_rf is None:      modelos_faltan.append("modelo_rf.pkl")
if modelo_xgb is None:     modelos_faltan.append("modelo_xgb.pkl")
if modelo_svm is None:     modelos_faltan.append("modelo_svm.pkl")
if modelo_stack is None:   modelos_faltan.append("modelo_stack.pkl")
if modelo_poisson is None: modelos_faltan.append("modelo_poisson.pkl")

if modelos_faltan:
    print(f"\n  ✗ Faltan modelos: {', '.join(modelos_faltan)}")
    print(f"    → Ejecute primero: python 3_entrenar_modelo.py")
    sys.exit(1)

# Extraer el pipeline real del dict de XGB
pipe_xgb = modelo_xgb["modelo"] if isinstance(modelo_xgb, dict) else modelo_xgb
xgb_ok = modelo_xgb.get("xgb_ok", False) if isinstance(modelo_xgb, dict) else False

# Diccionarios de estadísticas
elo_dict     = stats_ind.get("elo", {}) if stats_ind else {}
rank_dict    = stats_ind.get("ranking_fifa", {}) if stats_ind else {}
pts_dict     = stats_ind.get("puntos_fifa", {}) if stats_ind else {}
stats_goles  = stats_ind.get("stats_goles", {}) if stats_ind else {}
pen_dict     = stats_ind.get("tasa_penales", {}) if stats_ind else {}
tit_dict     = stats_ind.get("titulos", {}) if stats_ind else {}
elo_medio    = stats_ind.get("elo_medio", 1500.0) if stats_ind else 1500.0
rank_medio   = stats_ind.get("rank_medio", 50.0) if stats_ind else 50.0
puntos_medio = stats_ind.get("puntos_medio", 1000.0) if stats_ind else 1000.0

print(f"  ✓ Modelos cargados: RF, {'XGBoost' if xgb_ok else 'GradBoost'}, SVM, Stacking, Poisson")
print(f"  ✓ Features: {len(features_lista)} variables")
print(f"  ✓ Label encoder: {len(le.classes_)} clases")


# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────

def get_elo(equipo):
    """Retorna ELO del equipo, con fallback al promedio."""
    v = elo_dict.get(equipo, elo_medio)
    return elo_medio if pd.isna(v) else float(v)


def prob_elo(elo_a, elo_b):
    """Probabilidad de victoria de A sobre B según diferencia ELO."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def goles_esperados_elo(equipo, rival):
    """Estima goles esperados combinando promedio histórico y diferencia ELO."""
    ga, gr = stats_goles.get(equipo, (1.2, 1.0))
    ga_rival, gr_rival = stats_goles.get(rival, (1.2, 1.0))
    base = (ga + gr_rival) / 2.0
    ajuste = (get_elo(equipo) - get_elo(rival)) / 100.0 * 0.15
    return max(0.3, base + ajuste)


def construir_features_par(eq_a, eq_b):
    """Construye el vector de features para un partido eq_a vs eq_b."""
    elo_a = get_elo(eq_a)
    elo_b = get_elo(eq_b)

    fila = None
    if features_df is not None:
        mask1 = (features_df["equipo_local"] == eq_a) & (features_df["equipo_visitante"] == eq_b)
        mask2 = (features_df["equipo_local"] == eq_b) & (features_df["equipo_visitante"] == eq_a)
        if mask1.any():
            fila = features_df[mask1].iloc[0]
        elif mask2.any():
            r = features_df[mask2].iloc[0].copy()
            for col in ["dif_elo", "dif_ranking_fifa", "dif_puntos_fifa",
                        "dif_goles_netos", "dif_titulos"]:
                r[col] = -r[col]
            h2h_orig = r["tasa_h2h_local"]
            r["tasa_h2h_local"] = r["tasa_h2h_visitante"]
            r["tasa_h2h_visitante"] = h2h_orig
            r["elo_local"], r["elo_visitante"] = r["elo_visitante"], r["elo_local"]
            r["goles_anotados_local"], r["goles_anotados_visitante"] = \
                r["goles_anotados_visitante"], r["goles_anotados_local"]
            r["goles_recibidos_local"], r["goles_recibidos_visitante"] = \
                r["goles_recibidos_visitante"], r["goles_recibidos_local"]
            fila = r

    if fila is not None:
        feat = {f: fila.get(f, 0.0) for f in features_lista}
        # H2H vacío → probabilidad ELO proporcional
        if abs(feat.get("tasa_h2h_local", 0.5) - 0.5) < 0.01:
            feat["tasa_h2h_local"] = prob_elo(elo_a, elo_b)
            feat["tasa_h2h_visitante"] = 1.0 - feat["tasa_h2h_local"]
        return feat

    # Calcular desde cero
    rank_a = rank_dict.get(eq_a, rank_medio)
    rank_b = rank_dict.get(eq_b, rank_medio)
    if pd.isna(rank_a): rank_a = rank_medio
    if pd.isna(rank_b): rank_b = rank_medio
    pts_a = pts_dict.get(eq_a, puntos_medio)
    pts_b = pts_dict.get(eq_b, puntos_medio)
    if pd.isna(pts_a): pts_a = puntos_medio
    if pd.isna(pts_b): pts_b = puntos_medio
    ga_a, gr_a = stats_goles.get(eq_a, (1.2, 1.0))
    ga_b, gr_b = stats_goles.get(eq_b, (1.2, 1.0))

    h2h_local = prob_elo(elo_a, elo_b)  # ELO proporcional cuando no hay H2H

    return {
        "dif_elo": elo_a - elo_b,
        "dif_ranking_fifa": rank_b - rank_a,
        "dif_puntos_fifa": pts_a - pts_b,
        "tasa_h2h_local": h2h_local,
        "tasa_h2h_visitante": 1.0 - h2h_local,
        "goles_anotados_local": ga_a,
        "goles_recibidos_local": gr_a,
        "goles_anotados_visitante": ga_b,
        "goles_recibidos_visitante": gr_b,
        "dif_goles_netos": (ga_a - gr_a) - (ga_b - gr_b),
        "tasa_penales_local": pen_dict.get(eq_a, 0.5),
        "tasa_penales_visitante": pen_dict.get(eq_b, 0.5),
        "dif_titulos": tit_dict.get(eq_a, 0) - tit_dict.get(eq_b, 0),
        "elo_local": elo_a,
        "elo_visitante": elo_b,
    }


def predecir_probabilidades_modelos(eq_local, eq_visitante):
    """
    Predice probabilidades de TODOS los modelos para un partido.
    Retorna dict con probabilidades por modelo + Poisson lambdas.
    """
    feats = construir_features_par(eq_local, eq_visitante)
    X = pd.DataFrame([feats], columns=features_lista).fillna(0.0)

    resultado = {}

    # --- Random Forest ---
    probs_rf = modelo_rf.predict_proba(X)[0]
    clases_rf = modelo_rf.classes_
    # clases_rf son ints (0,1,2) codificados por LabelEncoder
    for cls_idx, prob in zip(clases_rf, probs_rf):
        nombre = le.classes_[cls_idx]
        if nombre == "victoria_local":
            resultado["rf_local"] = prob
        elif nombre == "empate":
            resultado["rf_empate"] = prob
        else:
            resultado["rf_visitante"] = prob

    # --- XGBoost / GradientBoosting ---
    probs_xgb = pipe_xgb.predict_proba(X)[0]
    clases_xgb = pipe_xgb.classes_
    for cls_idx, prob in zip(clases_xgb, probs_xgb):
        nombre = le.classes_[cls_idx]
        if nombre == "victoria_local":
            resultado["xgb_local"] = prob
        elif nombre == "empate":
            resultado["xgb_empate"] = prob
        else:
            resultado["xgb_visitante"] = prob

    # --- SVM Calibrado ---
    probs_svm = modelo_svm.predict_proba(X)[0]
    clases_svm = modelo_svm.classes_
    for cls_idx, prob in zip(clases_svm, probs_svm):
        nombre = le.classes_[cls_idx]
        if nombre == "victoria_local":
            resultado["svm_local"] = prob
        elif nombre == "empate":
            resultado["svm_empate"] = prob
        else:
            resultado["svm_visitante"] = prob

    # --- Stacking Ensemble ---
    probs_stack = modelo_stack.predict_proba(X)[0]
    clases_stack = modelo_stack.classes_
    for cls_idx, prob in zip(clases_stack, probs_stack):
        nombre = le.classes_[cls_idx]
        if nombre == "victoria_local":
            resultado["stack_local"] = prob
        elif nombre == "empate":
            resultado["stack_empate"] = prob
        else:
            resultado["stack_visitante"] = prob

    # --- Poisson (lambdas) ---
    lambda_l = max(0.1, modelo_poisson["local"].predict(X)[0])
    lambda_v = max(0.1, modelo_poisson["visitante"].predict(X)[0])
    resultado["poisson_lambda_local"] = lambda_l
    resultado["poisson_lambda_visitante"] = lambda_v

    # --- Combinación final: 65% ensemble + 35% ELO ---
    elo_a = get_elo(eq_local)
    elo_b = get_elo(eq_visitante)
    p_elo_local = prob_elo(elo_a, elo_b)

    # Ensemble = promedio de los 4 clasificadores
    ens_local = np.mean([
        resultado.get("rf_local", 0.33),
        resultado.get("xgb_local", 0.33),
        resultado.get("svm_local", 0.33),
        resultado.get("stack_local", 0.33),
    ])
    ens_empate = np.mean([
        resultado.get("rf_empate", 0.33),
        resultado.get("xgb_empate", 0.33),
        resultado.get("svm_empate", 0.33),
        resultado.get("stack_empate", 0.33),
    ])
    ens_visitante = np.mean([
        resultado.get("rf_visitante", 0.33),
        resultado.get("xgb_visitante", 0.33),
        resultado.get("svm_visitante", 0.33),
        resultado.get("stack_visitante", 0.33),
    ])

    # Combinar: 65% ensemble + 35% ELO
    # ELO no predice empate → redistribuir proporcionalmente
    p_elo_no_empate = 1.0 - ens_empate
    p_elo_local_ajust = p_elo_local * p_elo_no_empate
    p_elo_visit_ajust = (1.0 - p_elo_local) * p_elo_no_empate

    resultado["final_local"] = 0.65 * ens_local + 0.35 * p_elo_local_ajust + 0.35 * ens_empate * (p_elo_local / (p_elo_local + (1-p_elo_local) + 1e-9))
    resultado["final_empate"] = 0.65 * ens_empate
    resultado["final_visitante"] = 0.65 * ens_visitante + 0.35 * p_elo_visit_ajust + 0.35 * ens_empate * ((1-p_elo_local) / (p_elo_local + (1-p_elo_local) + 1e-9))

    # Normalizar
    total = resultado["final_local"] + resultado["final_empate"] + resultado["final_visitante"]
    if total > 0:
        resultado["final_local"] /= total
        resultado["final_empate"] /= total
        resultado["final_visitante"] /= total

    # --- Goles: 50% Poisson + 50% ELO ajustado ---
    ge_local = goles_esperados_elo(eq_local, eq_visitante)
    ge_visitante = goles_esperados_elo(eq_visitante, eq_local)

    goles_l_raw = 0.50 * lambda_l + 0.50 * ge_local
    goles_v_raw = 0.50 * lambda_v + 0.50 * ge_visitante

    resultado["goles_local"] = goles_l_raw
    resultado["goles_visitante"] = goles_v_raw

    # Score probabilities (top 5 más probables)
    score_probs = []
    for gl in range(0, 5):
        for gv in range(0, 5):
            p = poisson_dist.pmf(gl, goles_l_raw) * poisson_dist.pmf(gv, goles_v_raw)
            score_probs.append((f"{gl}-{gv}", p))
    score_probs.sort(key=lambda x: x[1], reverse=True)
    resultado["top_scores"] = score_probs[:5]

    return resultado


def predecir_partido(eq_local, eq_visitante):
    """
    Predice el ganador y marcador de un partido eliminatorio.
    Retorna: (ganador, goles_l, goles_v, prob_local_final, prob_visit_final, prob_empate, detalles)
    """
    det = predecir_probabilidades_modelos(eq_local, eq_visitante)

    prob_local = det["final_local"]
    prob_visit = det["final_visitante"]
    prob_empate = det["final_empate"]

    goles_l = max(0, round(det["goles_local"]))
    goles_v = max(0, round(det["goles_visitante"]))

    # En eliminación directa no hay empate
    if goles_l == goles_v:
        if prob_local >= prob_visit:
            goles_l = goles_v + 1
        else:
            goles_v = goles_l + 1

    ganador = eq_local if goles_l > goles_v else eq_visitante
    prob_ganador = prob_local if ganador == eq_local else prob_visit

    return ganador, goles_l, goles_v, prob_local, prob_visit, prob_empate, det


# ─── [2/6] Determinar cruces iniciales ────────────────────────
print("\n[2/6] Determinando cruces de Dieciseisavos de Final...")

cruces_r32 = []

if schedule_2026 is not None:
    col_round = next((c for c in ["Round", "round", "Ronda"] if c in schedule_2026.columns), None)
    col_home  = next((c for c in ["home_team", "Home", "home"] if c in schedule_2026.columns), None)
    col_away  = next((c for c in ["away_team", "Away", "away"] if c in schedule_2026.columns), None)

    if col_round and col_home and col_away:
        mask_r32 = schedule_2026[col_round].str.contains(
            "Dieciseisavos|Round of 32|16avos", case=False, na=False
        )
        partidos_r32 = schedule_2026[mask_r32]
        if not partidos_r32.empty:
            for _, row in partidos_r32.iterrows():
                h = str(row[col_home]).strip()
                a = str(row[col_away]).strip()
                if h and a and "Winner" not in h and "Winner" not in a:
                    cruces_r32.append((h, a))

if not cruces_r32:
    print("  ⚠ No se encontraron cruces en schedule_2026.pkl")
    print("  → Asegúrese de haber ejecutado 0_actualizar_datos.py")
    sys.exit(1)

print(f"  → {len(cruces_r32)} cruces de Dieciseisavos de Final:")
for i, (h, a) in enumerate(cruces_r32, 1):
    print(f"    {i:2d}. {flag(h)} {h} vs {flag(a)} {a}")


# ─── Función para imprimir bloque visual de partido ────────────
def imprimir_bloque_partido(num, eq_local, eq_visitante, det,
                            ganador, goles_l, goles_v,
                            prob_local, prob_visit):
    """Imprime el bloque visual completo de un partido con formato Unicode."""
    fl = flag(eq_local)
    fv = flag(eq_visitante)
    gan_es_local = (ganador == eq_local)

    print(ct("┌" + "─"*66 + "┐", V))
    # Línea de título
    titulo = f"  Match {num:02d}: {eq_local}  vs  {eq_visitante}"
    print(ct("│", V) + f"{titulo:<66}" + ct("│", V))
    print(ct("├" + "─"*66 + "┤", V))

    # RF
    print(ct("│", V) + f"  RF      │ Local: {det.get('rf_local',0)*100:5.1f}%  "
          f"Empate: {det.get('rf_empate',0)*100:5.1f}%  "
          f"Visitante: {det.get('rf_visitante',0)*100:5.1f}%        " + ct("│", V))
    # XGB
    xgb_nombre = "XGB" if xgb_ok else "GB"
    print(ct("│", V) + f"  {xgb_nombre:3s}     │ Local: {det.get('xgb_local',0)*100:5.1f}%  "
          f"Empate: {det.get('xgb_empate',0)*100:5.1f}%  "
          f"Visitante: {det.get('xgb_visitante',0)*100:5.1f}%        " + ct("│", V))
    # SVM
    print(ct("│", V) + f"  SVM     │ Local: {det.get('svm_local',0)*100:5.1f}%  "
          f"Empate: {det.get('svm_empate',0)*100:5.1f}%  "
          f"Visitante: {det.get('svm_visitante',0)*100:5.1f}%        " + ct("│", V))
    print(ct("├" + "─"*66 + "┤", V))

    # Stacking
    print(ct("│", V) + f"  STACKING│ Local: {det.get('stack_local',0)*100:5.1f}%  "
          f"Empate: {det.get('stack_empate',0)*100:5.1f}%  "
          f"Visitante: {det.get('stack_visitante',0)*100:5.1f}%        " + ct("│", V))
    # Poisson
    lam_l = det.get("poisson_lambda_local", 1.0)
    lam_v = det.get("poisson_lambda_visitante", 1.0)
    print(ct("│", V) + f"  POISSON │ λ local={lam_l:.2f}   λ visitante={lam_v:.2f}"
          + " " * max(0, 66 - 38 - len(f"{lam_l:.2f}") - len(f"{lam_v:.2f}"))
          + ct("│", V))
    print(ct("├" + "─"*30 + "┤", V) + " " * 35 + ct("│", V))

    # Top scores
    for score, prob in det.get("top_scores", []):
        bar_len = int(prob * 100 / 2)
        bar = ct("█" * bar_len, C) + ct("░" * (10 - bar_len), G)
        linea = f"  {score} → {bar} {prob*100:5.2f}%"
        print(ct("│", V) + f"{linea:<30}" + ct("│", V) + " " * 35 + ct("│", V))
    print(ct("├" + "─"*66 + "┤", V))

    # AVANCE
    prob_gan = prob_local if gan_es_local else prob_visit
    bar_gan = ct("█" * int(prob_gan * 16), D) + ct("░" * (16 - int(prob_gan * 16)), G)
    fg = flag(ganador)
    linea_avance = f"  AVANCE  │ {ganador}: {prob_gan*100:.1f}% {bar_gan}  │  "
    otro = eq_visitante if gan_es_local else eq_local
    prob_otro = prob_visit if gan_es_local else prob_local
    fo = flag(otro)
    linea_avance += f"{otro}: {prob_otro*100:.1f}%"
    print(ct("│", V) + f"{linea_avance:<66}" + ct("│", V))

    # GANADOR
    linea_ganador = f"  GANADOR │ {ganador}  {goles_l} - {goles_v}  {'█'*int(prob_gan*14)}░  {prob_gan*100:.0f}%"
    print(ct("│", V) + ct(f"{linea_ganador:<66}", D, B) + ct("│", V))
    print(ct("└" + "─"*66 + "┘", V))
    print()


# ─── [3/6] Simular fase eliminatoria completa ─────────────────
print("\n[3/6] Simulando fase eliminatoria completa...\n")

resultados_llave = []
match_counter = 0


def simular_ronda(cruces, nombre_ronda):
    """Simula una ronda completa y retorna los ganadores."""
    global match_counter
    encabezado_ronda(nombre_ronda)
    ganadores = []

    for eq_a, eq_b in cruces:
        match_counter += 1
        ganador, gl, gv, pl, pv, pe, det = predecir_partido(eq_a, eq_b)

        imprimir_bloque_partido(
            match_counter, eq_a, eq_b, det,
            ganador, gl, gv, pl, pv
        )

        ganadores.append(ganador)
        prob_g = pl if ganador == eq_a else pv

        resultados_llave.append({
            "Ronda": nombre_ronda,
            "Match": match_counter,
            "Equipo Local": eq_a,
            "Equipo Visitante": eq_b,
            "Goles Local": gl,
            "Goles Visitante": gv,
            "Ganador": ganador,
            "Prob Local": round(pl, 4),
            "Prob Visitante": round(pv, 4),
            "Prob Empate": round(pe, 4),
            "Prob Victoria": round(prob_g, 4),
            # Probabilidades por modelo
            "RF Local": round(det.get("rf_local", 0), 4),
            "RF Empate": round(det.get("rf_empate", 0), 4),
            "RF Visitante": round(det.get("rf_visitante", 0), 4),
            "XGB Local": round(det.get("xgb_local", 0), 4),
            "XGB Empate": round(det.get("xgb_empate", 0), 4),
            "XGB Visitante": round(det.get("xgb_visitante", 0), 4),
            "SVM Local": round(det.get("svm_local", 0), 4),
            "SVM Empate": round(det.get("svm_empate", 0), 4),
            "SVM Visitante": round(det.get("svm_visitante", 0), 4),
            "Stack Local": round(det.get("stack_local", 0), 4),
            "Stack Empate": round(det.get("stack_empate", 0), 4),
            "Stack Visitante": round(det.get("stack_visitante", 0), 4),
            "Poisson λ Local": round(det.get("poisson_lambda_local", 0), 3),
            "Poisson λ Visitante": round(det.get("poisson_lambda_visitante", 0), 3),
        })

        time.sleep(0.03)  # Pequeña pausa para efecto visual

    return ganadores


# --- Dieciseisavos de Final (32 equipos → 16 partidos) ---
ganadores_r32 = simular_ronda(cruces_r32, "Dieciseisavos de Final")

# --- Octavos de Final (16 equipos → 8 partidos) ---
cruces_r16 = [(ganadores_r32[i], ganadores_r32[i + 1])
              for i in range(0, len(ganadores_r32) - 1, 2)]
ganadores_r16 = simular_ronda(cruces_r16, "Octavos de Final")

# --- Cuartos de Final (8 equipos → 4 partidos) ---
cruces_cuartos = [(ganadores_r16[i], ganadores_r16[i + 1])
                  for i in range(0, len(ganadores_r16) - 1, 2)]
ganadores_cuartos = simular_ronda(cruces_cuartos, "Cuartos de Final")

# --- Semifinales ---
cruces_semi = [(ganadores_cuartos[i], ganadores_cuartos[i + 1])
               for i in range(0, len(ganadores_cuartos) - 1, 2)]
ganadores_semi = simular_ronda(cruces_semi, "Semifinales")

# --- Tercer y Cuarto Puesto ---
perdedores_semi = []
for r in resultados_llave:
    if r["Ronda"] == "Semifinales":
        perdedor = r["Equipo Visitante"] if r["Ganador"] == r["Equipo Local"] else r["Equipo Local"]
        perdedores_semi.append(perdedor)

if len(perdedores_semi) >= 2:
    simular_ronda([(perdedores_semi[0], perdedores_semi[1])], "Tercer y Cuarto Puesto")

# --- Final ---
if len(ganadores_semi) >= 2:
    ganadores_final = simular_ronda([(ganadores_semi[0], ganadores_semi[1])], "Final")
    campeon = ganadores_final[0]
else:
    campeon = "Por determinar"


# ─── [4/6] Trofeo ASCII animado y campeón ─────────────────────
print("\n[4/6] Presentando campeón...")

for frame in range(3):
    trofeo = f"""
    {ct('╔══════════════════════════════════════════╗', D)}
    {ct('║                                          ║', D)}
    {ct('║              CAMPEÓN MUNDIAL             ║', D)}
    {ct('║                                          ║', D)}
    {ct('║' + f'  {flag(campeon)} {campeon}'.center(42) + '║', D, B)}
    {ct('║                                          ║', D)}
    {ct('╚══════════════════════════════════════════╝', D)}
"""
    print(trofeo)
    time.sleep(0.4)

final_row = next((r for r in resultados_llave if r["Ronda"] == "Final"), None)
prob_campeon = final_row["Prob Victoria"] if final_row else 0.0
print(ct(f"  CAMPEÓN: {campeon} | Probabilidad en la Final: {prob_campeon:.0%}", D, B))


# ─── [5/6] Top 5 favoritos y resumen ejecutivo ────────────────
print("\n[5/6] Calculando estadísticas de avance...\n")


def calcular_probabilidades_avance(resultados):
    """Calcula probabilidad acumulada de llegar a cada instancia."""
    equipos = set()
    for r in resultados:
        equipos.add(r["Equipo Local"])
        equipos.add(r["Equipo Visitante"])

    # Claves alineadas con la nomenclatura oficial en español:
    # Dieciseisavos (32) → Octavos (16) → Cuartos (8) → Semifinales (4) → Final (2) → Campeón (1)
    ruta = {eq: {"Dieciseisavos": 1.0, "Octavos": 0.0, "Cuartos": 0.0,
                  "Semifinales": 0.0, "Final": 0.0, "Campeón": 0.0}
            for eq in equipos}

    for r in resultados:
        ronda = r["Ronda"]
        local = r["Equipo Local"]
        visitante = r["Equipo Visitante"]
        pl = r["Prob Local"]
        pv = r["Prob Visitante"]

        if ronda == "Dieciseisavos de Final":
            ruta[local]["Octavos"] = pl
            ruta[visitante]["Octavos"] = pv
        elif ronda == "Octavos de Final":
            ruta[local]["Cuartos"] = ruta[local]["Octavos"] * pl
            ruta[visitante]["Cuartos"] = ruta[visitante]["Octavos"] * pv
        elif ronda == "Cuartos de Final":
            ruta[local]["Semifinales"] = ruta[local]["Cuartos"] * pl
            ruta[visitante]["Semifinales"] = ruta[visitante]["Cuartos"] * pv
        elif ronda == "Semifinales":
            ruta[local]["Final"] = ruta[local]["Semifinales"] * pl
            ruta[visitante]["Final"] = ruta[visitante]["Semifinales"] * pv
        elif ronda == "Final":
            ruta[local]["Campeón"] = ruta[local]["Final"] * pl
            ruta[visitante]["Campeón"] = ruta[visitante]["Final"] * pv

    return ruta


ruta_prob = calcular_probabilidades_avance(resultados_llave)

# Top 5
favoritos = sorted(ruta_prob.items(), key=lambda x: x[1]["Campeón"], reverse=True)[:5]

print(ct("  TOP 5 FAVORITOS — Probabilidad acumulada por instancia", D, B))
print(ct("  " + "─"*64, V))

for idx, (eq, probs) in enumerate(favoritos, 1):
    f = flag(eq)
    print(ct(f"  {idx}. {f} {eq}", W, B))
    print(f"     Campeón   : {barra(probs['Campeón'], 22)}")
    print(f"     Final     : {barra(probs['Final'], 22)}")
    print(f"     Semifinal : {barra(probs['Semifinales'], 22)}")
    print(f"     Cuartos   : {barra(probs['Cuartos'], 22)}")
    print(f"     Octavos   : {barra(probs['Octavos'], 22)}")
    print(f"     16avos    : {barra(probs['Dieciseisavos'], 22)}")
    print()

# Resumen ejecutivo
totales = [r["Goles Local"] + r["Goles Visitante"] for r in resultados_llave]
promedio = sum(totales) / len(totales) if totales else 0.0
mejor_prob = max(resultados_llave, key=lambda r: max(r["Prob Local"], r["Prob Visitante"]))
mayor_sorpresa = min(resultados_llave, key=lambda r: min(r["Prob Local"], r["Prob Visitante"]))

print(ct("  RESUMEN EJECUTIVO", D, B))
print(ct("  " + "─"*64, V))
print(f"  Promedio de goles del torneo: {ct(f'{promedio:.2f}', C)}")
print(f"  Partido más predecible: {flag(mejor_prob['Equipo Local'])} {mejor_prob['Equipo Local']} vs "
      f"{flag(mejor_prob['Equipo Visitante'])} {mejor_prob['Equipo Visitante']} "
      f"({max(mejor_prob['Prob Local'], mejor_prob['Prob Visitante']):.0%})")
print(f"  Mayor sorpresa: {flag(mayor_sorpresa['Equipo Local'])} {mayor_sorpresa['Equipo Local']} vs "
      f"{flag(mayor_sorpresa['Equipo Visitante'])} {mayor_sorpresa['Equipo Visitante']} "
      f"(ganó {flag(mayor_sorpresa['Ganador'])} {mayor_sorpresa['Ganador']}, "
      f"{min(mayor_sorpresa['Prob Local'], mayor_sorpresa['Prob Visitante']):.0%})")


# ─── [6/6] Guardar resultados ─────────────────────────────────
print(f"\n[6/6] Guardando archivos de salida...")

df_resultados = pd.DataFrame(resultados_llave)

# 1. pasan_a_octavos.csv — 16 partidos de Dieciseisavos de Final con
#    probabilidades por modelo. Los ganadores de estos 16 partidos son,
#    literalmente, los equipos que "pasan a Octavos de Final".
df_r32 = df_resultados[df_resultados["Ronda"] == "Dieciseisavos de Final"].copy()
ruta_pasan = os.path.join(RAIZ, "pasan_a_octavos.csv")
df_r32.to_csv(ruta_pasan, index=False, encoding="utf-8-sig")
print(f"  ✓ pasan_a_octavos.csv ({len(df_r32)} partidos)")

# 2. predicciones_copa_mundial_2026.csv — todos los partidos
ruta_pred_csv = os.path.join(RAIZ, "predicciones_copa_mundial_2026.csv")
df_resultados.to_csv(ruta_pred_csv, index=False, encoding="utf-8-sig")
print(f"  ✓ predicciones_copa_mundial_2026.csv ({len(df_resultados)} partidos)")

# 3. procesados/predicciones_llave.pkl
ruta_pkl = os.path.join(PROCESADOS, "predicciones_llave.pkl")
with open(ruta_pkl, "wb") as f:
    pickle.dump({
        "resultados": df_resultados,
        "campeon": campeon,
        "prob_campeon": prob_campeon,
        "ruta_probabilidades": ruta_prob,
        "favoritos": [(eq, probs) for eq, probs in favoritos],
        "cruces_r32": cruces_r32,
        "cruces_r16": cruces_r16,
        "cruces_cuartos": cruces_cuartos,
        "cruces_semi": cruces_semi,
    }, f)
print(f"  ✓ predicciones_llave.pkl guardado")

# ─── Final ─────────────────────────────────────────────────────
print(f"\n{'═'*68}")
print(ct(f"  SIMULACIÓN COMPLETADA", D, B))
print(ct(f"  CAMPEÓN PREDICHO: {campeon}", D, B))
print(f"  Archivos generados:")
print(f"    • pasan_a_octavos.csv")
print(f"    • predicciones_copa_mundial_2026.csv")
print(f"    • procesados/predicciones_llave.pkl")
print(f"\n  Próximo paso: python 5_visualizar_llave.py")
print("═"*68 + "\n")
