"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 4: Simular Llave
==============================================================
Descripción: Lee los resultados reales de la fase de grupos,
determina los cruces de octavos y simula toda la fase
eliminatoria hasta la Final.
v2 - Corregido para predicciones más coherentes con ELO
==============================================================
"""

import os
import sys
import pickle
import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Presentación y utilidades visuales
# ──────────────────────────────────────────────────────────────

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DORADO = "\033[38;5;220m"
ANSI_VERDE = "\033[38;5;82m"
ANSI_AZUL = "\033[38;5;19m"
ANSI_BLANCO = "\033[97m"
ANSI_GRIS = "\033[90m"
ANSI_CIAN = "\033[38;5;51m"
ANSI_ROJO = "\033[38;5;196m"

FLAG_MAP = {
    "argentina": "🇦🇷", "brasil": "🇧🇷", "alemania": "🇩🇪", "francia": "🇫🇷",
    "españa": "🇪🇸", "inglaterra": "🏴", "portugal": "🇵🇹", "marruecos": "🇲🇦",
    "uruguay": "🇺🇾", "méxico": "🇲🇽", "colombia": "🇨🇴", "senegal": "🇸🇳",
    "japón": "🇯🇵", "croacia": "🇭🇷", "estados unidos": "🇺🇸", "costa rica": "🇨🇷",
    "canadá": "🇨🇦", "ecuador": "🇪🇨", "belgica": "🇧🇪", "holanda": "🇳🇱",
    "paises bajos": "🇳🇱", "inglaterra": "🏴", "suiza": "🇨🇭", "italia": "🇮🇹",
    "uruguay": "🇺🇾", "argentina": "🇦🇷", "corea del sur": "🇰🇷", "corea del norte": "🇰🇵",
    "argelia": "🇩🇿", "tunez": "🇹🇳", "egipto": "🇪🇬", "camerun": "🇨🇲",
    "senegal": "🇸🇳", "costarica": "🇨🇷", "arabia saudita": "🇸🇦", "australia": "🇦🇺",
}

ROUND_ORDER = [
    "Dieciseisavos de Final",
    "Octavos de Final",
    "Cuartos de Final",
    "Semifinales",
    "Tercer y Cuarto Puesto",
    "Final",
]


def color_text(text, *codes):
    return f"{''.join(codes)}{text}{ANSI_RESET}"


def flag_emoji(equipo):
    clave = str(equipo).strip().lower()
    return FLAG_MAP.get(clave, "")


def progress_bar(valor, ancho=22):
    valor = max(0.0, min(1.0, float(valor)))
    llenado = int(round(valor * ancho))
    barra = color_text("█" * llenado, ANSI_VERDE) + color_text("░" * (ancho - llenado), ANSI_GRIS)
    return f"{barra} {int(valor * 100):>3d}%"


def formato_equipo(equipo, ganador=False, ancho=22):
    bandera = flag_emoji(equipo)
    texto = f"{bandera} {equipo}" if bandera else str(equipo)
    texto = texto[:ancho].ljust(ancho)
    if ganador:
        return color_text(texto, ANSI_DORADO, ANSI_BOLD)
    return color_text(texto, ANSI_GRIS)


def encabezado_ronda(titulo):
    linea = "═" * 62
    print(color_text(linea, ANSI_VERDE))
    print(color_text(titulo.center(62), ANSI_DORADO, ANSI_BOLD))
    print(color_text(linea, ANSI_VERDE))


def texto_animado(texto, retardo=0.008):
    for caracter in texto:
        sys.stdout.write(caracter)
        sys.stdout.flush()
        time.sleep(retardo)
    sys.stdout.write("\n")


def ordenar_nombre_ronda(ronda):
    if ronda not in ROUND_ORDER:
        return 99
    return ROUND_ORDER.index(ronda)

# ─── Configuración de rutas ───────────────────────────────────
RAIZ = os.path.dirname(os.path.abspath(__file__))
PROCESADOS = os.path.join(RAIZ, "procesados")
SALIDA = os.path.join(RAIZ, "salida")
os.makedirs(SALIDA, exist_ok=True)


def cargar_pickle(nombre):
    ruta = os.path.join(PROCESADOS, f"{nombre}.pkl")
    if not os.path.exists(ruta):
        print(f"  ⚠ {nombre}.pkl no encontrado.")
        return None
    with open(ruta, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PREDICCIÓN MUNDIALISTA 2026 - SIMULACIÓN DE LLAVE")
print("═"*60)

# ─── Cargar modelos y datos ───────────────────────────────────
print("\n[1/6] Cargando modelos y datos...")
modelo_rf      = cargar_pickle("modelo_rf")
modelo_poisson = cargar_pickle("modelo_poisson")
features_df    = cargar_pickle("features")
stats_ind      = cargar_pickle("stats_individuales")
features_lista = cargar_pickle("features_modelo")
wc2026         = cargar_pickle("wc2026")
schedule_2026  = cargar_pickle("schedule_2026")

if modelo_rf is None or modelo_poisson is None:
    print("  ✗ Error: modelos no encontrados. Ejecute primero 3_entrenar_modelo.py")
    sys.exit(1)

elo_dict     = stats_ind.get("elo", {}) if stats_ind else {}
rank_dict    = stats_ind.get("ranking_fifa", {}) if stats_ind else {}
pts_dict     = stats_ind.get("puntos_fifa", {}) if stats_ind else {}
stats_goles  = stats_ind.get("stats_goles", {}) if stats_ind else {}
pen_dict     = stats_ind.get("tasa_penales", {}) if stats_ind else {}
tit_dict     = stats_ind.get("titulos", {}) if stats_ind else {}
elo_medio    = stats_ind.get("elo_medio", 1500.0) if stats_ind else 1500.0
rank_medio   = stats_ind.get("rank_medio", 50.0) if stats_ind else 50.0
puntos_medio = stats_ind.get("puntos_medio", 1000.0) if stats_ind else 1000.0


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def get_elo(equipo):
    """Retorna ELO del equipo, con fallback al promedio."""
    v = elo_dict.get(equipo, elo_medio)
    return elo_medio if pd.isna(v) else float(v)


def prob_elo(elo_a, elo_b):
    """
    Probabilidad de victoria de A sobre B según diferencia ELO.
    Fórmula estándar ELO de fútbol.
    """
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def goles_esperados(equipo, rival):
    """
    Estima los goles esperados de un equipo contra un rival
    combinando su promedio histórico con la diferencia de ELO.
    """
    ga, gr = stats_goles.get(equipo, (1.2, 1.0))
    ga_rival, gr_rival = stats_goles.get(rival, (1.2, 1.0))

    # Base: promedio de goles anotados del equipo vs goles recibidos del rival
    base = (ga + gr_rival) / 2.0

    # Ajuste por diferencia ELO: cada 100 puntos de diferencia mueve ~0.15 goles
    elo_a = get_elo(equipo)
    elo_b = get_elo(rival)
    ajuste_elo = (elo_a - elo_b) / 100.0 * 0.15

    esperados = max(0.3, base + ajuste_elo)
    return esperados


# ──────────────────────────────────────────────────────────────
# FUNCIÓN CENTRAL: construir features
# ──────────────────────────────────────────────────────────────

def construir_features_par(eq_a, eq_b):
    """Construye el vector de features para un partido eq_a vs eq_b."""

    elo_a = get_elo(eq_a)
    elo_b = get_elo(eq_b)

    # Intentar buscar en la matriz de features precalculada
    fila = None
    if features_df is not None:
        mask1 = (features_df["equipo_local"] == eq_a) & (features_df["equipo_visitante"] == eq_b)
        mask2 = (features_df["equipo_local"] == eq_b) & (features_df["equipo_visitante"] == eq_a)
        if mask1.any():
            fila = features_df[mask1].iloc[0]
        elif mask2.any():
            r = features_df[mask2].iloc[0].copy()
            # Invertir diferencias direccionales
            for col in ["dif_elo", "dif_ranking_fifa", "dif_puntos_fifa",
                        "dif_goles_netos", "dif_titulos"]:
                r[col] = -r[col]
            h2h_orig = r["tasa_h2h_local"]
            r["tasa_h2h_local"] = r["tasa_h2h_visitante"]
            r["tasa_h2h_visitante"] = h2h_orig
            r["elo_local"], r["elo_visitante"] = r["elo_visitante"], r["elo_local"]
            r["goles_anotados_local"], r["goles_anotados_visitante"] = (
                r["goles_anotados_visitante"], r["goles_anotados_local"])
            r["goles_recibidos_local"], r["goles_recibidos_visitante"] = (
                r["goles_recibidos_visitante"], r["goles_recibidos_local"])
            fila = r

    if fila is not None:
        feat = {f: fila.get(f, 0.0) for f in features_lista}
        # CORRECCIÓN CLAVE: sobreescribir H2H vacío con probabilidad ELO
        # Si el H2H es exactamente 0.5 significa que no hay historial real
        if abs(feat.get("tasa_h2h_local", 0.5) - 0.5) < 0.01:
            feat["tasa_h2h_local"] = prob_elo(elo_a, elo_b)
            feat["tasa_h2h_visitante"] = 1.0 - feat["tasa_h2h_local"]
        return feat

    # Calcular desde cero cuando no está en la matriz
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

    # CORRECCIÓN CLAVE: H2H basado en ELO cuando no hay historial
    h2h_local = prob_elo(elo_a, elo_b)

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


# ──────────────────────────────────────────────────────────────
# FUNCIÓN CENTRAL: predecir partido
# ──────────────────────────────────────────────────────────────

def predecir_partido(equipo_local, equipo_visitante):
    """
    Predice el ganador y marcador de un partido eliminatorio.
    Combina Random Forest con goles esperados ajustados por ELO.
    """
    feats = construir_features_par(equipo_local, equipo_visitante)
    X = pd.DataFrame([feats], columns=features_lista).fillna(0.0)

    # Probabilidades del Random Forest
    probs = modelo_rf.predict_proba(X)[0]
    clases = modelo_rf.classes_
    prob_dict = {c: p for c, p in zip(clases, probs)}
    prob_local     = prob_dict.get("victoria_local", 0.0)
    prob_empate    = prob_dict.get("empate", 0.0)
    prob_visitante = prob_dict.get("victoria_visitante", 0.0)

    # CORRECCIÓN CLAVE: combinar prob RF con prob ELO para suavizar anomalías
    elo_a = get_elo(equipo_local)
    elo_b = get_elo(equipo_visitante)
    p_elo_local = prob_elo(elo_a, elo_b)
    p_elo_visit = 1.0 - p_elo_local

    # Peso 60% RF + 40% ELO → evita que el RF se vuelva loco con equipos sin historial
    peso_rf  = 0.60
    peso_elo = 0.40
    prob_local_final     = peso_rf * prob_local     + peso_elo * p_elo_local
    prob_visitante_final = peso_rf * prob_visitante + peso_elo * p_elo_visit

    # Goles esperados ajustados por ELO (más coherentes que Poisson puro)
    ge_local     = goles_esperados(equipo_local, equipo_visitante)
    ge_visitante = goles_esperados(equipo_visitante, equipo_local)

    # Combinar Poisson con goles esperados ajustados
    goles_poisson_l = max(0.0, modelo_poisson["local"].predict(X)[0])
    goles_poisson_v = max(0.0, modelo_poisson["visitante"].predict(X)[0])

    # Promedio ponderado: 50% Poisson + 50% goles esperados por ELO
    goles_l_raw = 0.50 * goles_poisson_l + 0.50 * ge_local
    goles_v_raw = 0.50 * goles_poisson_v + 0.50 * ge_visitante

    goles_l = max(0, round(goles_l_raw))
    goles_v = max(0, round(goles_v_raw))

    # En eliminación directa no hay empate: resolver según probabilidad final
    if goles_l == goles_v:
        if prob_local_final >= prob_visitante_final:
            goles_l = goles_v + 1
        else:
            goles_v = goles_l + 1

    ganador = equipo_local if goles_l > goles_v else equipo_visitante
    prob_ganador = prob_local_final if ganador == equipo_local else prob_visitante_final

    return ganador, goles_l, goles_v, prob_local_final, prob_visitante_final, prob_empate


# ─── Leer resultados de fase de grupos ────────────────────────
print("\n[2/6] Leyendo resultados reales de la fase de grupos...")

grupos = {}
clasificados = {}

if wc2026 is not None:
    col_grupo    = next((c for c in ["Group", "grupo", "Grupo", "group"] if c in wc2026.columns), None)
    col_finished = next((c for c in ["Finished", "finished", "Played"] if c in wc2026.columns), None)
    col_home     = "Home" if "Home" in wc2026.columns else None
    col_away     = "Away" if "Away" in wc2026.columns else None
    col_hg       = next((c for c in ["HGFT", "home_score", "HomeGoals"] if c in wc2026.columns), None)
    col_ag       = next((c for c in ["AGFT", "away_score", "AwayGoals"] if c in wc2026.columns), None)

    if col_grupo and col_home and col_away and col_hg and col_ag:
        if col_finished:
            df_jugados = wc2026[wc2026[col_finished] == 1].copy()
        else:
            df_jugados = wc2026.dropna(subset=[col_hg, col_ag]).copy()

        for grupo_id in df_jugados[col_grupo].dropna().unique():
            partidos_grupo = df_jugados[df_jugados[col_grupo] == grupo_id]
            grupos[str(grupo_id)] = partidos_grupo

        for grupo_id, partidos in grupos.items():
            tabla = {}
            for _, row in partidos.iterrows():
                hl = row[col_home]
                av = row[col_away]
                hg = int(row[col_hg]) if pd.notna(row[col_hg]) else 0
                ag = int(row[col_ag]) if pd.notna(row[col_ag]) else 0

                for eq in [hl, av]:
                    if eq not in tabla:
                        tabla[eq] = {"pts": 0, "gf": 0, "gc": 0, "pj": 0}

                tabla[hl]["pj"] += 1
                tabla[av]["pj"] += 1
                tabla[hl]["gf"] += hg
                tabla[hl]["gc"] += ag
                tabla[av]["gf"] += ag
                tabla[av]["gc"] += hg

                if hg > ag:
                    tabla[hl]["pts"] += 3
                elif hg == ag:
                    tabla[hl]["pts"] += 1
                    tabla[av]["pts"] += 1
                else:
                    tabla[av]["pts"] += 3

            tabla_ord = sorted(
                tabla.items(),
                key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["gc"], x[1]["gf"]),
                reverse=True
            )
            clasificados[str(grupo_id)] = [t[0] for t in tabla_ord]

        print(f"  → {len(grupos)} grupos detectados desde WorldCup2026.xlsx")
        for g, equipos in clasificados.items():
            print(f"    Grupo {g}: {' | '.join(equipos)}")


# ─── Determinar cruces ────────────────────────────────────────
print("\n[3/6] Determinando cruces de Dieciseisavos de Final...")


def get_clasificado(grupo, posicion, clasificados):
    equipos = clasificados.get(str(grupo), [])
    if len(equipos) > posicion:
        return equipos[posicion]
    return f"Equipo_{grupo}_{posicion+1}"


cruces_dieciseisavos = []

if schedule_2026 is not None:
    col_round = next((c for c in ["Round", "round", "Ronda"] if c in schedule_2026.columns), None)
    col_home  = next((c for c in ["home_team", "Home", "home"] if c in schedule_2026.columns), None)
    col_away  = next((c for c in ["away_team", "Away", "away"] if c in schedule_2026.columns), None)

    if col_round and col_home and col_away:
        # Filtrar cualquier ronda que NO sea fase de grupos
        mask_ko = ~schedule_2026[col_round].str.contains(
            "Group|group|Grupo|grupo|fase de grupos",
            case=False, na=False
        )
        partidos_ko = schedule_2026[mask_ko].copy()

        if not partidos_ko.empty:
            # Tomar solo la primera ronda disponible (la más próxima)
            primera_ronda = partidos_ko[col_round].iloc[0]
            partidos_primera = partidos_ko[partidos_ko[col_round] == primera_ronda]

            print(f"  → Ronda detectada en schedule: '{primera_ronda}'")

            def resolver_ref(ref, clasificados):
                ref = str(ref).strip()
                if len(ref) == 2 and ref[0].isdigit() and ref[1].isalpha():
                    pos = int(ref[0]) - 1
                    grp = ref[1].upper()
                    return get_clasificado(grp, pos, clasificados)
                return ref

            for _, row in partidos_primera.iterrows():
                h = resolver_ref(row[col_home], clasificados)
                a = resolver_ref(row[col_away], clasificados)
                if h and a and str(h).strip() and str(a).strip():
                    cruces_dieciseisavos.append((h, a))

# Fallback automático desde tablas de grupos
if not cruces_dieciseisavos and clasificados:
    grupos_ord = sorted(clasificados.keys())
    n = len(grupos_ord)
    for i in range(0, n, 2):
        if i + 1 < n:
            g1 = grupos_ord[i]
            g2 = grupos_ord[i + 1]
            cruces_dieciseisavos.append((get_clasificado(g1, 0, clasificados),
                                         get_clasificado(g2, 1, clasificados)))
            cruces_dieciseisavos.append((get_clasificado(g2, 0, clasificados),
                                         get_clasificado(g1, 1, clasificados)))

# Fallback manual
if not cruces_dieciseisavos:
    print("  ⚠ No se pudieron determinar los cruces automáticamente.")
    print("  → Usando equipos de ejemplo para demostración.")
    cruces_dieciseisavos = [
        ("Argentina", "Francia"),
        ("Brasil", "Portugal"),
        ("España", "Alemania"),
        ("Inglaterra", "Países Bajos"),
        ("Marruecos", "Uruguay"),
        ("Estados Unidos", "México"),
        ("Japón", "Croacia"),
        ("Colombia", "Senegal"),
    ]

print(f"  → {len(cruces_dieciseisavos)} cruces determinados:")
for i, (h, a) in enumerate(cruces_dieciseisavos, 1):
    print(f"    {i:2d}. {h} vs {a}")


# ─── Simular fase eliminatoria ────────────────────────────────
print("\n[4/6] Simulando fase eliminatoria completa...")

resultados_llave = []


def simular_ronda(cruces, nombre_ronda):
    print(f"\n  {'─'*55}")
    print(f"  ⚽ {nombre_ronda.upper()}")
    print(f"  {'─'*55}")
    ganadores = []
    for eq_a, eq_b in cruces:
        ganador, gl, gv, pl, pv, pe = predecir_partido(eq_a, eq_b)
        ganadores.append(ganador)
        prob_g = pl if ganador == eq_a else pv
        print(f"  {eq_a:<25} {gl} - {gv} {eq_b:<25}  → {ganador} ({prob_g:.0%})")
        resultados_llave.append({
            "Ronda": nombre_ronda,
            "Equipo Local": eq_a,
            "Equipo Visitante": eq_b,
            "Goles Local": gl,
            "Goles Visitante": gv,
            "Ganador": ganador,
            "Probabilidad Local": round(pl, 3),
            "Probabilidad Visitante": round(pv, 3),
            "Probabilidad Empate Tiempo Normal": round(pe, 3),
            "Probabilidad Victoria": round(prob_g, 3),
        })
    return ganadores


def calcular_avance_probabilidades(resultados):
    estadisticas = {}
    for partido in resultados:
        for equipo in [partido["Equipo Local"], partido["Equipo Visitante"]]:
            if equipo not in estadisticas:
                estadisticas[equipo] = {
                    "Dieciseisavos": 0.0,
                    "Octavos": 0.0,
                    "Cuartos": 0.0,
                    "Semifinales": 0.0,
                    "Final": 0.0,
                    "Campeon": 0.0,
                }

    # Probabilidades de avance por partido
    ganadores_prob = {
        "Dieciseisavos de Final": "Octavos",
        "Octavos de Final": "Cuartos",
        "Cuartos de Final": "Semifinales",
        "Semifinales": "Final",
        "Final": "Campeon",
    }

    ruta_prob = {equipo: {"Dieciseisavos": 1.0, "Octavos": 0.0, "Cuartos": 0.0, "Semifinales": 0.0, "Final": 0.0, "Campeon": 0.0}
                 for equipo in estadisticas}

    for partido in resultados:
        ronda = partido["Ronda"]
        local = partido["Equipo Local"]
        visitante = partido["Equipo Visitante"]
        prob_local = partido["Probabilidad Local"]
        prob_visit = partido["Probabilidad Visitante"]

        if ronda == "Dieciseisavos de Final":
            ruta_prob[local]["Octavos"] = prob_local
            ruta_prob[visitante]["Octavos"] = prob_visit
        elif ronda == "Octavos de Final":
            ruta_prob[local]["Cuartos"] = ruta_prob[local]["Octavos"] * prob_local
            ruta_prob[visitante]["Cuartos"] = ruta_prob[visitante]["Octavos"] * prob_visit
        elif ronda == "Cuartos de Final":
            ruta_prob[local]["Semifinales"] = ruta_prob[local]["Cuartos"] * prob_local
            ruta_prob[visitante]["Semifinales"] = ruta_prob[visitante]["Cuartos"] * prob_visit
        elif ronda == "Semifinales":
            ruta_prob[local]["Final"] = ruta_prob[local]["Semifinales"] * prob_local
            ruta_prob[visitante]["Final"] = ruta_prob[visitante]["Semifinales"] * prob_visit
        elif ronda == "Final":
            ruta_prob[local]["Campeon"] = ruta_prob[local]["Final"] * prob_local
            ruta_prob[visitante]["Campeon"] = ruta_prob[visitante]["Final"] * prob_visit

    for equipo, probs in ruta_prob.items():
        estadisticas[equipo]["Dieciseisavos"] = probs["Dieciseisavos"]
        estadisticas[equipo]["Octavos"] = probs["Octavos"]
        estadisticas[equipo]["Cuartos"] = probs["Cuartos"]
        estadisticas[equipo]["Semifinales"] = probs["Semifinales"]
        estadisticas[equipo]["Final"] = probs["Final"]
        estadisticas[equipo]["Campeon"] = probs["Campeon"]

    return estadisticas


def ordenar_favoritos(estadisticas):
    favoritos = []
    for equipo, datos in estadisticas.items():
        favoritos.append({
            "Equipo": equipo,
            "Campeon": datos["Campeon"],
            "Final": datos["Final"],
            "Semifinales": datos["Semifinales"],
            "Cuartos": datos["Cuartos"],
            "Octavos": datos["Octavos"],
            "Dieciseisavos": datos["Dieciseisavos"],
        })
    return sorted(favoritos, key=lambda x: x["Campeon"], reverse=True)[:5]


def barra_estadistica(valor, ancho=18):
    return color_text("█" * int(valor * ancho), ANSI_VERDE) + color_text("░" * (ancho - int(valor * ancho)), ANSI_GRIS)


def mostrar_top5_favoritos(favoritos):
    print(color_text("\nTOP 5 FAVORITOS", ANSI_DORADO, ANSI_BOLD))
    print(color_text("─" * 62, ANSI_VERDE))
    for idx, fav in enumerate(favoritos, 1):
        print(color_text(f"{idx}. {fav['Equipo']}", ANSI_BLANCO, ANSI_BOLD))
        print(f"   Campeón : {barra_estadistica(fav['Campeon'])} {fav['Campeon']:.0%}")
        print(f"   Final   : {barra_estadistica(fav['Final'])} {fav['Final']:.0%}")
        print(f"   Semi    : {barra_estadistica(fav['Semifinales'])} {fav['Semifinales']:.0%}")
        print(f"   Cuartos : {barra_estadistica(fav['Cuartos'])} {fav['Cuartos']:.0%}")
        print(f"   Octavos : {barra_estadistica(fav['Octavos'])} {fav['Octavos']:.0%}")
        print(f"   16avos  : {barra_estadistica(fav['Dieciseisavos'])} {fav['Dieciseisavos']:.0%}\n")


def resumen_torneo(resultados):
    totales = [r["Goles Local"] + r["Goles Visitante"] for r in resultados]
    promedio = sum(totales) / len(totales) if totales else 0.0
    mejor_prob = max(resultados, key=lambda r: max(r["Probabilidad Local"], r["Probabilidad Visitante"]))
    sorpresa = min(resultados, key=lambda r: min(r["Probabilidad Local"], r["Probabilidad Visitante"]))
    return promedio, mejor_prob, sorpresa


def imprimir_bracket_conectado(resultados):
    for ronda in ROUND_ORDER:
        partidos_ronda = [r for r in resultados if r["Ronda"] == ronda]
        encabezado_ronda(ronda.upper())
        if not partidos_ronda:
            print(color_text("  No hay partidos disponibles para esta ronda, se proyectan resultados.", ANSI_GRIS))
            continue
        for partido in partidos_ronda:
            local = partido["Equipo Local"]
            visitante = partido["Equipo Visitante"]
            ganador = partido["Ganador"]
            marcador = f"{partido['Goles Local']} - {partido['Goles Visitante']}"
            prob_local = partido["Probabilidad Local"]
            prob_visit = partido["Probabilidad Visitante"]
            ganador_local = ganador == local
            print(color_text(f"  {formato_equipo(local, ganador_local)}  {marcador}  {formato_equipo(visitante, not ganador_local)}", ANSI_BLANCO))
            print(f"    {progress_bar(prob_local)} vs {progress_bar(prob_visit)}")
        print()


# Ronda inicial (Dieciseisavos de Final)
ganadores_r1 = simular_ronda(cruces_dieciseisavos, "Dieciseisavos de Final")

# Octavos de final
cruces_octavos = [(ganadores_r1[i], ganadores_r1[i + 1])
                   for i in range(0, len(ganadores_r1) - 1, 2)]
ganadores_octavos = simular_ronda(cruces_octavos, "Octavos de Final")

# Cuartos de final
cruces_cuartos = [(ganadores_octavos[i], ganadores_octavos[i + 1])
                   for i in range(0, len(ganadores_octavos) - 1, 2)]
ganadores_cuartos = simular_ronda(cruces_cuartos, "Cuartos de Final")

# Semifinales
cruces_semi = [(ganadores_cuartos[i], ganadores_cuartos[i + 1])
               for i in range(0, len(ganadores_cuartos) - 1, 2)]
ganadores_semi = simular_ronda(cruces_semi, "Semifinales")

# Tercer puesto
perdedores_semi = []
for r in resultados_llave:
    if r["Ronda"] == "Semifinales":
        perdedor = (r["Equipo Local"] if r["Ganador"] == r["Equipo Visitante"]
                    else r["Equipo Visitante"])
        perdedores_semi.append(perdedor)

if len(perdedores_semi) >= 2:
    simular_ronda([(perdedores_semi[0], perdedores_semi[1])], "Tercer y Cuarto Puesto")

# Final
if len(ganadores_semi) >= 2:
    ganadores_final = simular_ronda([(ganadores_semi[0], ganadores_semi[1])], "Final")
    campeon = ganadores_final[0] if ganadores_final else "Por determinar"
else:
    campeon = "Por determinar"


# ─── Mostrar llave completa ───────────────────────────────────
print("\n[5/6] Presentando bracket oficial en consola...")
imprimir_bracket_conectado(resultados_llave)

final_row = next((r for r in resultados_llave if r["Ronda"] == "Final"), None)
prob_campeon = final_row["Probabilidad Victoria"] if final_row else 0.0

texto_animado(color_text("╔" + "═"*58 + "╗", ANSI_VERDE))
texto_animado(color_text("║" + "  🏆  CAMPEÓN MUNDIALISTA 2026  🏆  ".center(58) + "║", ANSI_DORADO, ANSI_BOLD))
texto_animado(color_text("╚" + "═"*58 + "╝", ANSI_VERDE))
print(color_text(f"  CAMPEÓN: {campeon}", ANSI_DORADO, ANSI_BOLD))
print(color_text(f"  Probabilidad de victoria en la Final: {prob_campeon:.0%}", ANSI_VERDE))

campeon_stats = [r for r in resultados_llave if r["Equipo Local"] == campeon or r["Equipo Visitante"] == campeon]
if campeon_stats:
    goles_totales = sum(r["Goles Local"] + r["Goles Visitante"] for r in campeon_stats)
    partidos_jugados = len(campeon_stats)
    promedio_campeon = goles_totales / partidos_jugados
    print(color_text(f"  Partidos del campeón: {partidos_jugados}", ANSI_BLANCO))
    print(color_text(f"  Goles totales en partidos del campeón: {goles_totales}", ANSI_BLANCO))
    print(color_text(f"  Promedio de goles por partido del campeón: {promedio_campeon:.2f}", ANSI_BLANCO))

estadisticas = calcular_avance_probabilidades(resultados_llave)
favoritos = ordenar_favoritos(estadisticas)
mostrar_top5_favoritos(favoritos)

promedio_goles, partido_mayor_prob, mayor_sorpresa = resumen_torneo(resultados_llave)
print(color_text("RESUMEN EJECUTIVO", ANSI_DORADO, ANSI_BOLD))
print(color_text("─"*62, ANSI_VERDE))
print(color_text(f"  Promedio de goles del torneo: {promedio_goles:.2f}", ANSI_BLANCO))
print(color_text(f"  Partido con mayor probabilidad: {partido_mayor_prob['Equipo Local']} vs {partido_mayor_prob['Equipo Visitante']} ({max(partido_mayor_prob['Probabilidad Local'], partido_mayor_prob['Probabilidad Visitante']):.0%})", ANSI_BLANCO))
print(color_text(f"  Mayor sorpresa: {mayor_sorpresa['Equipo Local']} vs {mayor_sorpresa['Equipo Visitante']} (ganó {mayor_sorpresa['Ganador']}, probabilidad {min(mayor_sorpresa['Probabilidad Local'], mayor_sorpresa['Probabilidad Visitante']):.0%})", ANSI_BLANCO))

# ─── Guardar resultados ───────────────────────────────────────
print("\n[6/6] Guardando resultados de la simulación...")
df_resultados = pd.DataFrame(resultados_llave)

ruta_pred = os.path.join(PROCESADOS, "predicciones_llave.pkl")
with open(ruta_pred, "wb") as f:
    pickle.dump({
        "resultados": df_resultados,
        "campeon": campeon,
        "cruces_octavos": cruces_octavos,
    }, f)

print(f"  ✓ predicciones_llave.pkl guardado")
print(f"\n{'═'*60}")
print(f"  SIMULACIÓN COMPLETADA")
print(f"  CAMPEÓN PREDICHO: {campeon}")
print("═"*60 + "\n")