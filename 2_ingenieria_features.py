"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 2: Ingeniería de Features
==============================================================
Descripción: Calcula variables predictoras para cada par de
selecciones participantes en el Mundial 2026.
==============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

# ─── Configuración de rutas ───────────────────────────────────
RAIZ = os.path.dirname(os.path.abspath(__file__))
PROCESADOS = os.path.join(RAIZ, "procesados")


def cargar_pickle(nombre):
    """Carga un dataset procesado desde pickle."""
    ruta = os.path.join(PROCESADOS, f"{nombre}.pkl")
    if not os.path.exists(ruta):
        print(f"  ⚠ Advertencia: {nombre}.pkl no encontrado. Ejecute primero 1_cargar_datos.py")
        return None
    with open(ruta, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PREDICCIÓN MUNDIALISTA 2026 - INGENIERÍA DE FEATURES")
print("═"*60)

# ─── Cargar datasets ──────────────────────────────────────────
print("\n[1/8] Cargando datasets procesados...")
elo_rankings     = cargar_pickle("elo_rankings")
fifa_rankings    = cargar_pickle("fifa_rankings")
fifa_rank_2026   = cargar_pickle("fifa_ranking_2026")
matches_wc       = cargar_pickle("matches_wc")
shootouts        = cargar_pickle("shootouts")
world_cup        = cargar_pickle("world_cup")
wc2026           = cargar_pickle("wc2026")
schedule_2026    = cargar_pickle("schedule_2026")

# ─── Obtener lista de participantes del Mundial 2026 ──────────
print("\n[2/8] Identificando selecciones participantes...")
equipos_2026 = set()

if wc2026 is not None:
    if "Home" in wc2026.columns:
        equipos_2026.update(wc2026["Home"].dropna().unique())
    if "Away" in wc2026.columns:
        equipos_2026.update(wc2026["Away"].dropna().unique())

if schedule_2026 is not None:
    cols_team = [c for c in schedule_2026.columns if "team" in c.lower() or c in ["home_team", "away_team"]]
    for c in cols_team:
        equipos_2026.update(schedule_2026[c].dropna().unique())

equipos_2026 = sorted([e for e in equipos_2026 if pd.notna(e) and str(e).strip() != ""])
print(f"  → {len(equipos_2026)} selecciones identificadas:")
for i, eq in enumerate(equipos_2026):
    print(f"    {i+1:2d}. {eq}")


# ══════════════════════════════════════════════════════════════
# HELPERS DE LOOKUP
# ══════════════════════════════════════════════════════════════

def _col_team(df, candidates=("team", "Team", "country", "Country")):
    """Retorna el nombre de la columna de equipo en un df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_elo(equipo):
    """Retorna el rating ELO de un equipo."""
    if elo_rankings is None:
        return np.nan
    col = _col_team(elo_rankings)
    if col is None:
        return np.nan
    fila = elo_rankings[elo_rankings[col] == equipo]
    if fila.empty:
        return np.nan
    col_rating = next((c for c in ["rating", "Rating", "elo", "ELO"] if c in elo_rankings.columns), None)
    return float(fila[col_rating].values[0]) if col_rating else np.nan


def get_ranking_fifa(equipo):
    """Retorna el ranking FIFA de un equipo."""
    if fifa_rankings is None:
        return np.nan
    col = _col_team(fifa_rankings)
    if col is None:
        return np.nan
    fila = fifa_rankings[fifa_rankings[col] == equipo]
    if fila.empty:
        return np.nan
    col_rank = next((c for c in ["rank", "Rank", "position"] if c in fifa_rankings.columns), None)
    return float(fila[col_rank].values[0]) if col_rank else np.nan


def get_puntos_fifa(equipo):
    """Retorna los puntos FIFA de un equipo (ranking 2026)."""
    if fifa_rank_2026 is None:
        return np.nan
    col = _col_team(fifa_rank_2026, ("team", "Team", "country", "Country", "association"))
    if col is None:
        return np.nan
    fila = fifa_rank_2026[fifa_rank_2026[col] == equipo]
    if fila.empty:
        return np.nan
    col_pts = next((c for c in ["points", "Points", "total_points"] if c in fifa_rank_2026.columns), None)
    return float(fila[col_pts].values[0]) if col_pts else np.nan


# ─── Partidos eliminatorios históricos ────────────────────────
RONDAS_ELIMINATORIAS = [
    "Round of 16", "Quarter-finals", "Semi-finals", "Final",
    "Third place", "Octavos", "Cuartos", "Semifinal", "Final",
    "Round of 32", "Play-off for third place"
]

matches_ko = pd.DataFrame()
if matches_wc is not None:
    col_round = next((c for c in ["Round", "round", "stage"] if c in matches_wc.columns), None)
    if col_round:
        mask = matches_wc[col_round].str.contains(
            "Round of 16|Quarter|Semi|Final|Octavos|Cuartos|third|Third|Play-off",
            case=False, na=False
        )
        matches_ko = matches_wc[mask].copy()
        print(f"\n  → {len(matches_ko)} partidos eliminatorios históricos identificados")


def get_head_to_head(equipo_a, equipo_b):
    """Tasa de victoria de equipo_a vs equipo_b en fases eliminatorias del Mundial."""
    if matches_ko.empty:
        return 0.5
    partidos = matches_ko[
        ((matches_ko["home_team"] == equipo_a) & (matches_ko["away_team"] == equipo_b)) |
        ((matches_ko["home_team"] == equipo_b) & (matches_ko["away_team"] == equipo_a))
    ]
    if partidos.empty:
        return 0.5  # sin historial: 50%
    victorias_a = 0
    for _, row in partidos.iterrows():
        hs = row.get("home_score", 0) or 0
        as_ = row.get("away_score", 0) or 0
        if row["home_team"] == equipo_a:
            if hs > as_:
                victorias_a += 1
        else:
            if as_ > hs:
                victorias_a += 1
    return victorias_a / len(partidos)


def get_stats_mundial(equipo):
    """Promedio de goles anotados y recibidos en Mundiales (todos los partidos)."""
    if matches_wc is None:
        return 0.0, 0.0
    como_local = matches_wc[matches_wc["home_team"] == equipo]
    como_visitante = matches_wc[matches_wc["away_team"] == equipo]
    total = len(como_local) + len(como_visitante)
    if total == 0:
        return 0.0, 0.0
    col_hs = next((c for c in ["home_score", "home_goals"] if c in matches_wc.columns), None)
    col_as = next((c for c in ["away_score", "away_goals"] if c in matches_wc.columns), None)
    if not col_hs or not col_as:
        return 0.0, 0.0
    goles_a = (como_local[col_hs].sum() + como_visitante[col_as].sum())
    goles_r = (como_local[col_as].sum() + como_visitante[col_hs].sum())
    return float(goles_a / total), float(goles_r / total)


def get_tasa_penales(equipo):
    """Tasa de victorias en penales."""
    if shootouts is None:
        return 0.5
    col_w = next((c for c in ["winner", "Winner"] if c in shootouts.columns), None)
    if not col_w:
        return 0.5
    partidos_penales = shootouts[
        (shootouts.get("home_team", pd.Series()) == equipo) |
        (shootouts.get("away_team", pd.Series()) == equipo)
    ]
    if partidos_penales.empty:
        return 0.5
    victorias = (partidos_penales[col_w] == equipo).sum()
    return float(victorias / len(partidos_penales))


def get_titulos(equipo):
    """Número de títulos mundiales."""
    if world_cup is None:
        return 0
    col = next((c for c in ["Champion", "champion", "Winner"] if c in world_cup.columns), None)
    if not col:
        return 0
    return int((world_cup[col] == equipo).sum())


# ─── Calcular features para todos los pares ───────────────────
print("\n[3/8] Calculando ranking ELO por equipo...")
elo_dict = {eq: get_elo(eq) for eq in equipos_2026}
elo_vals = [v for v in elo_dict.values() if not np.isnan(v)]
elo_medio = float(np.mean(elo_vals)) if elo_vals else 1500.0

print("\n[4/8] Calculando rankings FIFA por equipo...")
rank_fifa_dict = {eq: get_ranking_fifa(eq) for eq in equipos_2026}
rank_vals = [v for v in rank_fifa_dict.values() if not np.isnan(v)]
rank_medio = float(np.mean(rank_vals)) if rank_vals else 50.0

print("\n[5/8] Calculando puntos FIFA por equipo...")
puntos_dict = {eq: get_puntos_fifa(eq) for eq in equipos_2026}
puntos_vals = [v for v in puntos_dict.values() if not np.isnan(v)]
puntos_medio = float(np.mean(puntos_vals)) if puntos_vals else 1000.0

print("\n[6/8] Calculando estadísticas de goles en Mundiales...")
stats_dict = {eq: get_stats_mundial(eq) for eq in equipos_2026}

print("\n[7/8] Calculando tasas de penales y títulos...")
penales_dict = {eq: get_tasa_penales(eq) for eq in equipos_2026}
titulos_dict = {eq: get_titulos(eq) for eq in equipos_2026}

print("\n[8/8] Construyendo matriz de features para todos los cruces posibles...")

filas = []
pares = list(combinations(equipos_2026, 2))
print(f"  → Calculando {len(pares)} pares de equipos...")

for idx, (eq_a, eq_b) in enumerate(pares):
    if idx % 100 == 0:
        print(f"  ... procesando par {idx}/{len(pares)}")

    # ELO
    elo_a = elo_dict.get(eq_a, elo_medio)
    elo_b = elo_dict.get(eq_b, elo_medio)
    if np.isnan(elo_a): elo_a = elo_medio
    if np.isnan(elo_b): elo_b = elo_medio

    # Ranking FIFA
    rank_a = rank_fifa_dict.get(eq_a, rank_medio)
    rank_b = rank_fifa_dict.get(eq_b, rank_medio)
    if np.isnan(rank_a): rank_a = rank_medio
    if np.isnan(rank_b): rank_b = rank_medio

    # Puntos FIFA
    pts_a = puntos_dict.get(eq_a, puntos_medio)
    pts_b = puntos_dict.get(eq_b, puntos_medio)
    if np.isnan(pts_a): pts_a = puntos_medio
    if np.isnan(pts_b): pts_b = puntos_medio

    # Estadísticas mundiales
    ga_a, gr_a = stats_dict.get(eq_a, (0.0, 0.0))
    ga_b, gr_b = stats_dict.get(eq_b, (0.0, 0.0))

    # Head-to-head
    h2h_a = get_head_to_head(eq_a, eq_b)

    # Penales y títulos
    pen_a = penales_dict.get(eq_a, 0.5)
    pen_b = penales_dict.get(eq_b, 0.5)
    tit_a = titulos_dict.get(eq_a, 0)
    tit_b = titulos_dict.get(eq_b, 0)

    filas.append({
        "equipo_local": eq_a,
        "equipo_visitante": eq_b,
        "elo_local": elo_a,
        "elo_visitante": elo_b,
        "dif_elo": elo_a - elo_b,
        "ranking_fifa_local": rank_a,
        "ranking_fifa_visitante": rank_b,
        "dif_ranking_fifa": rank_b - rank_a,  # positivo = local mejor clasificado
        "puntos_fifa_local": pts_a,
        "puntos_fifa_visitante": pts_b,
        "dif_puntos_fifa": pts_a - pts_b,
        "tasa_h2h_local": h2h_a,
        "tasa_h2h_visitante": 1.0 - h2h_a,
        "goles_anotados_local": ga_a,
        "goles_recibidos_local": gr_a,
        "goles_anotados_visitante": ga_b,
        "goles_recibidos_visitante": gr_b,
        "dif_goles_netos": (ga_a - gr_a) - (ga_b - gr_b),
        "tasa_penales_local": pen_a,
        "tasa_penales_visitante": pen_b,
        "titulos_local": tit_a,
        "titulos_visitante": tit_b,
        "dif_titulos": tit_a - tit_b,
    })

features = pd.DataFrame(filas)

# Guardar
ruta_feat = os.path.join(PROCESADOS, "features.csv")
features.to_csv(ruta_feat, index=False, encoding="utf-8")

ruta_feat_pkl = os.path.join(PROCESADOS, "features.pkl")
with open(ruta_feat_pkl, "wb") as f:
    pickle.dump(features, f)

# Guardar diccionarios de stats individuales
stats_individuales = {
    "elo": elo_dict,
    "ranking_fifa": rank_fifa_dict,
    "puntos_fifa": puntos_dict,
    "stats_goles": stats_dict,
    "tasa_penales": penales_dict,
    "titulos": titulos_dict,
    "elo_medio": elo_medio,
    "rank_medio": rank_medio,
    "puntos_medio": puntos_medio,
}
with open(os.path.join(PROCESADOS, "stats_individuales.pkl"), "wb") as f:
    pickle.dump(stats_individuales, f)

print(f"\n{'═'*60}")
print(f"  FEATURES COMPLETADOS")
print(f"  → {len(features)} pares de equipos × {len(features.columns)} variables")
print(f"  → Guardado en: {ruta_feat}")
print(f"  Columnas generadas:")
for c in features.columns:
    print(f"    · {c}")
print("═"*60 + "\n")
