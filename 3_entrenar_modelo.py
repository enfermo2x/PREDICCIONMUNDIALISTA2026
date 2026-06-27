"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 3: Entrenamiento de Modelos
==============================================================
Descripción: Entrena un clasificador Random Forest para predecir
el resultado y modelos de Poisson para predecir el marcador.
==============================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from sklearn.ensemble import VotingClassifier
except ImportError:
    pass

# ─── Configuración de rutas ───────────────────────────────────
RAIZ = os.path.dirname(os.path.abspath(__file__))
PROCESADOS = os.path.join(RAIZ, "procesados")

FEATURES_MODELO = [
    "dif_elo",
    "dif_ranking_fifa",
    "dif_puntos_fifa",
    "tasa_h2h_local",
    "goles_anotados_local",
    "goles_recibidos_local",
    "goles_anotados_visitante",
    "goles_recibidos_visitante",
    "dif_goles_netos",
    "tasa_penales_local",
    "tasa_penales_visitante",
    "dif_titulos",
    "elo_local",
    "elo_visitante",
]


def cargar_pickle(nombre):
    ruta = os.path.join(PROCESADOS, f"{nombre}.pkl")
    if not os.path.exists(ruta):
        print(f"  ⚠ {nombre}.pkl no encontrado. Ejecute primero los scripts anteriores.")
        return None
    with open(ruta, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PREDICCIÓN MUNDIALISTA 2026 - ENTRENAMIENTO DE MODELOS")
print("═"*60)

# ─── Cargar datos ─────────────────────────────────────────────
print("\n[1/7] Cargando datasets y features...")
matches_wc    = cargar_pickle("matches_wc")
features_df   = cargar_pickle("features")
stats_ind     = cargar_pickle("stats_individuales")

if matches_wc is None or features_df is None:
    print("  ✗ Error crítico: faltan datos. Ejecute los scripts 1 y 2 primero.")
    sys.exit(1)

# ─── Preparar datos de entrenamiento ──────────────────────────
print("\n[2/7] Preparando datos de entrenamiento desde partidos eliminatorios...")

RONDAS_KO = ["Round of 16", "Quarter-finals", "Semi-finals", "Final",
              "Third place", "Round of 32", "Play-off for third place"]

col_round = next((c for c in ["Round", "round", "stage"] if c in matches_wc.columns), None)
if col_round:
    mask_ko = matches_wc[col_round].str.contains(
        "Round of 16|Quarter|Semi|Final|Third|Play-off|third",
        case=False, na=False
    )
    df_train_raw = matches_wc[mask_ko].copy()
else:
    df_train_raw = matches_wc.copy()

print(f"  → {len(df_train_raw)} partidos eliminatorios disponibles para entrenamiento")

# Detectar columnas de goles
col_hs = next((c for c in ["home_score", "home_goals"] if c in matches_wc.columns), None)
col_as = next((c for c in ["away_score", "away_goals"] if c in matches_wc.columns), None)

if not col_hs or not col_as:
    print("  ✗ Error: no se encuentran columnas de goles en matches_1930_2022.csv")
    sys.exit(1)

df_train_raw = df_train_raw.dropna(subset=[col_hs, col_as])
df_train_raw[col_hs] = pd.to_numeric(df_train_raw[col_hs], errors="coerce").fillna(0)
df_train_raw[col_as] = pd.to_numeric(df_train_raw[col_as], errors="coerce").fillna(0)

# Etiqueta: 1 = victoria local, 0 = empate, -1 = victoria visitante
def etiquetar_resultado(row):
    hs = row[col_hs]
    as_ = row[col_as]
    if hs > as_:
        return "victoria_local"
    elif hs < as_:
        return "victoria_visitante"
    else:
        return "empate"

df_train_raw["resultado"] = df_train_raw.apply(etiquetar_resultado, axis=1)

# ─── Unir con features ────────────────────────────────────────
print("\n[3/7] Uniendo partidos históricos con la matriz de features...")

def get_features_par(equipo_a, equipo_b, features_df, stats_ind):
    """Obtiene o calcula features para un par de equipos."""
    fila = features_df[
        (features_df["equipo_local"] == equipo_a) &
        (features_df["equipo_visitante"] == equipo_b)
    ]
    if not fila.empty:
        return fila.iloc[0]
    # Intentar en orden inverso y negar diferencias
    fila_inv = features_df[
        (features_df["equipo_local"] == equipo_b) &
        (features_df["equipo_visitante"] == equipo_a)
    ]
    if not fila_inv.empty:
        r = fila_inv.iloc[0].copy()
        r["dif_elo"] = -r["dif_elo"]
        r["dif_ranking_fifa"] = -r["dif_ranking_fifa"]
        r["dif_puntos_fifa"] = -r["dif_puntos_fifa"]
        r["dif_goles_netos"] = -r["dif_goles_netos"]
        r["dif_titulos"] = -r["dif_titulos"]
        r["tasa_h2h_local"] = r["tasa_h2h_visitante"]
        r["tasa_h2h_visitante"] = 1.0 - r["tasa_h2h_local"]
        r["elo_local"], r["elo_visitante"] = r["elo_visitante"], r["elo_local"]
        r["goles_anotados_local"], r["goles_anotados_visitante"] = r["goles_anotados_visitante"], r["goles_anotados_local"]
        r["goles_recibidos_local"], r["goles_recibidos_visitante"] = r["goles_recibidos_visitante"], r["goles_recibidos_local"]
        return r
    return None


# Construir X_train con features de los partidos históricos
X_rows = []
y_goles_local = []
y_goles_visitante = []
y_resultado = []

stats = stats_ind if stats_ind else {}
elo_dict = stats.get("elo", {})
rank_dict = stats.get("ranking_fifa", {})
pts_dict = stats.get("puntos_fifa", {})
stats_goles = stats.get("stats_goles", {})
pen_dict = stats.get("tasa_penales", {})
tit_dict = stats.get("titulos", {})
elo_medio = stats.get("elo_medio", 1500.0)
rank_medio = stats.get("rank_medio", 50.0)
puntos_medio = stats.get("puntos_medio", 1000.0)

for _, row in df_train_raw.iterrows():
    eq_a = row["home_team"]
    eq_b = row["away_team"]

    feat = get_features_par(eq_a, eq_b, features_df, stats_ind)

    if feat is not None:
        fila_feat = {f: feat.get(f, 0.0) for f in FEATURES_MODELO}
    else:
        # Calcular desde diccionarios
        elo_a = elo_dict.get(eq_a, elo_medio)
        elo_b = elo_dict.get(eq_b, elo_medio)
        if pd.isna(elo_a): elo_a = elo_medio
        if pd.isna(elo_b): elo_b = elo_medio
        rank_a = rank_dict.get(eq_a, rank_medio)
        rank_b = rank_dict.get(eq_b, rank_medio)
        if pd.isna(rank_a): rank_a = rank_medio
        if pd.isna(rank_b): rank_b = rank_medio
        pts_a = pts_dict.get(eq_a, puntos_medio)
        pts_b = pts_dict.get(eq_b, puntos_medio)
        if pd.isna(pts_a): pts_a = puntos_medio
        if pd.isna(pts_b): pts_b = puntos_medio
        ga_a, gr_a = stats_goles.get(eq_a, (1.0, 1.0))
        ga_b, gr_b = stats_goles.get(eq_b, (1.0, 1.0))

        fila_feat = {
            "dif_elo": elo_a - elo_b,
            "dif_ranking_fifa": rank_b - rank_a,
            "dif_puntos_fifa": pts_a - pts_b,
            "tasa_h2h_local": 0.5,
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

    X_rows.append(fila_feat)
    y_resultado.append(row["resultado"])
    y_goles_local.append(int(row[col_hs]))
    y_goles_visitante.append(int(row[col_as]))

X = pd.DataFrame(X_rows, columns=FEATURES_MODELO).fillna(0.0)
y = np.array(y_resultado)
y_gl = np.array(y_goles_local)
y_gv = np.array(y_goles_visitante)

print(f"  → Dataset de entrenamiento: {len(X)} partidos × {len(FEATURES_MODELO)} features")
print(f"  → Distribución de resultados:")
for res, cnt in zip(*np.unique(y, return_counts=True)):
    print(f"    {res}: {cnt} ({100*cnt/len(y):.1f}%)")

# ─── Entrenar Random Forest ───────────────────────────────────
print("\n[4/7] Entrenando clasificador Random Forest...")

pipeline_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_rf = cross_val_score(pipeline_rf, X, y, cv=cv, scoring="accuracy")

pipeline_rf.fit(X, y)

print(f"  → Precisión en validación cruzada (5-fold): {scores_rf.mean():.3f} ± {scores_rf.std():.3f}")
print(f"  → Puntuaciones por fold: {[f'{s:.3f}' for s in scores_rf]}")

# Importancia de variables
importancias = pipeline_rf.named_steps["clf"].feature_importances_
print("\n  📊 Importancia de variables (Random Forest):")
idx_sorted = np.argsort(importancias)[::-1]
for i in idx_sorted:
    nombre_var = FEATURES_MODELO[i]
    nombres_legibles = {
        "dif_elo": "Diferencia de rating ELO",
        "dif_ranking_fifa": "Diferencia de ranking FIFA",
        "dif_puntos_fifa": "Diferencia de puntos FIFA",
        "tasa_h2h_local": "Tasa H2H (historial directo)",
        "goles_anotados_local": "Goles anotados por partido (local)",
        "goles_recibidos_local": "Goles recibidos por partido (local)",
        "goles_anotados_visitante": "Goles anotados por partido (visitante)",
        "goles_recibidos_visitante": "Goles recibidos por partido (visitante)",
        "dif_goles_netos": "Diferencia de goles netos",
        "tasa_penales_local": "Tasa de éxito en penales (local)",
        "tasa_penales_visitante": "Tasa de éxito en penales (visitante)",
        "dif_titulos": "Diferencia de títulos mundiales",
        "elo_local": "Rating ELO absoluto (local)",
        "elo_visitante": "Rating ELO absoluto (visitante)",
    }
    print(f"    {nombres_legibles.get(nombre_var, nombre_var):<45} {importancias[i]:.4f}")

# ─── Modelos de Poisson para goles ────────────────────────────
print("\n[5/7] Entrenando modelos de Poisson para predicción de goles...")

try:
    from sklearn.linear_model import PoissonRegressor

    modelo_poisson_local = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", PoissonRegressor(alpha=0.5, max_iter=500))
    ])
    modelo_poisson_visitante = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", PoissonRegressor(alpha=0.5, max_iter=500))
    ])

    modelo_poisson_local.fit(X, y_gl)
    modelo_poisson_visitante.fit(X, y_gv)

    pred_gl = modelo_poisson_local.predict(X)
    pred_gv = modelo_poisson_visitante.predict(X)
    mae_local = np.mean(np.abs(pred_gl - y_gl))
    mae_visit = np.mean(np.abs(pred_gv - y_gv))
    print(f"  → MAE goles locales:     {mae_local:.3f}")
    print(f"  → MAE goles visitantes:  {mae_visit:.3f}")
    print("  ✓ Modelos Poisson entrenados exitosamente")
    poisson_ok = True

except Exception as e:
    print(f"  ⚠ Error con PoissonRegressor: {e}")
    print("  → Usando regresión lineal como alternativa...")
    from sklearn.linear_model import Ridge

    modelo_poisson_local = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", Ridge(alpha=1.0))
    ])
    modelo_poisson_visitante = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", Ridge(alpha=1.0))
    ])
    modelo_poisson_local.fit(X, y_gl)
    modelo_poisson_visitante.fit(X, y_gv)
    poisson_ok = True
    print("  ✓ Modelos de regresión entrenados como alternativa")

# ─── Guardar modelos ──────────────────────────────────────────
print("\n[6/7] Guardando modelos entrenados...")

with open(os.path.join(PROCESADOS, "modelo_rf.pkl"), "wb") as f:
    pickle.dump(pipeline_rf, f)

with open(os.path.join(PROCESADOS, "modelo_poisson.pkl"), "wb") as f:
    pickle.dump({
        "local": modelo_poisson_local,
        "visitante": modelo_poisson_visitante
    }, f)

with open(os.path.join(PROCESADOS, "features_modelo.pkl"), "wb") as f:
    pickle.dump(FEATURES_MODELO, f)

print("  ✓ modelo_rf.pkl guardado")
print("  ✓ modelo_poisson.pkl guardado")
print("  ✓ features_modelo.pkl guardado")

# ─── Reporte final ────────────────────────────────────────────
print("\n[7/7] Reporte de clasificación completo:")
y_pred_completo = pipeline_rf.predict(X)
clases = pipeline_rf.classes_
nombres_clases = {
    "victoria_local": "Victoria Local",
    "empate": "Empate",
    "victoria_visitante": "Victoria Visitante"
}
print(classification_report(
    y, y_pred_completo,
    target_names=[nombres_clases.get(c, c) for c in clases]
))

print(f"\n{'═'*60}")
print(f"  MODELOS ENTRENADOS EXITOSAMENTE")
print(f"  → Random Forest: {scores_rf.mean():.1%} precisión promedio (CV)")
print(f"  → Modelos Poisson: listos para predicción de marcadores")
print("═"*60 + "\n")
