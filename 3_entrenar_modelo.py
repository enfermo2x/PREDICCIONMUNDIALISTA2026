"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 3: Entrenamiento de Modelos
==============================================================
Entrena CINCO modelos usando partidos eliminatorios históricos:
  1. Random Forest (400 árboles, max_depth=8, balanced)
  2. XGBoost / GradientBoosting fallback (300 estim, lr=0.05)
  3. SVM Calibrado (SVC RBF + CalibratedClassifierCV)
  4. Stacking Ensemble (RF+SVM base, LogisticRegression meta)
  5. Regresión de Poisson (local + visitante, Ridge fallback)

Guarda cada modelo como pickle separado en procesados/.
Imprime tabla comparativa de accuracy CV al final.
==============================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# ─── Intentar importar XGBoost ────────────────────────────────
XGB_OK = False
try:
    from xgboost import XGBClassifier
    XGB_OK = True
    print("  ✓ XGBoost disponible")
except ImportError:
    print("  ⚠ XGBoost no instalado → se usará GradientBoostingClassifier como fallback")
    print("    Para instalar: pip install xgboost")

# ─── Intentar importar PoissonRegressor ───────────────────────
POISSON_OK = False
try:
    from sklearn.linear_model import PoissonRegressor
    POISSON_OK = True
except ImportError:
    print("  ⚠ PoissonRegressor no disponible → se usará Ridge como fallback")

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
    """Carga un pickle desde la carpeta procesados/."""
    ruta = os.path.join(PROCESADOS, f"{nombre}.pkl")
    if not os.path.exists(ruta):
        print(f"  ⚠ {nombre}.pkl no encontrado. Ejecute primero los scripts anteriores.")
        return None
    with open(ruta, "rb") as f:
        return pickle.load(f)


def guardar_pickle(obj, nombre):
    """Guarda un objeto como pickle en procesados/."""
    ruta = os.path.join(PROCESADOS, nombre)
    with open(ruta, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓ {nombre} guardado")


def get_features_par(eq_a, eq_b, features_df, stats_ind):
    """Obtiene o calcula features para un par de equipos."""
    # Buscar en la matriz precalculada
    fila = features_df[
        (features_df["equipo_local"] == eq_a) &
        (features_df["equipo_visitante"] == eq_b)
    ]
    if not fila.empty:
        return fila.iloc[0]

    # Intentar orden inverso y negar diferencias
    fila_inv = features_df[
        (features_df["equipo_local"] == eq_b) &
        (features_df["equipo_visitante"] == eq_a)
    ]
    if not fila_inv.empty:
        r = fila_inv.iloc[0].copy()
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
        return r

    return None


# ══════════════════════════════════════════════════════════════
#  INICIO
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PREDICCIÓN MUNDIALISTA 2026 - ENTRENAMIENTO DE MODELOS")
print("═"*60)

# ─── [1/8] Cargar datos ──────────────────────────────────────
print("\n[1/8] Cargando datasets y features...")

matches_wc  = cargar_pickle("matches_wc")
features_df = cargar_pickle("features")
stats_ind   = cargar_pickle("stats_individuales")

if matches_wc is None or features_df is None:
    print("  ✗ Error crítico: faltan datos.")
    print("    → Ejecute primero 1_cargar_datos.py y 2_calcular_features.py")
    sys.exit(1)

# ─── [2/8] Filtrar partidos eliminatorios ─────────────────────
print("\n[2/8] Filtrando partidos eliminatorios (rondas KO)...")

col_round = next((c for c in ["Round", "round", "stage"] if c in matches_wc.columns), None)
if col_round:
    mask_ko = matches_wc[col_round].str.contains(
        "Round of 16|Quarter|Semi|Final|Third|Play-off|third|Round of 32",
        case=False, na=False
    )
    df_train_raw = matches_wc[mask_ko].copy()
else:
    df_train_raw = matches_wc.copy()

print(f"  → {len(df_train_raw)} partidos eliminatorios encontrados")

# ─── [3/8] Preparar etiquetas ─────────────────────────────────
print("\n[3/8] Preparando etiquetas de resultado...")

col_hs = next((c for c in ["home_score", "home_goals"] if c in matches_wc.columns), None)
col_as = next((c for c in ["away_score", "away_goals"] if c in matches_wc.columns), None)

if not col_hs or not col_as:
    print("  ✗ Error: no se encuentran columnas de goles en matches_wc")
    sys.exit(1)

df_train_raw = df_train_raw.dropna(subset=[col_hs, col_as])
df_train_raw[col_hs] = pd.to_numeric(df_train_raw[col_hs], errors="coerce").fillna(0)
df_train_raw[col_as] = pd.to_numeric(df_train_raw[col_as], errors="coerce").fillna(0)


def etiquetar_resultado(row):
    hs, as_ = row[col_hs], row[col_as]
    if hs > as_:
        return "victoria_local"
    elif hs < as_:
        return "victoria_visitante"
    return "empate"


df_train_raw["resultado"] = df_train_raw.apply(etiquetar_resultado, axis=1)

# ─── [4/8] Construir matriz de features ───────────────────────
print("\n[4/8] Construyendo matriz de features para entrenamiento...")

# Diccionarios de fallback
stats = stats_ind if stats_ind else {}
elo_dict     = stats.get("elo", {})
rank_dict    = stats.get("ranking_fifa", {})
pts_dict     = stats.get("puntos_fifa", {})
stats_goles  = stats.get("stats_goles", {})
pen_dict     = stats.get("tasa_penales", {})
tit_dict     = stats.get("titulos", {})
elo_medio    = stats.get("elo_medio", 1500.0)
rank_medio   = stats.get("rank_medio", 50.0)
puntos_medio = stats.get("puntos_medio", 1000.0)

X_rows = []
y_resultado = []
y_goles_local = []
y_goles_visitante = []

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
        ga_a, gr_a = stats_goles.get(eq_a, (1.2, 1.0))
        ga_b, gr_b = stats_goles.get(eq_b, (1.2, 1.0))

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

print(f"  → Dataset: {len(X)} partidos × {len(FEATURES_MODELO)} features")
print(f"  → Distribución de resultados:")
for res, cnt in zip(*np.unique(y, return_counts=True)):
    print(f"    {res}: {cnt} ({100*cnt/len(y):.1f}%)")

# ─── Codificar etiquetas ──────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Almacenar resultados para tabla comparativa
resultados_cv = {}

# ══════════════════════════════════════════════════════════════
#  MODELO 1: Random Forest
# ══════════════════════════════════════════════════════════════
print("\n[5/8] Entrenando Random Forest (400 árboles)...")

pipe_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

scores_rf = cross_val_score(pipe_rf, X, y_enc, cv=cv, scoring="accuracy")
pipe_rf.fit(X, y_enc)

resultados_cv["Random Forest"] = scores_rf.mean()

print(f"  → Accuracy CV: {scores_rf.mean():.3f} ± {scores_rf.std():.3f}")
print(f"  → Folds: {[f'{s:.3f}' for s in scores_rf]}")

# Importancia de variables
importancias = pipe_rf.named_steps["clf"].feature_importances_
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
idx_sorted = np.argsort(importancias)[::-1]
print("\n  📊 Top 5 variables más importantes:")
for i in idx_sorted[:5]:
    print(f"    {nombres_legibles.get(FEATURES_MODELO[i], FEATURES_MODELO[i]):<45} {importancias[i]:.4f}")

guardar_pickle(pipe_rf, "modelo_rf.pkl")

# ══════════════════════════════════════════════════════════════
#  MODELO 2: XGBoost / GradientBoosting fallback
# ══════════════════════════════════════════════════════════════
print("\n  ─── Modelo 2: XGBoost ───")

if XGB_OK:
    print("  Usando XGBClassifier...")
    pipe_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
else:
    print("  ⚠ XGBoost no disponible → usando GradientBoostingClassifier como fallback transparente")
    pipe_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ])

scores_xgb = cross_val_score(pipe_xgb, X, y_enc, cv=cv, scoring="accuracy")
pipe_xgb.fit(X, y_enc)

resultados_cv["XGBoost" if XGB_OK else "GradientBoosting"] = scores_xgb.mean()

print(f"  → Accuracy CV: {scores_xgb.mean():.3f} ± {scores_xgb.std():.3f}")
print(f"  → Folds: {[f'{s:.3f}' for s in scores_xgb]}")

# Guardar modelo XGB con dict que incluye modelo, clases y flag xgb_ok
guardar_pickle({
    "modelo": pipe_xgb,
    "clases": le.classes_,
    "xgb_ok": XGB_OK
}, "modelo_xgb.pkl")

# ══════════════════════════════════════════════════════════════
#  MODELO 3: SVM Calibrado
# ══════════════════════════════════════════════════════════════
print("\n  ─── Modelo 3: SVM Calibrado ───")

svm_base = SVC(
    kernel="rbf",
    C=5.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)
pipe_svm = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", CalibratedClassifierCV(
        estimator=svm_base,
        cv=3,
        method="sigmoid"
    ))
])

scores_svm = cross_val_score(pipe_svm, X, y_enc, cv=cv, scoring="accuracy")
pipe_svm.fit(X, y_enc)

resultados_cv["SVM Calibrado"] = scores_svm.mean()

print(f"  → Accuracy CV: {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")
print(f"  → Folds: {[f'{s:.3f}' for s in scores_svm]}")
print(f"  → SVC(kernel=rbf, C=5.0, gamma=scale) envuelto en CalibratedClassifierCV")

guardar_pickle(pipe_svm, "modelo_svm.pkl")

# ══════════════════════════════════════════════════════════════
#  MODELO 4: Stacking Ensemble
# ══════════════════════════════════════════════════════════════
print("\n  ─── Modelo 4: Stacking Ensemble ───")

# Clonar estimadores base para el stacking (no los ya entrenados)
rf_base = RandomForestClassifier(
    n_estimators=400, max_depth=8, min_samples_split=5,
    min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
)
svm_base_stack = CalibratedClassifierCV(
    estimator=SVC(kernel="rbf", C=5.0, gamma="scale",
                  class_weight="balanced", random_state=42),
    cv=3, method="sigmoid"
)

stacking = StackingClassifier(
    estimators=[
        ("rf", rf_base),
        ("svm", svm_base_stack),
    ],
    final_estimator=LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    cv=5,
    n_jobs=-1
)

pipe_stack = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", stacking)
])

scores_stack = cross_val_score(pipe_stack, X, y_enc, cv=cv, scoring="accuracy")
pipe_stack.fit(X, y_enc)

resultados_cv["Stacking Ensemble"] = scores_stack.mean()

print(f"  → Accuracy CV: {scores_stack.mean():.3f} ± {scores_stack.std():.3f}")
print(f"  → Folds: {[f'{s:.3f}' for s in scores_stack]}")
print(f"  → Base: RF(400) + SVM(RBF) | Meta: LogisticRegression | cv=5")

guardar_pickle(pipe_stack, "modelo_stack.pkl")

# ══════════════════════════════════════════════════════════════
#  MODELO 5: Regresión de Poisson (goles local + visitante)
# ══════════════════════════════════════════════════════════════
print("\n[6/8] Entrenando modelos de Poisson para goles...")

if POISSON_OK:
    print("  Usando PoissonRegressor...")
    pipe_poisson_l = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", PoissonRegressor(alpha=0.5, max_iter=500))
    ])
    pipe_poisson_v = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", PoissonRegressor(alpha=0.5, max_iter=500))
    ])
else:
    print("  ⚠ PoissonRegressor no disponible → usando Ridge como fallback")
    pipe_poisson_l = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", Ridge(alpha=1.0))
    ])
    pipe_poisson_v = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", Ridge(alpha=1.0))
    ])

pipe_poisson_l.fit(X, y_gl)
pipe_poisson_v.fit(X, y_gv)

pred_gl = pipe_poisson_l.predict(X)
pred_gv = pipe_poisson_v.predict(X)
mae_local = np.mean(np.abs(pred_gl - y_gl))
mae_visit = np.mean(np.abs(pred_gv - y_gv))

print(f"  → MAE goles locales:     {mae_local:.3f}")
print(f"  → MAE goles visitantes:  {mae_visit:.3f}")
print("  ✓ Modelos Poisson entrenados")

guardar_pickle({
    "local": pipe_poisson_l,
    "visitante": pipe_poisson_v
}, "modelo_poisson.pkl")

# ─── Guardar features y label encoder ─────────────────────────
print("\n[7/8] Guardando archivos auxiliares...")

guardar_pickle(FEATURES_MODELO, "features_modelo.pkl")
guardar_pickle(le, "label_encoder.pkl")

# ══════════════════════════════════════════════════════════════
#  TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════
print("\n[8/8] Tabla comparativa de Accuracy CV (5-fold estratificada):")
print("\n  " + "─"*52)
print(f"  {'Modelo':<28} {'Accuracy CV':>12} {'± std':>10}")
print("  " + "─"*52)

# Ordenar de mejor a peor
for nombre, acc in sorted(resultados_cv.items(), key=lambda x: x[1], reverse=True):
    # Buscar std correspondiente
    if "Random" in nombre:
        std = scores_rf.std()
    elif "XGB" in nombre or "Gradient" in nombre:
        std = scores_xgb.std()
    elif "SVM" in nombre:
        std = scores_svm.std()
    elif "Stacking" in nombre:
        std = scores_stack.std()
    else:
        std = 0.0
    print(f"  {nombre:<28} {acc:>11.3%} {std:>9.3f}")

print("  " + "─"*52)
mejor = max(resultados_cv, key=resultados_cv.get)
print(f"  Mejor modelo: {mejor} ({resultados_cv[mejor]:.3%})")
print()

# ─── Resumen final ────────────────────────────────────────────
print(f"{'═'*60}")
print(f"  MODELOS ENTRENADOS EXITOSAMENTE")
print(f"  → modelo_rf.pkl      : Random Forest (400 árboles)")
print(f"  → modelo_xgb.pkl     : {'XGBoost' if XGB_OK else 'GradientBoosting'} (300 estimadores)")
print(f"  → modelo_svm.pkl     : SVM Calibrado (SVC RBF + CalibratedCV)")
print(f"  → modelo_stack.pkl   : Stacking (RF+SVM → LogisticRegression)")
print(f"  → modelo_poisson.pkl : Poisson (local + visitante)")
print(f"  → features_modelo.pkl: Lista de features usadas")
print(f"  → label_encoder.pkl  : Codificador de etiquetas")
print(f"\n  Próximo paso: python 4_simular_llave.py")
print("═"*60 + "\n")
