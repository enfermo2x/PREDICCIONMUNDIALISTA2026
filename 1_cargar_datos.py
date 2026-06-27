"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 1: Carga de Datos
==============================================================
Descripción: Carga todos los archivos de datos, estandariza
nombres de selecciones y genera versiones limpias en procesados/
==============================================================
"""

import os
import sys
import json
import pickle
import pandas as pd

# ─── Configuración de rutas ───────────────────────────────────
RAIZ = os.path.dirname(os.path.abspath(__file__))
WC_DATA = os.path.join(RAIZ, "wc-data")
FUCHIBOL = os.path.join(WC_DATA, "Data Fuchibol")
PROCESADOS = os.path.join(RAIZ, "procesados")

os.makedirs(PROCESADOS, exist_ok=True)

# ─── Mapa de nombres en español ───────────────────────────────
NOMBRES_ESPANOL = {
    "France": "Francia",
    "Spain": "España",
    "Germany": "Alemania",
    "Brazil": "Brasil",
    "Argentina": "Argentina",
    "England": "Inglaterra",
    "Portugal": "Portugal",
    "Netherlands": "Países Bajos",
    "Morocco": "Marruecos",
    "Japan": "Japón",
    "South Korea": "Corea del Sur",
    "Australia": "Australia",
    "Croatia": "Croacia",
    "Switzerland": "Suiza",
    "Poland": "Polonia",
    "Serbia": "Serbia",
    "Denmark": "Dinamarca",
    "Mexico": "México",
    "USA": "Estados Unidos",
    "United States": "Estados Unidos",
    "Canada": "Canadá",
    "Belgium": "Bélgica",
    "Italy": "Italia",
    "Uruguay": "Uruguay",
    "Colombia": "Colombia",
    "Ecuador": "Ecuador",
    "Chile": "Chile",
    "Peru": "Perú",
    "Bolivia": "Bolivia",
    "Paraguay": "Paraguay",
    "Venezuela": "Venezuela",
    "Costa Rica": "Costa Rica",
    "Panama": "Panamá",
    "Honduras": "Honduras",
    "El Salvador": "El Salvador",
    "Guatemala": "Guatemala",
    "Cameroon": "Camerún",
    "Senegal": "Senegal",
    "Ghana": "Ghana",
    "Nigeria": "Nigeria",
    "Tunisia": "Túnez",
    "Egypt": "Egipto",
    "Algeria": "Argelia",
    "Saudi Arabia": "Arabia Saudita",
    "Iran": "Irán",
    "Australia": "Australia",
    "South Africa": "Sudáfrica",
    "Ivory Coast": "Costa de Marfil",
    "Cote d'Ivoire": "Costa de Marfil",
    "Republic of Ireland": "Irlanda",
    "Czech Republic": "República Checa",
    "Czechia": "República Checa",
    "Slovakia": "Eslovaquia",
    "Slovenia": "Eslovenia",
    "Sweden": "Suecia",
    "Norway": "Noruega",
    "Finland": "Finlandia",
    "Russia": "Rusia",
    "Ukraine": "Ucrania",
    "Turkey": "Turquía",
    "Greece": "Grecia",
    "Romania": "Rumanía",
    "Hungary": "Hungría",
    "Austria": "Austria",
    "Scotland": "Escocia",
    "Wales": "Gales",
    "Albania": "Albania",
    "New Zealand": "Nueva Zelanda",
    "Uzbekistan": "Uzbekistán",
    "Iraq": "Irak",
    "Jordan": "Jordania",
    "Indonesia": "Indonesia",
    "Venezuela": "Venezuela",
    "Jamaica": "Jamaica",
    "Trinidad and Tobago": "Trinidad y Tobago",
    "Cuba": "Cuba",
    "Haiti": "Haití",
    "Burkina Faso": "Burkina Faso",
    "Mali": "Malí",
    "Congo DR": "Congo RD",
    "Democratic Republic of the Congo": "Congo RD",
    "Mozambique": "Mozambique",
    "Angola": "Angola",
    "Tanzania": "Tanzania",
    "Kenya": "Kenia",
    "Ethiopia": "Etiopía",
    "Libya": "Libia",
}


def traducir_nombre(nombre):
    """Convierte nombre de selección al español si existe traducción."""
    if pd.isna(nombre):
        return nombre
    return NOMBRES_ESPANOL.get(str(nombre).strip(), str(nombre).strip())


def leer_csv_robusto(ruta, nombre_archivo):
    """Lee un CSV probando diferentes encodings."""
    for enc in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
        try:
            df = pd.read_csv(ruta, encoding=enc)
            print(f"  ✓ {nombre_archivo} leído con encoding {enc}")
            return df
        except Exception:
            continue
    print(f"  ✗ ERROR: No se pudo leer {nombre_archivo}. Verifique el archivo.")
    return None


def verificar_archivo(ruta, nombre):
    """Verifica que un archivo existe."""
    if not os.path.exists(ruta):
        print(f"\n  ✗ ARCHIVO FALTANTE: {nombre}")
        print(f"    Ruta esperada: {ruta}")
        print(f"    Solución: Copie el archivo a la carpeta indicada.")
        return False
    return True


def mostrar_resumen(df, nombre):
    """Imprime un resumen del DataFrame."""
    if df is None:
        return
    print(f"\n  {'─'*50}")
    print(f"  Dataset: {nombre}")
    print(f"  Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"  Columnas: {list(df.columns)}")
    print(f"  Muestra (3 primeras filas):")
    print(df.head(3).to_string(index=False))


# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PREDICCIÓN MUNDIALISTA 2026 - CARGA DE DATOS")
print("═"*60)

# ─── 1. Cargar former_names.csv ───────────────────────────────
print("\n[1/9] Cargando tabla de nombres históricos...")
ruta_fn = os.path.join(WC_DATA, "former_names.csv")
former_names = None
mapa_nombres = {}
if verificar_archivo(ruta_fn, "former_names.csv"):
    former_names = leer_csv_robusto(ruta_fn, "former_names.csv")
    if former_names is not None:
        for _, row in former_names.iterrows():
            if pd.notna(row.get("former")) and pd.notna(row.get("current")):
                mapa_nombres[str(row["former"]).strip()] = str(row["current"]).strip()
        print(f"  → {len(mapa_nombres)} alias históricos cargados.")
        mostrar_resumen(former_names, "former_names.csv")


def estandarizar_nombre(nombre):
    """Aplica mapa de nombres históricos y luego traduce al español."""
    if pd.isna(nombre):
        return nombre
    n = str(nombre).strip()
    n = mapa_nombres.get(n, n)
    return traducir_nombre(n)


# ─── 2. Cargar results.csv ────────────────────────────────────
print("\n[2/9] Cargando resultados históricos internacionales...")
ruta_res = os.path.join(WC_DATA, "results.csv")
results = None
if verificar_archivo(ruta_res, "results.csv"):
    results = leer_csv_robusto(ruta_res, "results.csv")
    if results is not None:
        results["home_team"] = results["home_team"].apply(estandarizar_nombre)
        results["away_team"] = results["away_team"].apply(estandarizar_nombre)
        mostrar_resumen(results, "results.csv")

# ─── 3. Cargar goalscorers.csv ────────────────────────────────
print("\n[3/9] Cargando goleadores históricos...")
ruta_gs = os.path.join(WC_DATA, "goalscorers.csv")
goalscorers = None
if verificar_archivo(ruta_gs, "goalscorers.csv"):
    goalscorers = leer_csv_robusto(ruta_gs, "goalscorers.csv")
    if goalscorers is not None:
        goalscorers["home_team"] = goalscorers["home_team"].apply(estandarizar_nombre)
        goalscorers["away_team"] = goalscorers["away_team"].apply(estandarizar_nombre)
        goalscorers["team"] = goalscorers["team"].apply(estandarizar_nombre)
        mostrar_resumen(goalscorers, "goalscorers.csv")

# ─── 4. Cargar shootouts.csv ──────────────────────────────────
print("\n[4/9] Cargando tandas de penales...")
ruta_sh = os.path.join(WC_DATA, "shootouts.csv")
shootouts = None
if verificar_archivo(ruta_sh, "shootouts.csv"):
    shootouts = leer_csv_robusto(ruta_sh, "shootouts.csv")
    if shootouts is not None:
        shootouts["home_team"] = shootouts["home_team"].apply(estandarizar_nombre)
        shootouts["away_team"] = shootouts["away_team"].apply(estandarizar_nombre)
        shootouts["winner"] = shootouts["winner"].apply(estandarizar_nombre)
        mostrar_resumen(shootouts, "shootouts.csv")

# ─── 5. Cargar ELO rankings ───────────────────────────────────
print("\n[5/9] Cargando rankings ELO...")
ruta_elo = os.path.join(WC_DATA, "elo_rankings.json")
elo_rankings = None
if verificar_archivo(ruta_elo, "elo_rankings.json"):
    try:
        with open(ruta_elo, "r", encoding="utf-8") as f:
            elo_data = json.load(f)
        elo_rankings = pd.DataFrame(elo_data)
        elo_rankings["team"] = elo_rankings["team"].apply(estandarizar_nombre)
        print(f"  ✓ ELO rankings cargado: {len(elo_rankings)} equipos")
        mostrar_resumen(elo_rankings, "elo_rankings.json")
    except Exception as e:
        print(f"  ✗ Error al leer elo_rankings.json: {e}")

# ─── 6. Cargar FIFA rankings ──────────────────────────────────
print("\n[6/9] Cargando rankings FIFA actuales...")
ruta_fifa = os.path.join(WC_DATA, "fifa_rankings.json")
fifa_rankings = None
if verificar_archivo(ruta_fifa, "fifa_rankings.json"):
    try:
        with open(ruta_fifa, "r", encoding="utf-8") as f:
            fifa_data = json.load(f)
        fifa_rankings = pd.DataFrame(fifa_data)
        fifa_rankings["team"] = fifa_rankings["team"].apply(estandarizar_nombre)
        print(f"  ✓ FIFA rankings cargado: {len(fifa_rankings)} equipos")
        mostrar_resumen(fifa_rankings, "fifa_rankings.json")
    except Exception as e:
        print(f"  ✗ Error al leer fifa_rankings.json: {e}")

# ─── 7. Cargar partidos históricos de Mundiales ───────────────
print("\n[7/9] Cargando partidos históricos de Mundiales (1930-2022)...")
ruta_matches = os.path.join(FUCHIBOL, "matches_1930_2022.csv")
matches_wc = None
if verificar_archivo(ruta_matches, "matches_1930_2022.csv"):
    matches_wc = leer_csv_robusto(ruta_matches, "matches_1930_2022.csv")
    if matches_wc is not None:
        matches_wc["home_team"] = matches_wc["home_team"].apply(estandarizar_nombre)
        matches_wc["away_team"] = matches_wc["away_team"].apply(estandarizar_nombre)
        mostrar_resumen(matches_wc, "matches_1930_2022.csv")

# ─── 8. Cargar calendario 2026 ────────────────────────────────
print("\n[8/9] Cargando calendario del Mundial 2026...")
ruta_sched = os.path.join(FUCHIBOL, "schedule_2026.csv")
schedule_2026 = None
if verificar_archivo(ruta_sched, "schedule_2026.csv"):
    schedule_2026 = leer_csv_robusto(ruta_sched, "schedule_2026.csv")
    if schedule_2026 is not None:
        if "home_team" in schedule_2026.columns:
            schedule_2026["home_team"] = schedule_2026["home_team"].apply(estandarizar_nombre)
        if "away_team" in schedule_2026.columns:
            schedule_2026["away_team"] = schedule_2026["away_team"].apply(estandarizar_nombre)
        mostrar_resumen(schedule_2026, "schedule_2026.csv")

# ─── 9. Cargar WorldCup2026.xlsx ──────────────────────────────
print("\n[9/9] Cargando datos del Mundial 2026 (resultados fase de grupos)...")
ruta_wc2026 = os.path.join(WC_DATA, "WorldCup2026.xlsx")
wc2026 = None
if verificar_archivo(ruta_wc2026, "WorldCup2026.xlsx"):
    try:
        wc2026 = pd.read_excel(ruta_wc2026)
        if "Home" in wc2026.columns:
            wc2026["Home"] = wc2026["Home"].apply(estandarizar_nombre)
        if "Away" in wc2026.columns:
            wc2026["Away"] = wc2026["Away"].apply(estandarizar_nombre)
        print(f"  ✓ WorldCup2026.xlsx cargado: {wc2026.shape}")
        mostrar_resumen(wc2026, "WorldCup2026.xlsx")
    except Exception as e:
        print(f"  ✗ Error al leer WorldCup2026.xlsx: {e}")

# ─── Cargar ficheros de ranking FIFA extra ────────────────────
ruta_fifa26 = os.path.join(FUCHIBOL, "fifa_ranking_2026-06-08.csv")
fifa_ranking_2026 = None
if os.path.exists(ruta_fifa26):
    fifa_ranking_2026 = leer_csv_robusto(ruta_fifa26, "fifa_ranking_2026-06-08.csv")
    if fifa_ranking_2026 is not None:
        col_team = [c for c in fifa_ranking_2026.columns if "team" in c.lower()]
        if col_team:
            fifa_ranking_2026[col_team[0]] = fifa_ranking_2026[col_team[0]].apply(estandarizar_nombre)

ruta_world_cup = os.path.join(FUCHIBOL, "world_cup.csv")
world_cup = None
if os.path.exists(ruta_world_cup):
    world_cup = leer_csv_robusto(ruta_world_cup, "world_cup.csv")

# ─── Guardar datos procesados ─────────────────────────────────
print("\n" + "─"*60)
print("Guardando datos procesados en procesados/...")

datasets = {
    "results": results,
    "goalscorers": goalscorers,
    "shootouts": shootouts,
    "elo_rankings": elo_rankings,
    "fifa_rankings": fifa_rankings,
    "matches_wc": matches_wc,
    "schedule_2026": schedule_2026,
    "wc2026": wc2026,
    "fifa_ranking_2026": fifa_ranking_2026,
    "world_cup": world_cup,
    "former_names": former_names,
    "mapa_nombres": mapa_nombres,
    "nombres_espanol": NOMBRES_ESPANOL,
}

guardados = 0
for nombre, datos in datasets.items():
    if datos is not None:
        ruta_out = os.path.join(PROCESADOS, f"{nombre}.pkl")
        with open(ruta_out, "wb") as f:
            pickle.dump(datos, f)
        print(f"  ✓ {nombre}.pkl guardado")
        guardados += 1
    else:
        print(f"  ⚠ {nombre}: no disponible, se omite")

print(f"\n{'═'*60}")
print(f"  CARGA COMPLETADA: {guardados}/{len(datasets)} datasets guardados")
print(f"  Directorio de salida: {PROCESADOS}")
print("═"*60 + "\n")
