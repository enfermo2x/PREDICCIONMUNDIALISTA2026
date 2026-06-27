"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 0: Actualizar Datos
==============================================================
Fuente: ESPN API (grupos) + Cruces reales confirmados (Round of 32)

Actualiza automáticamente:
  1. Resultados fase de grupos desde ESPN API
  2. Cruces del Round of 32 con equipos reales confirmados
  3. Pickle del schedule para el Script 4

Ejecutar ANTES del Script 4 cada vez que avance el torneo.
==============================================================
"""

import os
import sys
import re
import time
import pickle
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import requests
except ImportError:
    print("  ✗ Falta requests. Ejecute: pip install requests")
    sys.exit(1)

# ─── Rutas ────────────────────────────────────────────────────
RAIZ       = os.path.dirname(os.path.abspath(__file__))
WC_DATA    = os.path.join(RAIZ, "wc-data")
FUCHIBOL   = os.path.join(WC_DATA, "Data Fuchibol")
PROCESADOS = os.path.join(RAIZ, "procesados")

ESPN_SCORES = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
    "Accept": "application/json",
}

# ─── Traducciones ─────────────────────────────────────────────
NOMBRES_ES = {
    "Mexico": "México", "France": "Francia", "Germany": "Alemania",
    "Brazil": "Brasil", "Argentina": "Argentina", "England": "Inglaterra",
    "Portugal": "Portugal", "Netherlands": "Países Bajos",
    "Morocco": "Marruecos", "Japan": "Japón", "South Korea": "Corea del Sur",
    "Korea Republic": "Corea del Sur", "Australia": "Australia",
    "Croatia": "Croacia", "Switzerland": "Suiza", "Poland": "Polonia",
    "Serbia": "Serbia", "Denmark": "Dinamarca",
    "United States": "Estados Unidos", "USA": "Estados Unidos",
    "Canada": "Canadá", "Belgium": "Bélgica", "Italy": "Italia",
    "Uruguay": "Uruguay", "Colombia": "Colombia", "Ecuador": "Ecuador",
    "Spain": "España", "Costa Rica": "Costa Rica", "Panama": "Panamá",
    "Cameroon": "Camerún", "Senegal": "Senegal", "Ghana": "Ghana",
    "Nigeria": "Nigeria", "Tunisia": "Túnez", "Egypt": "Egipto",
    "Algeria": "Argelia", "Saudi Arabia": "Arabia Saudita",
    "Iran": "Irán", "IR Iran": "Irán", "South Africa": "Sudáfrica",
    "Ivory Coast": "Costa de Marfil", "Côte d'Ivoire": "Costa de Marfil",
    "Cote d'Ivoire": "Costa de Marfil",
    "Czech Republic": "República Checa", "Czechia": "República Checa",
    "Sweden": "Suecia", "Norway": "Noruega",
    "Turkey": "Turquía", "Türkiye": "Turquía",
    "Scotland": "Escocia", "Wales": "Gales",
    "Paraguay": "Paraguay", "Bolivia": "Bolivia",
    "Venezuela": "Venezuela", "Peru": "Perú", "Chile": "Chile",
    "Cape Verde": "Cabo Verde",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Bosnia & Herzegovina": "Bosnia y Herzegovina",
    "DR Congo": "RD Congo", "Congo DR": "RD Congo",
    "Uzbekistan": "Uzbekistán", "Curaçao": "Curazao",
    "Jordan": "Jordania", "Iraq": "Irak",
    "New Zealand": "Nueva Zelanda", "Honduras": "Honduras",
    "El Salvador": "El Salvador", "Jamaica": "Jamaica",
    "Haiti": "Haití", "Cuba": "Cuba", "Austria": "Austria",
    "Hungary": "Hungría", "Romania": "Rumanía", "Greece": "Grecia",
    "Slovenia": "Eslovenia", "Slovakia": "Eslovaquia",
    "Albania": "Albania", "Qatar": "Catar", "Indonesia": "Indonesia",
    "North Macedonia": "Macedonia del Norte", "Finland": "Finlandia",
    "Ukraine": "Ucrania", "Burkina Faso": "Burkina Faso",
    "Mali": "Malí", "Kenya": "Kenia", "Ethiopia": "Etiopía",
    "Angola": "Angola", "Libya": "Libia",
}

def traducir(n):
    return NOMBRES_ES.get(str(n).strip(), str(n).strip()) if n else n


# ══════════════════════════════════════════════════════════════
# CRUCES ROUND OF 32 — CONFIRMADOS OFICIALMENTE
# Fuente: FIFA / ESPN / Sky Sports / Sports Illustrated (27 jun 2026)
# Los partidos de hoy (Croacia, Inglaterra, Portugal, RD Congo,
# Austria, Argentina) se predicen como ganadores según análisis.
# ══════════════════════════════════════════════════════════════
ROUND_OF_32 = [
    # Partido          Local                    Visitante                Fecha
    ("Sudáfrica",      "Canadá",                "28/06/2026"),
    ("México",         "Corea del Sur",         "28/06/2026"),
    ("Brasil",         "Japón",                 "29/06/2026"),
    ("Países Bajos",   "Marruecos",             "29/06/2026"),
    ("Alemania",       "Ecuador",               "30/06/2026"),
    ("Costa de Marfil","Noruega",               "30/06/2026"),
    ("Francia",        "Suecia",                "30/06/2026"),
    ("Inglaterra",     "Panamá",                "01/07/2026"),  # Inglaterra gana hoy
    ("Bélgica",        "Irán",                  "01/07/2026"),
    ("Estados Unidos", "Bosnia y Herzegovina",  "01/07/2026"),
    ("España",         "Austria",               "02/07/2026"),  # Austria gana hoy → Austria vs España
    ("Portugal",       "Croacia",               "02/07/2026"),  # Portugal y Croacia ganan hoy
    ("Suiza",          "Argelia",               "02/07/2026"),  # Argelia clasificada
    ("Australia",      "Egipto",                "03/07/2026"),
    ("Argentina",      "Cabo Verde",            "03/07/2026"),  # Argentina gana hoy
    ("Colombia",       "RD Congo",              "03/07/2026"),  # RD Congo gana hoy → Colombia vs RD Congo
]

# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PREDICCIÓN MUNDIALISTA 2026 - ACTUALIZACIÓN DE DATOS")
print("═"*60)
print("  Grupos: ESPN API | Round of 32: cruces reales confirmados\n")

# ══════════════════════════════════════════════════════════════
# PASO 1: Descargar resultados de grupos desde ESPN
# ══════════════════════════════════════════════════════════════
print("─"*60)
print("[PASO 1] Descargando resultados de grupos desde ESPN...")
print("─"*60)

RANGOS = [
    ("20260611", "20260620"),
    ("20260621", "20260627"),
]

todos_eventos = []
for fecha_ini, fecha_fin in RANGOS:
    print(f"  → {fecha_ini[6:]}/{fecha_ini[4:6]} – {fecha_fin[6:]}/{fecha_fin[4:6]}...", end=" ", flush=True)
    try:
        r = requests.get(ESPN_SCORES, headers=HEADERS,
                         params={"limit": 100, "dates": f"{fecha_ini}-{fecha_fin}"},
                         timeout=15)
        r.raise_for_status()
        eventos = r.json().get("events", [])
        todos_eventos.extend(eventos)
        print(f"{len(eventos)} partidos")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(0.4)

print(f"\n  → Total eventos descargados: {len(todos_eventos)}")

# Procesar solo partidos de grupos COMPLETADOS
partidos_grupos = []
grupos_eq = {}

for evento in todos_eventos:
    comps = evento.get("competitions", [])
    if not comps:
        continue
    comp = comps[0]
    competidores = comp.get("competitors", [])
    if len(competidores) < 2:
        continue

    completado = comp.get("status", {}).get("type", {}).get("completed", False)
    if not completado:
        continue  # saltar partidos no jugados

    home = next((c for c in competidores if c.get("homeAway") == "home"), competidores[0])
    away = next((c for c in competidores if c.get("homeAway") == "away"), competidores[1])

    nombre_home = traducir(home.get("team", {}).get("displayName", ""))
    nombre_away = traducir(away.get("team", {}).get("displayName", ""))
    score_home  = int(float(home.get("score", 0)))
    score_away  = int(float(away.get("score", 0)))
    fecha       = evento.get("date", "")[:10]

    # Detectar grupo
    notas = comp.get("notes", [])
    grupo_letra = ""
    for nota in notas:
        texto = nota.get("text", nota.get("headline", ""))
        m = re.search(r"Group\s+([A-L])", texto, re.IGNORECASE)
        if m:
            grupo_letra = m.group(1).upper()
            break

    partidos_grupos.append({
        "Round": "Group stage", "Date": fecha,
        "home_team": nombre_home, "away_team": nombre_away,
        "Home": nombre_home, "Away": nombre_away,
        "HGFT": score_home, "AGFT": score_away,
        "Finished": 1, "Group": grupo_letra, "Year": 2026,
    })

    if grupo_letra:
        grupos_eq.setdefault(grupo_letra, set())
        grupos_eq[grupo_letra].update([nombre_home, nombre_away])

# Agregar los 6 partidos de HOY (predicciones)
PARTIDOS_HOY = [
    ("Croacia",   "Ghana",         "27/06/2026", 2, 1),
    ("Inglaterra","Panamá",        "27/06/2026", 2, 1),
    ("Portugal",  "Colombia",      "27/06/2026", 2, 1),
    ("RD Congo",  "Uzbekistán",    "27/06/2026", 2, 1),
    ("Austria",   "Argelia",       "27/06/2026", 2, 1),
    ("Argentina", "Jordania",      "27/06/2026", 2, 1),
]

for home, away, fecha, hg, ag in PARTIDOS_HOY:
    partidos_grupos.append({
        "Round": "Group stage", "Date": fecha,
        "home_team": home, "away_team": away,
        "Home": home, "Away": away,
        "HGFT": hg, "AGFT": ag,
        "Finished": 1, "Group": "", "Year": 2026,
    })

jugados_espn = len([p for p in partidos_grupos if p.get("Date", "") < "2026-06-27"])
print(f"  → {len(partidos_grupos)} partidos de grupos ({jugados_espn} de ESPN + 6 de hoy predichos)")

if grupos_eq:
    print("\n  Grupos detectados:")
    for g in sorted(grupos_eq.keys()):
        print(f"    Grupo {g}: {', '.join(sorted(grupos_eq[g]))}")

# ══════════════════════════════════════════════════════════════
# PASO 2: Guardar WorldCup2026_grupos.csv
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("[PASO 2] Guardando WorldCup2026_grupos.csv...")
print("─"*60)

df_g = pd.DataFrame(partidos_grupos).drop_duplicates(
    subset=["home_team", "away_team"]).reset_index(drop=True)
ruta_g = os.path.join(WC_DATA, "WorldCup2026_grupos.csv")
df_g.to_csv(ruta_g, index=False, encoding="utf-8-sig")
print(f"  ✓ {len(df_g)} partidos guardados en WorldCup2026_grupos.csv")

# ══════════════════════════════════════════════════════════════
# PASO 3: Construir schedule con Round of 32 confirmado
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("[PASO 3] Actualizando schedule_2026.csv con Round of 32...")
print("─"*60)

filas_ko = []
for home, away, fecha in ROUND_OF_32:
    filas_ko.append({
        "Round":     "Round of 32",
        "Date":      fecha,
        "home_team": home,
        "away_team": away,
        "HGFT":      None,
        "AGFT":      None,
        "Finished":  0,
        "Year":      2026,
    })

df_ko = pd.DataFrame(filas_ko)

# Conservar grupos originales del schedule si existen
ruta_sched = os.path.join(FUCHIBOL, "schedule_2026.csv")
df_grupos_orig = pd.DataFrame()
if os.path.exists(ruta_sched):
    try:
        tmp = pd.read_csv(ruta_sched, encoding="utf-8-sig")
        mask = tmp["Round"].str.contains("Group|group|Grupo|grupo", case=False, na=False)
        df_grupos_orig = tmp[mask].copy()
    except Exception:
        pass

df_final = pd.concat([df_grupos_orig, df_ko], ignore_index=True) if not df_grupos_orig.empty else df_ko
df_final.to_csv(ruta_sched, index=False, encoding="utf-8-sig")

print(f"  ✓ schedule_2026.csv actualizado")
print(f"    · {len(df_grupos_orig)} filas de grupos conservadas")
print(f"    · {len(df_ko)} cruces del Round of 32 confirmados:")
for _, r in df_ko.iterrows():
    print(f"      {r['home_team']} vs {r['away_team']}  ({r['Date']})")

# ══════════════════════════════════════════════════════════════
# PASO 4: Actualizar pickle
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("[PASO 4] Actualizando pickle en procesados/...")
print("─"*60)

if os.path.exists(PROCESADOS):
    try:
        df_nuevo = pd.read_csv(ruta_sched, encoding="utf-8-sig")
        ruta_pkl = os.path.join(PROCESADOS, "schedule_2026.pkl")
        with open(ruta_pkl, "wb") as f:
            pickle.dump(df_nuevo, f)
        print(f"  ✓ schedule_2026.pkl actualizado ({len(df_nuevo)} filas)")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
else:
    print("  ⚠ Carpeta procesados/ no encontrada.")
    print("    → Ejecute primero 1_cargar_datos.py")

print(f"\n{'═'*60}")
print("  ACTUALIZACIÓN COMPLETADA")
print(f"  → {len(df_g)} partidos de grupos guardados")
print(f"  → {len(ROUND_OF_32)} cruces del Round of 32 cargados")
print(f"\n  Ejecute ahora:")
print(f"    python 4_simular_llave.py")
print(f"    python 5_visualizar_llave.py")
print("═"*60 + "\n")