"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 0: Actualizar Datos
==============================================================
Fuente: datos manuales finales de grupos + ESPN API como respaldo

Actualiza automáticamente:
  1. Resultados finales de fase de grupos cargados manualmente
  2. Cruces de Dieciseisavos de Final con equipos clasificados
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
    """Traduce nombre de equipo del inglés al español."""
    return NOMBRES_ES.get(str(n).strip(), str(n).strip()) if n else n


def marca_equipo(equipo):
    """Prefijo visual sin emojis, útil para alinear prints."""
    return ""


# ══════════════════════════════════════════════════════════════
#  GRUPOS FINALES DEL MUNDIAL 2026 SEGUN LOS RESULTADOS MANUALES
#  12 grupos de 4 equipos — fase de grupos finalizada
# ══════════════════════════════════════════════════════════════
GRUPOS_2026 = {
    "A": ["México", "Qatar", "Suiza", "Camerún"],
    "B": ["Inglaterra", "Panamá", "Corea del Sur", "Serbia"],
    "C": ["Brasil", "Arabia Saudita", "Estados Unidos", "Noruega"],
    "D": ["Argentina", "Argelia", "Austria", "Jordania"],
    "E": ["Alemania", "Nueva Zelanda", "Canadá", "Islandia"],
    "F": ["Marruecos", "Irak", "Sudáfrica", "Chile"],
    "G": ["Portugal", "Indonesia", "Colombia", "Costa de Marfil"],
    "H": ["España", "Bolivia", "Japón", "Ucrania"],
    "I": ["Francia", "Fiyi", "Suecia", "Australia"],
    "J": ["Paraguay", "Vietnam", "Ecuador", "Turquía"],
    "K": ["Países Bajos", "Uzbekistán", "República Checa", "RD Congo"],
    "L": ["Croacia", "Honduras", "Ghana", "Senegal"],
}

# ══════════════════════════════════════════════════════════════
#  RESULTADOS FINALES DE LA FASE DE GRUPOS
#  Estos datos manuales son la fuente principal porque ESPN puede
#  ir desfasado durante la transición a Dieciseisavos de Final.
#  Formato: (local, visitante, goles_local, goles_visitante, grupo)
# ══════════════════════════════════════════════════════════════
PARTIDOS_GRUPOS_FINALIZADOS = [
    ("México", "Qatar", 3, 0, "A"), ("Suiza", "Camerún", 2, 1, "A"),
    ("México", "Camerún", 2, 0, "A"), ("Suiza", "Qatar", 3, 0, "A"),
    ("México", "Suiza", 3, 0, "A"), ("Camerún", "Qatar", 2, 0, "A"),
    ("Inglaterra", "Panamá", 4, 1, "B"), ("Corea del Sur", "Serbia", 2, 1, "B"),
    ("Inglaterra", "Serbia", 1, 1, "B"), ("Corea del Sur", "Panamá", 3, 0, "B"),
    ("Inglaterra", "Corea del Sur", 2, 1, "B"), ("Serbia", "Panamá", 3, 0, "B"),
    ("Brasil", "Arabia Saudita", 2, 0, "C"), ("Estados Unidos", "Noruega", 2, 1, "C"),
    ("Brasil", "Noruega", 1, 1, "C"), ("Estados Unidos", "Arabia Saudita", 3, 0, "C"),
    ("Brasil", "Estados Unidos", 2, 1, "C"), ("Noruega", "Arabia Saudita", 3, 0, "C"),
    ("Argentina", "Argelia", 3, 1, "D"), ("Austria", "Jordania", 2, 1, "D"),
    ("Argentina", "Austria", 2, 0, "D"), ("Argelia", "Jordania", 1, 1, "D"),
    ("Argentina", "Jordania", 3, 1, "D"), ("Austria", "Argelia", 3, 3, "D"),
    ("Alemania", "Nueva Zelanda", 3, 0, "E"), ("Canadá", "Islandia", 2, 1, "E"),
    ("Alemania", "Islandia", 1, 1, "E"), ("Canadá", "Nueva Zelanda", 2, 0, "E"),
    ("Alemania", "Canadá", 2, 1, "E"), ("Islandia", "Nueva Zelanda", 3, 0, "E"),
    ("Marruecos", "Irak", 3, 1, "F"), ("Sudáfrica", "Chile", 2, 1, "F"),
    ("Marruecos", "Chile", 1, 1, "F"), ("Sudáfrica", "Irak", 3, 0, "F"),
    ("Marruecos", "Sudáfrica", 2, 1, "F"), ("Chile", "Irak", 3, 0, "F"),
    ("Portugal", "Indonesia", 3, 0, "G"), ("Colombia", "Costa de Marfil", 2, 2, "G"),
    ("Portugal", "Costa de Marfil", 1, 1, "G"), ("Colombia", "Indonesia", 3, 0, "G"),
    ("Portugal", "Colombia", 0, 0, "G"), ("Costa de Marfil", "Indonesia", 3, 0, "G"),
    ("España", "Bolivia", 3, 0, "H"), ("Japón", "Ucrania", 2, 1, "H"),
    ("España", "Ucrania", 1, 1, "H"), ("Japón", "Bolivia", 2, 0, "H"),
    ("España", "Japón", 2, 1, "H"), ("Ucrania", "Bolivia", 3, 0, "H"),
    ("Francia", "Fiyi", 4, 0, "I"), ("Suecia", "Australia", 2, 1, "I"),
    ("Francia", "Australia", 3, 1, "I"), ("Suecia", "Fiyi", 3, 0, "I"),
    ("Francia", "Suecia", 2, 0, "I"), ("Australia", "Fiyi", 3, 0, "I"),
    ("Paraguay", "Vietnam", 2, 1, "J"), ("Ecuador", "Turquía", 2, 1, "J"),
    ("Paraguay", "Turquía", 1, 1, "J"), ("Ecuador", "Vietnam", 2, 0, "J"),
    ("Paraguay", "Ecuador", 2, 1, "J"), ("Turquía", "Vietnam", 3, 0, "J"),
    ("Países Bajos", "Uzbekistán", 3, 0, "K"), ("República Checa", "RD Congo", 2, 1, "K"),
    ("Países Bajos", "RD Congo", 1, 1, "K"), ("República Checa", "Uzbekistán", 2, 0, "K"),
    ("Países Bajos", "República Checa", 2, 1, "K"), ("RD Congo", "Uzbekistán", 3, 1, "K"),
    ("Croacia", "Honduras", 3, 0, "L"), ("Ghana", "Senegal", 2, 1, "L"),
    ("Croacia", "Senegal", 1, 1, "L"), ("Ghana", "Honduras", 2, 0, "L"),
    ("Croacia", "Ghana", 2, 1, "L"), ("Senegal", "Honduras", 3, 0, "L"),
]

# ══════════════════════════════════════════════════════════════
#  CRUCES DE DIECISEISAVOS DE FINAL — EDITAR AQUÍ SI CAMBIA LA LLAVE
#  Esta es la ÚNICA sección que el usuario debe modificar.
#  Cada tupla: (local, visitante, fecha)
# ══════════════════════════════════════════════════════════════
ROUND_OF_32 = [
    # CRUCES 100% CONFIRMADOS — FIFA / ESPN / 28 jun 2026
    ("Sudáfrica",      "Canadá",               "28/06/2026"),
    ("Brasil",         "Japón",                "29/06/2026"),
    ("Países Bajos",   "Marruecos",            "29/06/2026"),
    ("Alemania",       "Paraguay",             "30/06/2026"),
    ("Costa de Marfil","Noruega",              "30/06/2026"),
    ("Francia",        "Suecia",               "30/06/2026"),
    ("México",         "Ecuador",              "01/07/2026"),
    ("Inglaterra",     "RD Congo",             "01/07/2026"),
    ("Bélgica",        "Senegal",              "01/07/2026"),
    ("Estados Unidos", "Bosnia y Herzegovina",  "01/07/2026"),
    ("España",         "Austria",              "02/07/2026"),
    ("Portugal",       "Croacia",              "02/07/2026"),
    ("Suiza",          "Argelia",              "02/07/2026"),
    ("Australia",      "Egipto",               "03/07/2026"),
    ("Argentina",      "Cabo Verde",           "03/07/2026"),
    ("Colombia",       "Ghana",                "03/07/2026"),
]


# ══════════════════════════════════════════════════════════════
#  INICIO DEL SCRIPT
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PREDICCIÓN MUNDIALISTA 2026 - ACTUALIZACIÓN DE DATOS")
print("═"*60)
print("  Grupos: datos manuales finales  |  Dieciseisavos confirmados\n")

# ─── PASO 1: Descargar resultados de grupos ───────────────────
print("─"*60)
print("[PASO 1] Descargando resultados de grupos desde ESPN...")
print("─"*60)

RANGOS_FECHAS = [
    ("20260611", "20260620"),   # Jornadas 1-2
    ("20260621", "20260627"),   # Jornada 3
]

todos_eventos = []
for fecha_ini, fecha_fin in RANGOS_FECHAS:
    print(f"  → {fecha_ini[6:]}/{fecha_ini[4:6]} – {fecha_fin[6:]}/{fecha_fin[4:6]}...", end=" ", flush=True)
    try:
        r = requests.get(
            ESPN_SCORES, headers=HEADERS,
            params={"limit": 100, "dates": f"{fecha_ini}-{fecha_fin}"},
            timeout=15
        )
        r.raise_for_status()
        eventos = r.json().get("events", [])
        todos_eventos.extend(eventos)
        print(f"{len(eventos)} partidos")
    except requests.exceptions.Timeout:
        print("TIMEOUT — reintentando más tarde")
    except requests.exceptions.ConnectionError:
        print("SIN CONEXIÓN — verifique su internet")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(0.4)

print(f"\n  → Total eventos descargados: {len(todos_eventos)}")

# ─── Procesar partidos de grupos completados ──────────────────
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

    # Solo partidos completados
    completado = comp.get("status", {}).get("type", {}).get("completed", False)
    if not completado:
        continue

    home = next((c for c in competidores if c.get("homeAway") == "home"), competidores[0])
    away = next((c for c in competidores if c.get("homeAway") == "away"), competidores[1])

    nombre_home = traducir(home.get("team", {}).get("displayName", ""))
    nombre_away = traducir(away.get("team", {}).get("displayName", ""))

    # Ignorar placeholders "Winner of..."
    if "Winner" in nombre_home or "Winner" in nombre_away:
        continue

    score_home = int(float(home.get("score", 0)))
    score_away = int(float(away.get("score", 0)))
    fecha = evento.get("date", "")[:10]

    # Detectar grupo
    grupo_letra = ""
    for nota in comp.get("notes", []):
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

print(f"  → {len(partidos_grupos)} partidos de grupos procesados (completed=true, sin placeholders)")

if grupos_eq:
    print("\n  Grupos detectados vía ESPN:")
    for g in sorted(grupos_eq.keys()):
        print(f"    Grupo {g}: {', '.join(sorted(grupos_eq[g]))}")

# ─── PASO 1.5: Cargar fase de grupos finalizada ───────────────
print("\n" + "─"*60)
print("[PASO 1.5] Cargando resultados finales de fase de grupos...")
print("─"*60)
print("  Los datos manuales prevalecen sobre ESPN porque la fase")
print("  de grupos ya terminó y ESPN puede quedar desfasado.\n")

partidos_grupos = []
for home, away, hg, ag, grupo in PARTIDOS_GRUPOS_FINALIZADOS:
    partidos_grupos.append({
        "Round": "Group stage", "Date": "2026-06-27",
        "home_team": home, "away_team": away,
        "Home": home, "Away": away,
        "HGFT": hg, "AGFT": ag,
        "Finished": 1, "Group": grupo, "Year": 2026,
    })
    print(f"    Grupo {grupo}: {home} {hg}-{ag} {away}")

print(f"\n  → {len(partidos_grupos)} partidos de grupos cargados manualmente")
print(f"  → {len(GRUPOS_2026)} grupos completos (A-L)")

# ─── PASO 2: Guardar WorldCup2026_grupos.csv ──────────────────
print("\n" + "─"*60)
print("[PASO 2] Guardando WorldCup2026_grupos.csv...")
print("─"*60)

df_g = pd.DataFrame(partidos_grupos).drop_duplicates(
    subset=["home_team", "away_team"]
).reset_index(drop=True)

ruta_g = os.path.join(WC_DATA, "WorldCup2026_grupos.csv")
os.makedirs(WC_DATA, exist_ok=True)
df_g.to_csv(ruta_g, index=False, encoding="utf-8-sig")
print(f"  ✓ {len(df_g)} partidos guardados en WorldCup2026_grupos.csv")

# ─── PASO 3: Construir schedule con Dieciseisavos ─────────────
print("\n" + "─"*60)
print("[PASO 3] Actualizando schedule_2026.csv con Dieciseisavos...")
print("─"*60)

# NOTA: en español, la ronda de 32 equipos (16 partidos) se llama
# "Dieciseisavos de Final". La siguiente ronda (16 equipos, 8 partidos)
# es "Octavos de Final". NUNCA se salta de una a otra.
filas_ko = []
for local, visitante, fecha in ROUND_OF_32:
    filas_ko.append({
        "Round": "Dieciseisavos de Final",
        "Date": fecha,
        "home_team": local,
        "away_team": visitante,
        "HGFT": None,
        "AGFT": None,
        "Finished": 0,
        "Year": 2026,
    })

df_ko = pd.DataFrame(filas_ko)

# Conservar grupos originales del schedule si existe
ruta_sched = os.path.join(FUCHIBOL, "schedule_2026.csv")
df_grupos_orig = pd.DataFrame()

if os.path.exists(ruta_sched):
    try:
        tmp = pd.read_csv(ruta_sched, encoding="utf-8-sig")
        mask = tmp["Round"].str.contains("Group|group|Grupo|grupo", case=False, na=False)
        df_grupos_orig = tmp[mask].copy()
        print(f"  → {len(df_grupos_orig)} filas de grupos conservadas del schedule original")
    except Exception as e:
        print(f"  ⚠ Error leyendo schedule original: {e}")

if not df_grupos_orig.empty:
    df_final = pd.concat([df_grupos_orig, df_ko], ignore_index=True)
else:
    df_final = df_ko

os.makedirs(FUCHIBOL, exist_ok=True)
df_final.to_csv(ruta_sched, index=False, encoding="utf-8-sig")

print(f"  ✓ schedule_2026.csv actualizado ({len(df_final)} filas totales)")
print(f"  → {len(df_ko)} cruces de Dieciseisavos confirmados:")
for _, r in df_ko.iterrows():
    print(f"    {r['home_team']:<25} vs {r['away_team']:<25}  ({r['Date']})")

# ─── PASO 4: Actualizar pickle ────────────────────────────────
print("\n" + "─"*60)
print("[PASO 4] Actualizando pickle en procesados/...")
print("─"*60)

os.makedirs(PROCESADOS, exist_ok=True)
try:
    df_nuevo = pd.read_csv(ruta_sched, encoding="utf-8-sig")
    ruta_pkl = os.path.join(PROCESADOS, "schedule_2026.pkl")
    with open(ruta_pkl, "wb") as f:
        pickle.dump(df_nuevo, f)
    print(f"  ✓ schedule_2026.pkl actualizado ({len(df_nuevo)} filas)")
except Exception as e:
    print(f"  ⚠ Error guardando pickle: {e}")
    print("    → Ejecute primero 1_cargar_datos.py para crear la carpeta procesados/")

# ─── Resumen final ────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  ACTUALIZACIÓN COMPLETADA")
print(f"  → {len(df_g)} partidos de grupos guardados (12 grupos completos)")
print(f"  → {len(ROUND_OF_32)} cruces de Dieciseisavos de Final cargados")
print(f"\n  Próximos pasos:")
print(f"    python 3_entrenar_modelo.py")
print(f"    python 4_simular_llave.py")
print(f"    python 5_visualizar_llave.py")
print("═"*60 + "\n")



