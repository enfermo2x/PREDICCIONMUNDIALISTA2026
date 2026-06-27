"""
==============================================================
PREDICCIÓN MUNDIALISTA 2026 - Script 5: Visualizar Llave
==============================================================
Descripción: Genera el bracket visual con matplotlib y exporta
los resultados a un Excel con formato profesional.
==============================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

try:
    import openpyxl
    from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                                  GradientFill)
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import BarChart, Reference
    from openpyxl.chart.label import DataLabelList
    OPENPYXL_OK = True
except ImportError:
    print("  ⚠ openpyxl no disponible. Instale con: pip install openpyxl")
    OPENPYXL_OK = False

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
print("  PREDICCIÓN MUNDIALISTA 2026 - VISUALIZACIÓN")
print("═"*60)

# ─── Cargar predicciones ──────────────────────────────────────
print("\n[1/4] Cargando predicciones guardadas...")
datos = cargar_pickle("predicciones_llave")

if datos is None:
    print("  ✗ Error: ejecute primero 4_simular_llave.py")
    sys.exit(1)

df = datos["resultados"]
campeon = datos.get("campeon", "Por determinar")
print(f"  → {len(df)} partidos cargados")
print(f"  → Campeón predicho: {campeon}")

# ──────────────────────────────────────────────────────────────
# Utilidades de probabilidad y paneles
# ──────────────────────────────────────────────────────────────

def construir_probabilidades_equipo(df):
    equipos = sorted(set(df["Equipo Local"]).union(df["Equipo Visitante"]))
    stats = {equipo: {"Dieciseisavos": 1.0, "Octavos": 0.0, "Cuartos": 0.0, "Semifinales": 0.0, "Final": 0.0, "Campeon": 0.0}
             for equipo in equipos}
    for _, row in df.iterrows():
        ronda = row["Ronda"]
        local = row["Equipo Local"]
        visit = row["Equipo Visitante"]
        prob_local = row["Probabilidad Local"]
        prob_visit = row["Probabilidad Visitante"]

        if ronda == "Dieciseisavos de Final":
            stats[local]["Octavos"] = prob_local
            stats[visit]["Octavos"] = prob_visit
        elif ronda == "Octavos de Final":
            stats[local]["Cuartos"] = stats[local]["Octavos"] * prob_local
            stats[visit]["Cuartos"] = stats[visit]["Octavos"] * prob_visit
        elif ronda == "Cuartos de Final":
            stats[local]["Semifinales"] = stats[local]["Cuartos"] * prob_local
            stats[visit]["Semifinales"] = stats[visit]["Cuartos"] * prob_visit
        elif ronda == "Semifinales":
            stats[local]["Final"] = stats[local]["Semifinales"] * prob_local
            stats[visit]["Final"] = stats[visit]["Semifinales"] * prob_visit
        elif ronda == "Final":
            stats[local]["Campeon"] = stats[local]["Final"] * prob_local
            stats[visit]["Campeon"] = stats[visit]["Final"] * prob_visit

    return pd.DataFrame([{"Equipo": equipo, **valores} for equipo, valores in stats.items()])


def formatear_porcentaje(valor):
    return f"{valor:.0%}"


def ordenar_top5(df_probs):
    return df_probs.sort_values("Campeon", ascending=False).head(5).reset_index(drop=True)


def colorear_celda(cell, color):
    cell.fill = PatternFill("solid", fgColor=color)

# ─── EXPORTAR EXCEL ──────────────────────────────────────────
print("\n[2/4] Exportando resultados a Excel con formato profesional...")

if OPENPYXL_OK:
    wb = openpyxl.Workbook()

    # ── Hoja 1: Tabla de predicciones ──────────────────────────
    ws = wb.active
    ws.title = "Predicciones Llave 2026"

    # Colores
    VERDE_OSCURO  = "1B5E20"   # encabezado
    VERDE_TEXTO   = "FFFFFF"   # texto encabezado
    VERDE_CLARO1  = "E8F5E9"   # fila par
    VERDE_CLARO2  = "FFFFFF"   # fila impar
    DORADO        = "FFD700"   # campeón
    DORADO_TEXTO  = "1B5E20"
    GRIS_BORDE    = "BDBDBD"

    encabezados = [
        "Ronda", "Equipo Local", "Equipo Visitante",
        "Goles Local", "Goles Visitante", "Marcador",
        "Ganador", "Prob. Victoria", "Prob. Local", "Prob. Visitante"
    ]

    # Añadir encabezados
    ws.append(encabezados)
    for col_idx, _ in enumerate(encabezados, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = PatternFill("solid", fgColor=VERDE_OSCURO)
        cell.font = Font(bold=True, color=VERDE_TEXTO, size=11, name="Calibri")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(
            bottom=Side(style="medium", color=GRIS_BORDE),
            right=Side(style="thin", color=GRIS_BORDE)
        )

    ws.row_dimensions[1].height = 30

    # Orden de rondas
    orden_rondas = ["Dieciseisavos de Final", "Octavos de Final", "Cuartos de Final",
                    "Semifinales", "Tercer y Cuarto Puesto", "Final"]

    df["_orden"] = df["Ronda"].apply(
        lambda r: orden_rondas.index(r) if r in orden_rondas else 99
    )
    df_sorted = df.sort_values("_orden").reset_index(drop=True)

    for fila_idx, (_, row) in enumerate(df_sorted.iterrows(), 2):
        es_campeon = (row["Ronda"] == "Final" and row["Ganador"] == campeon)
        es_par = fila_idx % 2 == 0

        marcador = f"{int(row['Goles Local'])} - {int(row['Goles Visitante'])}"
        datos_fila = [
            row["Ronda"],
            row["Equipo Local"],
            row["Equipo Visitante"],
            int(row["Goles Local"]),
            int(row["Goles Visitante"]),
            marcador,
            row["Ganador"],
            f"{row['Probabilidad Victoria']:.0%}",
            f"{row['Probabilidad Local']:.0%}",
            f"{row['Probabilidad Visitante']:.0%}",
        ]
        ws.append(datos_fila)

        for col_idx in range(1, len(encabezados) + 1):
            cell = ws.cell(row=fila_idx, column=col_idx)
            if es_campeon:
                cell.fill = PatternFill("solid", fgColor=DORADO)
                cell.font = Font(bold=True, color=DORADO_TEXTO, size=11, name="Calibri")
            else:
                color_fondo = VERDE_CLARO1 if es_par else VERDE_CLARO2
                cell.fill = PatternFill("solid", fgColor=color_fondo)
                cell.font = Font(size=11, name="Calibri")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = Border(
                bottom=Side(style="thin", color=GRIS_BORDE),
                right=Side(style="thin", color=GRIS_BORDE)
            )

        ws.row_dimensions[fila_idx].height = 22

    # Anchos de columna
    anchos = [22, 26, 26, 13, 14, 12, 26, 14, 13, 15]
    for col_idx, ancho in enumerate(anchos, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = ancho

    ws.freeze_panes = "A2"

    # ── Hoja 2: Resumen por ronda ──────────────────────────────
    ws2 = wb.create_sheet("Resumen por Ronda")
    ws2.append(["Ronda", "Partidos", "Goles Totales", "Promedio Goles"])
    ws2.cell(row=1, column=1).fill = PatternFill("solid", fgColor=VERDE_OSCURO)
    for c in range(1, 5):
        cell = ws2.cell(row=1, column=c)
        cell.fill = PatternFill("solid", fgColor=VERDE_OSCURO)
        cell.font = Font(bold=True, color=VERDE_TEXTO, size=11)
        cell.alignment = Alignment(horizontal="center")

    for i, ronda in enumerate(orden_rondas, 2):
        partidos_r = df_sorted[df_sorted["Ronda"] == ronda]
        if partidos_r.empty:
            continue
        total_goles = int(partidos_r["Goles Local"].sum() + partidos_r["Goles Visitante"].sum())
        prom = total_goles / len(partidos_r) if len(partidos_r) > 0 else 0
        ws2.append([ronda, len(partidos_r), total_goles, f"{prom:.2f}"])
        color = VERDE_CLARO1 if i % 2 == 0 else VERDE_CLARO2
        for c in range(1, 5):
            ws2.cell(row=i, column=c).fill = PatternFill("solid", fgColor=color)
            ws2.cell(row=i, column=c).alignment = Alignment(horizontal="center")

    for c, w in zip(range(1, 5), [28, 12, 14, 18]):
        ws2.column_dimensions[get_column_letter(c)].width = w

    # ── Hoja 3: Campeón destacado ─────────────────────────────
    ws3 = wb.create_sheet("Campeón Predicho")
    ws3.merge_cells("A1:H3")
    ws3["A1"] = "🏟️ LOGO MUNDIAL 2026 - PLACEHOLDER"
    ws3["A1"].font = Font(bold=True, size=16, color=VERDE_TEXTO, name="Calibri")
    ws3["A1"].fill = PatternFill("solid", fgColor=COLOR_FONDO if False else "1B5E20")
    ws3["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[1].height = 40

    ws3.merge_cells("A5:H5")
    ws3["A5"] = "🏆 PREDICCIÓN MUNDIALISTA 2026 🏆"
    ws3["A5"].font = Font(bold=True, size=24, color=DORADO_TEXTO, name="Calibri")
    ws3["A5"].fill = PatternFill("solid", fgColor=VERDE_OSCURO)
    ws3["A5"].alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[5].height = 35

    ws3.merge_cells("A7:H7")
    ws3["A7"] = f"CAMPEÓN: {campeon.upper()}"
    ws3["A7"].font = Font(bold=True, size=28, color=DORADO_TEXTO, name="Calibri")
    ws3["A7"].fill = PatternFill("solid", fgColor=DORADO)
    ws3["A7"].alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[7].height = 50

    ws3.merge_cells("A9:H9")
    ws3["A9"] = "Generado por el Sistema de Predicción Mundialista 2026"
    ws3["A9"].font = Font(italic=True, size=11, color="555555")
    ws3["A9"].alignment = Alignment(horizontal="center")

    for c in range(1, 9):
        ws3.column_dimensions[get_column_letter(c)].width = 18

    # ── Hoja 4: Dashboard ejecutivo ─────────────────────────────
    ws4 = wb.create_sheet("Dashboard")
    ws4.merge_cells("A1:H1")
    ws4["A1"] = "DASHBOARD EJECUTIVO - PREDICCIONES MUNDIAL 2026"
    ws4["A1"].font = Font(bold=True, size=20, color=VERDE_TEXTO, name="Calibri")
    ws4["A1"].fill = PatternFill("solid", fgColor=VERDE_OSCURO)
    ws4["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws4.row_dimensions[1].height = 35

    probabilidades = construir_probabilidades_equipo(df)
    top5_df = ordenar_top5(probabilidades)

    headers_dashboard = ["Ranking", "Equipo", "Prob. Campeón", "Prob. Final", "Prob. Semifinal", "Prob. Cuartos", "Prob. Octavos", "Prob. Dieciseisavos"]
    ws4.append(headers_dashboard)
    for col_idx, _ in enumerate(headers_dashboard, 1):
        cell = ws4.cell(row=2, column=col_idx)
        cell.fill = PatternFill("solid", fgColor=VERDE_OSCURO)
        cell.font = Font(bold=True, color=VERDE_TEXTO, size=11, name="Calibri")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=Side(style="medium", color=GRIS_BORDE), right=Side(style="thin", color=GRIS_BORDE))
    ws4.row_dimensions[2].height = 24

    for idx, row in top5_df.iterrows():
        ws4.append([
            idx + 1,
            row["Equipo"],
            row["Campeon"],
            row["Final"],
            row["Semifinales"],
            row["Cuartos"],
            row["Octavos"],
            row["Dieciseisavos"],
        ])

    for row_idx in range(3, 3 + len(top5_df)):
        fill_color = VERDE_CLARO1 if row_idx % 2 == 0 else VERDE_CLARO2
        for col_idx in range(1, len(headers_dashboard) + 1):
            cell = ws4.cell(row=row_idx, column=col_idx)
            cell.fill = PatternFill("solid", fgColor=fill_color)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if col_idx >= 3:
                cell.number_format = "0%"

    ws4.auto_filter.ref = f"A2:H{2 + len(top5_df)}"
    for col_idx, ancho in enumerate([10, 24, 14, 14, 14, 14, 14, 18], 1):
        ws4.column_dimensions[get_column_letter(col_idx)].width = ancho

    chart_campeon = BarChart()
    chart_campeon.type = "col"
    chart_campeon.style = 13
    chart_campeon.title = "Top 5 - Probabilidad de ser Campeón"
    chart_campeon.y_axis.majorGridlines = None
    data = Reference(ws4, min_col=3, min_row=2, max_row=2 + len(top5_df), max_col=3)
    categories = Reference(ws4, min_col=2, min_row=3, max_row=2 + len(top5_df))
    chart_campeon.add_data(data, titles_from_data=True)
    chart_campeon.set_categories(categories)
    chart_campeon.height = 10
    chart_campeon.width = 16
    ws4.add_chart(chart_campeon, "J3")

    chart_stack = BarChart()
    chart_stack.type = "col"
    chart_stack.grouping = "stacked"
    chart_stack.style = 12
    chart_stack.title = "Probabilidad por ronda - Top 5"
    data = Reference(ws4, min_col=3, min_row=2, max_col=8, max_row=2 + len(top5_df))
    chart_stack.add_data(data, titles_from_data=True)
    chart_stack.set_categories(categories)
    chart_stack.height = 12
    chart_stack.width = 20
    chart_stack.legend.position = "r"
    ws4.add_chart(chart_stack, "J20")

    ruta_excel = os.path.join(SALIDA, "predicciones_llave_2026.xlsx")
    wb.save(ruta_excel)
    print(f"  ✓ Excel guardado en: {ruta_excel}")
else:
    print("  ⚠ openpyxl no disponible. Exportando a CSV en su lugar.")
    ruta_csv = os.path.join(SALIDA, "predicciones_llave_2026.csv")
    df_sorted = df.copy()
    df_sorted.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"  ✓ CSV guardado en: {ruta_csv}")

# ─── GENERAR BRACKET VISUAL ──────────────────────────────────
print("\n[3/4] Generando bracket visual con matplotlib...")

# Paleta de colores del Mundial 2026
COLOR_FONDO    = "#081426"   # azul noche profundo
COLOR_PANEL    = "#12233F"   # panel oscuro azul
COLOR_VERDE    = "#00C75A"   # verde FIFA intenso
COLOR_DORADO   = "#FFD700"   # dorado
COLOR_TEXTO    = "#F5F7FA"   # texto claro
COLOR_GRIS     = "#8FA3B1"   # subtexto
COLOR_LINEA    = "#3E5E8D"   # líneas del bracket
COLOR_GANADOR  = "#FFD700"   # ganador destacado
COLOR_RONDA    = "#53D17A"   # encabezados de ronda
COLOR_FONDO_2  = "#0D1B33"   # degradado secundario

# Organizar partidos por ronda
orden_rondas = ["Dieciseisavos de Final", "Octavos de Final", "Cuartos de Final", "Semifinales", "Tercer y Cuarto Puesto", "Final"]
rondas_dict = {r: [] for r in orden_rondas}
for _, row in df.iterrows():
    if row["Ronda"] in rondas_dict:
        rondas_dict[row["Ronda"]].append(row)

# Panel de favoritos: top 5 campeones
probabilidades = construir_probabilidades_equipo(df)
top5_df = ordenar_top5(probabilidades)

fig = plt.figure(figsize=(36, 20), facecolor=COLOR_FONDO)
grid = GridSpec(1, 6, figure=fig, wspace=0.12, hspace=0.2)
ax_main = fig.add_subplot(grid[:, :5])
ax_side = fig.add_subplot(grid[:, 5:])
ax_main.set_xlim(0, 30)
ax_main.set_ylim(0, 18)
ax_main.axis("off")
ax_main.set_facecolor(COLOR_FONDO)
ax_side.set_facecolor(COLOR_FONDO_2)
ax_side.set_xlim(0, 1)
ax_side.set_ylim(0, 1)
ax_side.axis("off")

# Título principal
ax_main.text(15, 17.5, "🏆 PREDICCIÓN MUNDIALISTA 2026 🏆",
             ha="center", va="center", fontsize=34, fontweight="bold",
             color=COLOR_DORADO, family="DejaVu Sans")
ax_main.text(15, 16.5, f"Campeón predicho: {campeon}",
             ha="center", va="center", fontsize=20, weight="semibold",
             color=COLOR_VERDE)

# Posiciones X por ronda
X_RONDAS = {
    "Dieciseisavos de Final": 3.0,
    "Octavos de Final": 8.0,
    "Cuartos de Final": 13.0,
    "Semifinales": 18.0,
    "Final": 23.0,
    "Tercer y Cuarto Puesto": 18.0,
}
ANCHO_CAJA = 3.6
ALTO_CAJA  = 0.6

def dibujar_partido(ax, x, y, eq_local, eq_visit, gl, gv, ganador, prob):
    """Dibuja una caja de partido en el bracket."""
    # Caja fondo
    rect = FancyBboxPatch((x - ANCHO_CAJA/2, y - ALTO_CAJA),
                           ANCHO_CAJA, ALTO_CAJA * 2,
                           boxstyle="round,pad=0.05",
                           linewidth=1.5,
                           edgecolor=COLOR_VERDE,
                           facecolor=COLOR_PANEL,
                           zorder=2)
    ax.add_patch(rect)

    # Línea divisoria central
    ax.plot([x - ANCHO_CAJA/2 + 0.05, x + ANCHO_CAJA/2 - 0.05],
            [y, y], color=COLOR_LINEA, lw=0.8, zorder=3)

    # Equipo local
    color_l = COLOR_DORADO if eq_local == ganador else COLOR_TEXTO
    peso_l  = "bold" if eq_local == ganador else "normal"
    nombre_l = eq_local[:18] if len(eq_local) > 18 else eq_local
    ax.text(x - ANCHO_CAJA/2 + 0.15, y + ALTO_CAJA*0.45,
            nombre_l, ha="left", va="center",
            fontsize=7, color=color_l, fontweight=peso_l, zorder=4)
    ax.text(x + ANCHO_CAJA/2 - 0.15, y + ALTO_CAJA*0.45,
            str(gl), ha="right", va="center",
            fontsize=7, color=color_l, fontweight=peso_l, zorder=4)

    # Equipo visitante
    color_v = COLOR_DORADO if eq_visit == ganador else COLOR_TEXTO
    peso_v  = "bold" if eq_visit == ganador else "normal"
    nombre_v = eq_visit[:18] if len(eq_visit) > 18 else eq_visit
    ax.text(x - ANCHO_CAJA/2 + 0.15, y - ALTO_CAJA*0.45,
            nombre_v, ha="left", va="center",
            fontsize=7, color=color_v, fontweight=peso_v, zorder=4)
    ax.text(x + ANCHO_CAJA/2 - 0.15, y - ALTO_CAJA*0.45,
            str(gv), ha="right", va="center",
            fontsize=7, color=color_v, fontweight=peso_v, zorder=4)

    # Probabilidad
    ax.text(x, y - ALTO_CAJA - 0.12,
            f"{prob:.0%}", ha="center", va="top",
            fontsize=6, color=COLOR_GRIS, zorder=4)


# Calcular posiciones Y para cada ronda
def calcular_posiciones_y(n_partidos, y_min=1.0, y_max=14.0):
    if n_partidos == 0:
        return []
    espacio = (y_max - y_min) / n_partidos
    return [y_max - espacio * (i + 0.5) for i in range(n_partidos)]


posiciones_y = {}
centros_por_partido = {}  # para conectar líneas

for ronda in orden_rondas:
    partidos = rondas_dict.get(ronda, [])
    if not partidos:
        continue
    ys = calcular_posiciones_y(len(partidos))
    posiciones_y[ronda] = ys

    x = X_RONDAS[ronda]

    # Etiqueta de ronda
    ax_main.text(x, 14.3, ronda.upper(),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLOR_RONDA)

    for i, (row, y) in enumerate(zip(partidos, ys)):
        dibujar_partido(
            ax_main, x, y,
            row["Equipo Local"], row["Equipo Visitante"],
            int(row["Goles Local"]), int(row["Goles Visitante"]),
            row["Ganador"], row["Probabilidad Victoria"]
        )
        centros_por_partido.setdefault(ronda, []).append((x, y))

# Conectar rondas con líneas
def conectar_rondas(ax, ronda_a, ronda_b):
    centros_a = centros_por_partido.get(ronda_a, [])
    centros_b = centros_por_partido.get(ronda_b, [])
    if not centros_a or not centros_b:
        return
    x_a = centros_a[0][0] + ANCHO_CAJA/2
    x_b = centros_b[0][0] - ANCHO_CAJA/2
    x_mid = (x_a + x_b) / 2

    for j in range(0, len(centros_a), 2):
        if j+1 >= len(centros_a) or j//2 >= len(centros_b):
            break
        y1 = centros_a[j][1]
        y2 = centros_a[j+1][1]
        y_dest = centros_b[j//2][1]

        ax.plot([x_a, x_mid], [y1, y1], color=COLOR_LINEA, lw=1, zorder=1)
        ax.plot([x_a, x_mid], [y2, y2], color=COLOR_LINEA, lw=1, zorder=1)
        ax.plot([x_mid, x_mid], [y1, y2], color=COLOR_LINEA, lw=1, zorder=1)
        ax.plot([x_mid, x_b],  [y_dest, y_dest], color=COLOR_LINEA, lw=1, zorder=1)


pares_rondas = [
    ("Octavos de Final", "Cuartos de Final"),
    ("Cuartos de Final", "Semifinales"),
    ("Semifinales", "Final"),
]
for ra, rb in pares_rondas:
    conectar_rondas(ax_main, ra, rb)

# Trofeo / campeón a la derecha
final_partidos = rondas_dict.get("Final", [])
if final_partidos:
    x_trofeo = X_RONDAS["Final"] + ANCHO_CAJA/2 + 1.5
    y_final = posiciones_y.get("Final", [8])[0]
    ax_main.text(x_trofeo, y_final + 0.8, "🏆", ha="center", va="center", fontsize=30)
    ax_main.text(x_trofeo, y_final - 0.3, "CAMPEÓN", ha="center", va="center",
            fontsize=10, fontweight="bold", color=COLOR_DORADO)
    ax_main.text(x_trofeo, y_final - 0.85, campeon,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=COLOR_VERDE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_PANEL,
                      edgecolor=COLOR_DORADO, linewidth=1.5))

# Pie de figura
ax_main.text(12, 0.25,
        "Modelo: Random Forest + Regresión de Poisson  |  Datos: ELO, FIFA, Histórico Mundiales 1930-2022",
        ha="center", va="center", fontsize=8, color=COLOR_GRIS)

ruta_img = os.path.join(SALIDA, "bracket_mundial_2026.png")
plt.savefig(ruta_img, dpi=180, bbox_inches="tight",
            facecolor=COLOR_FONDO, edgecolor="none")
plt.close()
print(f"  ✓ Bracket guardado en: {ruta_img}")

# ─── Mostrar resumen en consola ───────────────────────────────
print("\n[4/4] Resumen de archivos generados:")
print(f"\n  📁 Directorio de salida: {SALIDA}")
archivos_salida = [
    ("predicciones_llave_2026.xlsx", "Excel con predicciones completas y formato"),
    ("bracket_mundial_2026.png",      "Bracket visual de la fase eliminatoria"),
]
for nombre, desc in archivos_salida:
    ruta = os.path.join(SALIDA, nombre)
    existe = "✓" if os.path.exists(ruta) else "✗"
    print(f"  {existe} {nombre} — {desc}")

print(f"\n{'═'*60}")
print(f"  VISUALIZACIÓN COMPLETADA")
print(f"  🏆 CAMPEÓN PREDICHO: {campeon}")
print("═"*60 + "\n")
