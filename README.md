# 🏆 Predicción Mundialista 2026

> Sistema de predicción de la fase eliminatoria del **Mundial FIFA 2026** basado en Machine Learning. Predice partidos, simula la llave completa y genera visualizaciones profesionales usando datos históricos de Mundiales desde 1930.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge)
![matplotlib](https://img.shields.io/badge/matplotlib-Viz-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## ¿Qué hace este proyecto?

- Descarga automáticamente los resultados reales del Mundial 2026 desde la **API pública de ESPN** (sin registro, sin API key)
- Calcula 20+ variables predictoras por equipo: ELO, ranking FIFA, historial directo, goles, penales, títulos
- Entrena un **clasificador Random Forest** para predecir el resultado de cada partido
- Entrena modelos de **regresión de Poisson** para predecir el marcador exacto
- Simula toda la fase eliminatoria ronda por ronda propagando ganadores automáticamente
- Genera un **bracket visual en alta resolución** y un **Excel profesional** con probabilidades

---

## Capturas

```
╔══════════════════════════════════════════════════════════╗
║   🏆  PREDICCIÓN MUNDIALISTA 2026 - LLAVE ELIMINATORIA   ║
╠══════════════════════════════════════════════════════════╣
║  DIECISEISAVOS DE FINAL                                  ║
║   🇧🇷 Brasil         2-1  🇯🇵 Japón     → Brasil  (61%)  ║
║   🇫🇷 Francia        2-1  🇸🇪 Suecia    → Francia (71%)  ║
║   🇦🇷 Argentina      2-1  🇨🇻 Cabo Verde → Argentina(66%)║
║  ...                                                     ║
║  🏅 CAMPEÓN PREDICHO: Brasil                             ║
╚══════════════════════════════════════════════════════════╝
```

---

## Estructura del proyecto

```
PREDICCION MUNDIALISTA/
│
├── 0_actualizar_datos.py       ← Descarga resultados desde ESPN + carga cruces reales
├── 1_cargar_datos.py           ← Carga y limpia todos los datasets
├── 2_ingenieria_features.py    ← Calcula variables predictoras
├── 3_entrenar_modelo.py        ← Entrena Random Forest + Poisson
├── 4_simular_llave.py          ← Simula toda la fase eliminatoria
├── 5_visualizar_llave.py       ← Bracket visual + Excel profesional
│
├── requirements.txt
├── README.md
│
├── wc-data/                    ← Datos principales (debes proveer)
│   ├── elo_rankings.json
│   ├── fifa_rankings.json
│   ├── former_names.csv
│   ├── goalscorers.csv
│   ├── results.csv
│   ├── shootouts.csv
│   └── WorldCup2026.xlsx
│
│   └── Data Fuchibol/          ← Datos complementarios (debes proveer)
│       ├── fifa_ranking_2022-10-06.csv
│       ├── fifa_ranking_2026-06-08.csv
│       ├── matches_1930_2022.csv
│       ├── schedule_2026.csv
│       └── world_cup.csv
│
├── procesados/                 ← Generado automáticamente
└── salida/                     ← Resultados finales
    ├── predicciones_llave_2026.xlsx
    └── bracket_mundial_2026.png
```

---

## Requisitos

- Python **3.8 o superior**
- pip actualizado
- Conexión a internet (para el Script 0)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/prediccion-mundialista-2026.git
cd prediccion-mundialista-2026
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Agregar los datos

Coloca los archivos de datos en las carpetas correspondientes según la estructura de arriba. Los archivos necesarios son:

| Archivo | Dónde conseguirlo |
|---|---|
| `results.csv`, `goalscorers.csv`, `shootouts.csv`, `former_names.csv` | [Kaggle - International football results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| `elo_rankings.json`, `fifa_rankings.json` | [ClubElo](http://clubelo.com) / [FIFA Rankings](https://www.fifa.com/fifa-world-ranking) |
| `matches_1930_2022.csv`, `world_cup.csv` | [Kaggle - FIFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) |
| `fifa_ranking_2026-06-08.csv` | [Kaggle - FIFA Rankings 2026](https://www.kaggle.com) |
| `schedule_2026.csv` | Generado por `0_actualizar_datos.py` |
| `WorldCup2026.xlsx` | Datos propios del torneo 2026 |

> Los archivos de datos **no están incluidos** en el repositorio por tamaño y derechos. Consulta las fuentes indicadas.

---

## Uso

### Primera vez (configuración inicial)

Ejecuta los scripts **en orden**:

```bash
python 1_cargar_datos.py
python 2_ingenieria_features.py
python 3_entrenar_modelo.py
```

Esto entrena los modelos y guarda todo en `procesados/`. Solo necesitas hacerlo una vez.

### Cada vez que quieras predecir

```bash
python 0_actualizar_datos.py   # Descarga resultados reales de ESPN
python 4_simular_llave.py      # Simula la llave
python 5_visualizar_llave.py   # Genera bracket y Excel
```

Solo estos tres. El script 0 actualiza automáticamente los datos sin que toques nada.

---

## Cómo se actualiza con el torneo en curso

El Script 0 hace tres cosas automáticamente cada vez que lo corres:

1. **Descarga desde ESPN API** los resultados reales de todos los partidos de fase de grupos ya jugados
2. **Carga los cruces eliminatorios** confirmados oficialmente (Round of 32, Round of 16, etc.)
3. **Actualiza el pickle** del schedule para que el Script 4 lo lea directamente sin reentrenar

Cuando avance una ronda eliminatoria, actualiza la lista `ROUND_OF_32` en `0_actualizar_datos.py` con los cruces de la siguiente ronda y vuelve a correr los 3 scripts.

---

## Metodología

### Variables predictoras (Features)

| Variable | Descripción |
|---|---|
| Diferencia ELO | Fuerza relativa según rating ELO actualizado |
| Diferencia ranking FIFA | Posición en el ranking FIFA oficial |
| Diferencia puntos FIFA | Puntos acumulados en ranking FIFA |
| Tasa H2H | Historial directo en fases eliminatorias de Mundiales |
| Goles anotados/recibidos | Promedio histórico en Copas del Mundo |
| Tasa de penales | Rendimiento histórico en tandas de penales |
| Títulos mundiales | Número de Copas del Mundo ganadas |

### Modelos

**Random Forest Classifier**
- Predice resultado: victoria local / empate / victoria visitante
- 300 árboles, profundidad máxima 8, balanceo de clases
- Validación cruzada 5-fold estratificada
- Probabilidades finales: 60% Random Forest + 40% ELO (para equipos sin historial)

**Regresión de Poisson**
- Dos modelos independientes: goles del local y goles del visitante
- Combinado 50% Poisson + 50% goles esperados ajustados por ELO
- En caso de empate (imposible en eliminación directa), se resuelve por probabilidad

---

## Rondas del torneo

El sistema predice las **6 rondas eliminatorias** del Mundial 2026:

```
Round of 32 (Dieciseisavos)  →  32 equipos  →  16 partidos
Round of 16 (Octavos)        →  16 equipos  →   8 partidos
Cuartos de Final             →   8 equipos  →   4 partidos
Semifinales                  →   4 equipos  →   2 partidos
Tercer y Cuarto Puesto       →   2 equipos  →   1 partido
Final                        →   2 equipos  →   1 partido
```

---

## Resultados esperados

Después de correr los scripts encontrarás:

- **`salida/bracket_mundial_2026.png`** — Bracket visual en alta resolución con todas las rondas, marcadores predichos, probabilidades y campeón destacado
- **`salida/predicciones_llave_2026.xlsx`** — Excel con 4 hojas: predicciones completas, resumen por ronda, dashboard de favoritos y hoja del campeón
- **Consola** — Llave completa impresa ronda por ronda con colores, barras de probabilidad y Top 5 de favoritos

---

## Personalización

### Cambiar los cruces eliminatorios

Edita la lista `ROUND_OF_32` en `0_actualizar_datos.py`:

```python
ROUND_OF_32 = [
    ("Brasil",     "Japón",    "29/06/2026"),
    ("Francia",    "Suecia",   "30/06/2026"),
    # ... etc
]
```

### Reentrenar el modelo

Solo si cambias las features (Script 2). Si solo cambias los cruces o el schedule, no es necesario:

```bash
python 2_ingenieria_features.py
python 3_entrenar_modelo.py
```

### Cambiar idioma de nombres de equipos

Edita el diccionario `NOMBRES_ES` en `1_cargar_datos.py`.

---

## Dependencias

```
pandas>=1.5.0
scikit-learn>=1.1.0
numpy>=1.23.0
matplotlib>=3.6.0
openpyxl>=3.0.10
statsmodels>=0.13.0
scipy>=1.9.0
seaborn>=0.12.0
requests>=2.28.0
beautifulsoup4>=4.11.0
```

---

## Limitaciones conocidas

- El modelo no considera lesiones, sanciones ni rotaciones de plantilla
- Equipos sin historial en Mundiales usan ELO como respaldo, lo que puede generar predicciones conservadoras
- La API de ESPN puede cambiar su estructura sin previo aviso
- Los resultados son probabilísticos, no deterministas

---

## Contribuir

1. Haz fork del repositorio
2. Crea una rama: `git checkout -b feature/mejora`
3. Haz tus cambios y commitea: `git commit -m 'Agrega mejora X'`
4. Push: `git push origin feature/mejora`
5. Abre un Pull Request

---

## Créditos y fuentes de datos

- **Resultados en vivo**: ESPN API pública (`site.api.espn.com`)
- **Resultados históricos internacionales**: Kaggle - International Football Results
- **Partidos de Mundiales**: Kaggle - FIFA World Cup Dataset
- **Ratings ELO**: ClubElo / EloRatings
- **Rankings FIFA**: FIFA World Rankings oficiales
- **Metodología ML**: scikit-learn (Random Forest, Poisson)
- **Visualización**: matplotlib, openpyxl

---

## Licencia

MIT License — libre para usar, modificar y distribuir con atribución.

---

*Proyecto desarrollado en Python 3.8+. Todos los outputs en español. Mundial FIFA 2026 — Estados Unidos, México y Canadá.*