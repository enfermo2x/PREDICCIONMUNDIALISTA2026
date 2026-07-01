# Predicción Mundialista 2026
Sistema de predicción de la fase eliminatoria del **Mundial FIFA 2026** basado en Machine Learning. Predice partidos, simula la llave completa y genera visualizaciones profesionales usando datos históricos de Mundiales desde 1930.
---
## Que hace este proyecto
- Carga los resultados finales de la fase de grupos del Mundial 2026 (12 grupos, 72 partidos)
- Entrena **cinco modelos de Machine Learning** sobre partidos eliminatorios históricos (1930–2022)
- Simula **todas las 6 rondas** de la fase eliminatoria sin saltarse ninguna
- Genera un **bracket visual en alta resolución** (300 DPI) y un **Excel profesional** con 4 hojas
---
## Modelos entrenados
| # | Modelo | Arquitectura | Guardado como |
|---|--------|-------------|---------------|
| 1 | Random Forest | 400 árboles, depth=8, class_weight=balanced | `modelo_rf.pkl` |
| 2 | XGBoost / GradientBoosting | 300 estimadores, lr=0.05, fallback transparente | `modelo_xgb.pkl` (dict con flag `xgb_ok`) |
| 3 | SVM Calibrado | SVC kernel RBF, C=5.0, CalibratedClassifierCV | `modelo_svm.pkl` |
| 4 | Stacking Ensemble | RF + SVM como base, LogisticRegression como meta | `modelo_stack.pkl` |
| 5 | Regresión de Poisson | 2 modelos (λ local + λ visitante), Ridge como fallback | `modelo_poisson.pkl` (dict con claves "local"/"visitante") |
### Tabla comparativa de accuracy (CV 5-fold estratificada)
Ejemplo típico con datos históricos:
| Modelo | Accuracy CV |
|--------|-------------|
| Stacking Ensemble | 83.1% |
| Random Forest | 81.4% |
| XGBoost / GradientBoosting | 79.8% |
| SVM Calibrado | 77.2% |
| Poisson (MAE goles) | 0.72 (local) / 0.69 (visitante) |
---
## Variables predictoras (14 features)
| Variable | Descripción |
|----------|-------------|
| Diferencia ELO | Fuerza relativa según rating ELO actualizado |
| Diferencia ranking FIFA | Posición en el ranking FIFA oficial |
| Diferencia puntos FIFA | Puntos acumulados en ranking FIFA |
| Tasa H2H | Historial directo en fases eliminatorias |
| Goles anotados (local/visitante) | Promedio histórico en Copas del Mundo |
| Goles recibidos (local/visitante) | Promedio histórico en Copas del Mundo |
| Diferencia de goles netos | (GA-GR) local − (GA-GR) visitante |
| Tasa de penales (local/visitante) | Rendimiento histórico en tandas |
| Diferencia de títulos | Copas del Mundo ganadas |
| ELO absoluto (local/visitante) | Rating individual sin comparar |
Cuando el H2H es vacío (0.5), se resuelve con **probabilidad ELO proporcional** en lugar de asignar 50/50.
---
## Combinación de predicciones
Las probabilidades finales de cada partido se calculan combinando:
- **65% ensemble de modelos** (promedio de RF + XGB + SVM + Stacking)
- **35% ELO puro** (para suavizar anomalías en equipos sin historial)
Los goles se calculan combinando:
- **50% Poisson** (λ predicho por el modelo)
- **50% goles esperados ajustados por ELO**
---
## Rondas del torneo (nomenclatura oficial en español)
El sistema predice las **6 rondas** sin saltarse ninguna:
```
Dieciseisavos de Final  ->  32 equipos  ->  16 partidos
Octavos de Final        ->  16 equipos  ->   8 partidos
Cuartos de Final        ->   8 equipos  ->   4 partidos
Semifinales             ->   4 equipos  ->   2 partidos
Tercer y Cuarto Puesto  ->   2 equipos  ->   1 partido
Final                   ->   2 equipos  ->   1 partido
```
---
## Estructura del proyecto
```
PREDICCION MUNDIALISTA/
|
+-- 0_actualizar_datos.py       # Carga datos de grupos + cruces Dieciseisavos
+-- 1_cargar_datos.py           # Carga y limpia datasets (no modificar)
+-- 2_calcular_features.py      # Calcula features (no modificar)
+-- 3_entrenar_modelo.py        # Entrena 5 modelos + tabla comparativa
+-- 4_simular_llave.py          # Simula toda la llave con formato visual
+-- 5_visualizar_llave.py       # Bracket PNG + Excel 4 hojas
|
+-- requirements.txt
+-- README.md
|
+-- wc-data/                    # Datos principales
+   +-- WorldCup2026_grupos.csv (generado por script 0)
+   +-- Data Fuchibol/
+       +-- schedule_2026.csv   (generado por script 0)
+       +-- matches_1930_2022.csv
+       +-- ... otros datasets ...
|
+-- procesados/                 # Generado automáticamente
+   +-- features.pkl
+   +-- stats_individuales.pkl
+   +-- schedule_2026.pkl
+   +-- modelo_rf.pkl
+   +-- modelo_xgb.pkl
+   +-- modelo_svm.pkl
+   +-- modelo_stack.pkl
+   +-- modelo_poisson.pkl
+   +-- features_modelo.pkl
+   +-- label_encoder.pkl
+   +-- predicciones_llave.pkl
|
+-- salida/                     # Resultados finales
+   +-- bracket_mundial_2026.png
+   +-- predicciones_llave_2026.xlsx
|
+-- pasan_a_octavos.csv         # 16 partidos de Dieciseisavos con probabilidades
+-- predicciones_copa_mundial_2026.csv  # Todos los partidos
```
---
## Requisitos
- Python **3.8 o superior**
- Conexión a internet (solo para el Script 0, para descargar desde ESPN)
---
## Instalación
### 1. Preparar el entorno
```bash
# Crear entorno virtual
python -m venv .venv
# Activar
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate
# Instalar dependencias
pip install -r requirements.txt
```
### 2. Agregar los datos
Coloca los archivos de datos en las carpetas correspondientes. Los archivos históricos (`matches_1930_2022.csv`, rankings FIFA, ELO, etc.) se consiguen desde Kaggle y fuentes públicas indicadas más abajo.
---
## Uso
### Primera vez (configuración inicial)
Ejecuta los scripts **en orden**:
```bash
python 1_cargar_datos.py          # Carga datasets + genera features
python 2_calcular_features.py     # Calcula variables predictoras
python 3_entrenar_modelo.py       # Entrena los 5 modelos
```
Esto entrena los modelos y guarda todo en `procesados/`. Solo necesitas hacerlo **una vez**.
### Cada vez que quieras predecir o actualizar
```bash
python 0_actualizar_datos.py      # Carga datos de grupos + cruces
python 4_simular_llave.py         # Simula la llave completa
python 5_visualizar_llave.py      # Genera bracket PNG + Excel
```
---
## Como actualizar con el torneo en curso
### Fase de grupos
Los datos de fase de grupos están en `0_actualizar_datos.py` como datos manuales confirmados (72 partidos, 12 grupos A-L). Si quieres actualizarlos con los resultados reales, edita la lista `PARTIDOS_GRUPOS_FINALIZADOS`.
El Script 0 también intenta descargar desde la **API de ESPN** como respaldo:
```
https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard
```
Sin registro, sin API key. Los datos manuales prevalecen sobre ESPN porque la API puede quedar desfasada durante la transición entre fases.
### Cambiar los cruces eliminatorios
Edita la lista `ROUND_OF_32` en `0_actualizar_datos.py`:
```python
# CRUCES 100% CONFIRMADOS - EDITAR AQUÍ SI CAMBIA LA LLAVE
ROUND_OF_32 = [
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
```
Cada tupla es: `(local, visitante, fecha)`. Cuando avance una ronda, reemplaza los cruces con los de la siguiente ronda.
### Actualización rápida
Solo necesitas correr estos 3 scripts cuando cambien los datos:
```bash
python 0_actualizar_datos.py
python 4_simular_llave.py
python 5_visualizar_llave.py
```
Los modelos **no necesitan reentrenarse** a menos que cambies las features.
---
## Resultados generados
### Consola (Script 4)
Llave completa impresa ronda por ronda con:
- Bloque visual por partido con probabilidades por modelo (RF, XGB, SVM, Stacking, Poisson)
- Top 5 score probabilities
- Ganador y probabilidad combinada (65% ensemble + 35% ELO)
- Trofeo con campeón y probabilidad en la Final
- Top 5 favoritos con barras de progreso
- Resumen ejecutivo (promedio goles, partido más predecible, mayor sorpresa)
### Archivos de salida
| Archivo | Descripción |
|---------|-------------|
| `salida/bracket_mundial_2026.png` | Bracket visual 300 DPI, fondo azul noche, bordes redondeados, líneas Bezier |
| `salida/predicciones_llave_2026.xlsx` | Excel con 4 hojas detalladas |
| `pasan_a_octavos.csv` | 16 partidos de Dieciseisavos con probabilidades por modelo |
| `predicciones_copa_mundial_2026.csv` | Todos los partidos de todas las rondas |
### Hojas del Excel
| Hoja | Contenido |
|------|-----------|
| Predicciones | Tabla completa con todos los modelos, encabezados verde oscuro, filas alternas, final en dorado |
| Resumen por Ronda | Partidos, goles totales y promedio por ronda |
| Dashboard | Top 5 favoritos con escala de colores verde + 2 gráficos incrustados (barras de campeón + apilado por ronda) |
| Campeón | Trofeo ASCII en celda A1, nombre del campeón en fuente 24 dorado, fondo verde oscuro |
---
## Paleta de colores ANSI en consola
| Color | Código | Uso |
|-------|--------|-----|
| Dorado | `\033[38;5;220m` | Campeón, ganadores, encabezados de ronda |
| Verde | `\033[38;5;82m` | Barras de progreso, bordes de bloques |
| Blanco | `\033[97m` | Texto general |
| Gris | `\033[90m` | Equipos perdedores |
| Cian | `\033[38;5;51m` | Score probabilities |
| Azul | `\033[38;5;39m` | Acentos |
| Lila | `\033[38;5;135m` | Acentos |
---
## Grupos oficiales del Mundial 2026
| Grupo | Equipos |
|-------|---------|
| A | México, Qatar, Suiza, Camerún |
| B | Inglaterra, Panamá, Corea del Sur, Serbia |
| C | Brasil, Arabia Saudita, Estados Unidos, Noruega |
| D | Argentina, Argelia, Austria, Jordania |
| E | Alemania, Nueva Zelanda, Canadá, Islandia |
| F | Marruecos, Irak, Sudáfrica, Chile |
| G | Portugal, Indonesia, Colombia, Costa de Marfil |
| H | España, Bolivia, Japón, Ucrania |
| I | Francia, Fiyi, Suecia, Australia |
| J | Paraguay, Vietnam, Ecuador, Turquía |
| K | Países Bajos, Uzbekistán, República Checa, RD Congo |
| L | Croacia, Honduras, Ghana, Senegal |
---
## Limitaciones conocidas
- El modelo no considera lesiones, sanciones ni rotaciones de plantilla
- Equipos sin historial en Mundiales usan ELO como respaldo, lo que puede generar predicciones conservadoras
- La API de ESPN puede cambiar su estructura sin previo aviso
- Los resultados son probabilísticos, no deterministas
- No incluye análisis táctico ni formaciones
---
## Contribuir
1. Haz fork del repositorio
2. Crea una rama: `git checkout -b feature/mejora`
3. Haz tus cambios y commitea: `git commit -m 'Agrega mejora X'`
4. Push: `git push origin feature/mejora`
5. Abre un Pull Request
---
## Creditos y fuentes de datos
- **Resultados en vivo**: ESPN API pública (`site.api.espn.com`)
- **Resultados históricos internacionales**: Kaggle - International Football Results
- **Partidos de Mundiales**: Kaggle - FIFA World Cup Dataset
- **Ratings ELO**: ClubElo / EloRatings
- **Rankings FIFA**: FIFA World Rankings oficiales
- **Metodología ML**: scikit-learn (Random Forest, SVM, Stacking, GradientBoosting, Poisson)
- **Visualización**: matplotlib, openpyxl
---
## Licencia
MIT License — libre para usar, modificar y distribuir con atribución.
---
*Proyecto desarrollado en Python 3.8+. Todos los textos y outputs en español. Mundial FIFA 2026 — Estados Unidos, Mexico y Canada.*

Created requirements.txt
# Core
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0
# Machine Learning
scikit-learn>=1.1.0
xgboost>=1.7.0
# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
# Excel output
openpyxl>=3.0.10
# HTTP requests (Script 0 - ESPN API)
requests>=2.28.0
# Optional: stats models for Poisson diagnostics
statsmodels>=0.13.0