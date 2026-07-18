# Fuentes de datos (procedencia y licencias)

Documento de procedencia de todos los datasets del proyecto, estilo *datasheet*.
Distingue **datos primarios** (insumos externos) de **datos derivados** (los genera
el código del repo). Última revisión: **27-jun-2026**.

Convención del repo: las **fuentes editables a mano** viven en `data/sources/*.csv`;
`convert_to_parquet.py` las convierte (más el histórico) a los **Parquet canónicos**
en `data/*.parquet`, que es lo que leen `wcsim.py`, `calibrate.py` y `elo_history.py`.

---

## 1. Datos primarios (insumos externos)

### 1.1 ELO de las selecciones — `data/sources/elo.csv` → `elo.parquet`
- **Contenido:** rating ELO de las 48 selecciones (48 filas: `team, elo, fuente`).
- **Origen:** [worldfootballrankings.com/rankings](https://worldfootballrankings.com/rankings),
  que aplica la metodología *World Football Elo Ratings* (escala clásica de
  [eloratings.net](https://eloratings.net): K variable, multiplicador por margen de gol).
- **Obtención:** **manual** (copiado del ranking). Las 48 selecciones tienen `fuente =
  worldfootballrankings` (al 27-jun-2026 ya no quedan valores `estimado`).
- **Licencia:** sitio público de rankings; sin licencia formal de redistribución. Se usa
  como referencia numérica, con atribución.
- **Escala:** ojo, **NO** es la de eloratings.net. En worldfootballrankings Argentina ≈ 1900
  y New Zealand ≈ 1270; en eloratings.net los mismos equipos están ~200 puntos más arriba.
  Al actualizar ELOs hay que tomarlos **siempre de worldfootballrankings** para no mezclar escalas.
- **Nota de consistencia temporal:** los 35 valores de las potencias son del snapshot inicial
  (Argentina 1889); los 13 que antes eran `estimado` se reemplazaron por el valor real de
  worldfootballrankings al 27-jun (Argentina ahí ≈ 1902). El desfase (~13 pts, uniforme) es
  despreciable frente a la corrección, pero conviene re-bajar las 48 juntas si se quiere
  consistencia estricta de fecha.

### 1.2 Estado del torneo — `groups.csv`, `fixtures.csv`, `played.csv`
- **Contenido:**
  - `groups.csv` (→ `groups.parquet`, 48 filas): tabla de cada grupo (PJ, pts, GF, GA).
  - `fixtures.csv` (→ `fixtures.parquet`): partidos de grupo que **faltan** jugar.
  - `played.csv` (→ `played.parquet`): partidos del Mundial **ya disputados** (para el
    backtest del Apéndice D y el modo de ELO dinámico).
- **Origen:** prensa deportiva —
  [ESPN](https://www.espn.com/soccer/story/_/id/48939282/2026-fifa-world-cup-fixtures-results-match-schedule-group-stage-knockout-rounds-bracket),
  [CBS Sports](https://www.cbssports.com/soccer/news/world-cup-group-standings-table-results/),
  [NBC Sports](https://www.nbcsports.com/soccer/news/2026-world-cup-group-stage-table-full-standings-for-all-12-groups).
- **Obtención:** **manual**. `groups.csv` se reconstruye desde `played.csv` con
  `reconciliar.py` (suma PJ/pts/GF/GA y deja en `fixtures.csv` solo lo no jugado),
  lo que mantiene log y tabla consistentes.
- **Cobertura (al 27-jun-2026):** `played` = 66 partidos (11-jun a 26-jun, fases de grupo
  A–I completas); `fixtures` = 6 pendientes (grupos J, K y L, que se juegan el 27-jun);
  `groups` = A–I con PJ=3, J/K/L con PJ=2.
- **Limitación:** carga **a mano** desde prensa → riesgo de desfase entre `played` y la tabla
  (de hecho se corrigió el 27-jun una desincronización de A/B/C/E/F). Verificar siempre por
  marcadores, no por las "tablas finales" derivadas de algunas fuentes (que pueden venir mal
  calculadas; p. ej. el orden del Grupo G).

### 1.3 Bracket de eliminación directa — `data/bracket.json`
- **Contenido:** plantilla oficial de la Ronda de 32 (M73–M88, con grupos admitidos por cada
  ranura de "tercero" y sede de cada partido) y el árbol hasta la final (M103).
- **Origen:** [worldcuppass.com/world-cup-2026-round-of-32](https://worldcuppass.com/world-cup-2026-round-of-32/)
  (estructura oficial FIFA del cuadro 2026).
- **Obtención:** manual. **Simplificación:** la asignación de los 8 mejores terceros usa
  *matching* bipartito sobre los grupos admitidos, no la tabla FIFA exacta de 495
  combinaciones (efecto de segundo orden sobre el campeón).

### 1.4 Histórico de partidos internacionales — `history.parquet`
- **Contenido:** **49.477** partidos internacionales (`date, home_team, away_team,
  home_score, away_score, tournament, neutral`).
- **Origen:** [martj42/international_results](https://github.com/martj42/international_results)
  (GitHub), descargado por `convert_to_parquet.py` desde
  `raw.githubusercontent.com/.../results.csv`.
- **Obtención:** **automática** (cacheado en `/tmp/intl_results.csv`).
- **Cobertura:** 1872-11-30 a 2026-06-27.
- **Licencia:** dataset público; publicado como **CC0** en su versión de Kaggle. Atribución a martj42.
- **Uso:** base de la calibración del modelo de goles (Apéndices B y C).

### 1.5 Alineaciones — `sb_matches.parquet`, `lineups.parquet`
- **Contenido:** `sb_matches` = 314 partidos (2018–2024); `lineups` = 6.908 filas
  jugador-partido (`match_id, team, player, position, minutes`).
- **Origen:** [StatsBomb Open Data](https://github.com/statsbomb/open-data),
  bajado por `lineups_data.py`.
- **Licencia:** **gratis con atribución, uso no comercial** (StatsBomb Open Data User Agreement).
- **Estado:** insumo de la rama **modelo de formaciones** (en desarrollo, ver
  `MODELO_FORMACIONES.md`). **NO alimenta el simulador actual.**

---

## 2. Datos derivados (los genera el código del repo)

| Archivo | Lo genera | A partir de |
|---|---|---|
| `calibration.json` | `calibrate.py` | MLE (regresión de Poisson IRLS) sobre `history.parquet` desde 2023-01-01: **435** partidos / 870 obs → `escala≈1281`, `μ≈1.21`, `home_adv≈87` ELO, `ρ≈-0.168` (Dixon-Coles). Apéndice B. |
| `calibration_prematch.json` | `elo_history.py` | Igual MLE pero con **ELO pre-partido reconstruido** (sin proxy): **3.630** partidos → `escala≈1306`, `home_adv≈127`. Valida `corr_vs_wfr = 0.917`. Apéndice C. |
| `elo_reconstructed.parquet` | `elo_history.py` | Corre el algoritmo eloratings sobre los 49k partidos del histórico. |
| `backtest_predictions.parquet` | `backtest.py` | P(H/E/A) analítica vs resultado real de cada partido jugado (acierto, Brier, log-loss). Apéndice D. |
| `resultados_1M.parquet` | `run.py` | Salida del Monte Carlo: P(campeón/final/semi) por selección. |

---

## 3. Resumen de la cadena

```
worldfootballrankings ─┐
prensa (ESPN/CBS/NBC) ─┼─► data/sources/*.csv ─(convert_to_parquet.py)─► data/*.parquet ─► wcsim.py ─► resultados_1M
worldcuppass (bracket)─┘                                                      ▲
martj42 (histórico) ──────────────────────────────► history.parquet ─(calibrate/elo_history)─┘ (calibración)
StatsBomb (alineaciones) ─► lineups.parquet ─► [modelo de formaciones, en desarrollo]
```

**Debilidades reconocidas:** (1) estado del torneo cargado a mano desde prensa;
(2) asignación de terceros simplificada vs la tabla FIFA de 495 casos; (3) leve desfase
temporal entre los 35 ELO del snapshot inicial y los 13 actualizados al 27-jun.

> **Histórico:** hasta el 27-jun-2026, 13 selecciones de perfil bajo/debutantes tenían ELO
> `estimado` a mano (banda 1300–1500). Se reemplazaron por los valores reales de
> worldfootballrankings; resultaron estar mayormente inflados (p. ej. New Zealand 1400→1270,
> Bosnia 1500→1409). Impacto sobre P(campeón) de las potencias: < 0.3 pp (segundo orden).
