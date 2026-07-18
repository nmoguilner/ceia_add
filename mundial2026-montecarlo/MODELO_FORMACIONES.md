# Modelo de formaciones (proyecto en curso)

Objetivo: un modelo a **nivel jugador/formación** que prediga los resultados **fecha a
fecha**, corriendo **en paralelo al Elo de equipo**, y que se **afine online** a medida que se
juegan los partidos. La idea central: *no es lo mismo Argentina con Messi que sin él* — el Elo de
equipo no lo ve; un modelo de formación sí.

## Decisiones de diseño

- **Valor del jugador:** semilla con **ratings FIFA/EA** (overall), actualizado online con los
  resultados (rating de jugador estilo Elo, sembrado por FIFA).
- **Timing:** predice con la **XI esperada** (proyección anticipada de la fecha) → requiere un
  **predictor de formación probable** (titulares según minutos/uso reciente).
- Dos motores, **dos predicciones por partido** (formación vs Elo), con un marcador en vivo
  (RPS/Brier) que muestra cuál predice mejor a medida que avanza el torneo.

## Arquitectura

```
ratings FIFA (semilla) ─┐
                        ├─> rating por jugador ──> agregación del XI ──> fuerza de equipo ┐
update online (Elo jug.)┘     (predictor de XI esperada)                                  ├─> modelo de partido (Poisson)
Elo de equipo reconstruido ──────────────────────────────────────────── feature ────────┘        + el MC que ya tenemos
                                                                                                   └─> proyección del torneo
```

## Datos (fuentes confirmadas)

| Fuente | Para qué | Estado |
|--------|----------|--------|
| **StatsBomb Open Data** | formaciones (XI titular) de WC 2018/2022, Euro 2020/2024, Copa América 2024, AFCON 2023 | ✅ bajado (`lineups_data.py` → `data/lineups.parquet`, 314 partidos, 1.797 jugadores, 34/48 selecciones del Mundial) |
| **Ratings FIFA/EA** | semilla de calidad por jugador | público (Hugging Face / repos sofifa) — pendiente bajar + **matchear nombres** |
| **api-football** | XI y resultados **en vivo** del Mundial 2026 | requiere API key (Fase 2) |
| `history.parquet` + Elo reconstruido | baseline Elo y feature de contexto | ✅ ya en el repo |

## Fases

- **Fase 1 (offline, el experimento):** formaciones + ratings FIFA → features de XI → modelo de
  partido → **backtest vs Elo** (RPS/Brier en partidos retenidos). Responde: *¿la info de
  formación le gana al Elo?*
  - [x] Paso 0 — dataset de formaciones (`data/lineups.parquet`, `data/sb_matches.parquet`).
  - [ ] Paso 1 — ratings FIFA + matcheo de nombres a los 1.797 jugadores.
  - [ ] Paso 2 — features de XI (overall por línea GK/DEF/MID/ATT, factor estrella, profundidad) + Elo.
  - [ ] Paso 3 — entrenar Poisson (GBM) y backtestear vs Elo.
- **Fase 2 (en vivo):** predictor de XI esperada + api-football + actualización por fecha +
  carrera de los dos modelos + publicación (Cloudflare Pages). Necesita API keys.

## Caveats honestos

- **El cuello de botella son los datos, no el modelo.** El 80% del trabajo es bajar/limpiar y
  **matchear nombres** (jugadores entre StatsBomb y FIFA; selecciones entre fuentes).
- Los ratings de jugadores **correlacionan fuerte con el Elo** (los buenos equipos tienen buenos
  jugadores), así que la **señal extra** de la formación viene casi toda de **rotaciones,
  ausencias y lesiones**. Es real pero fina, y con **pocos datos** (314 partidos con formación).
- Por eso la **carrera en vivo de los dos modelos es el experimento correcto**: quizá el modelo
  de formación arranque por debajo del Elo y lo alcance al juntar partidos del propio Mundial. Si
  no le gana, también es un resultado válido (gran apéndice: *"el Elo es difícil de batir"*).
