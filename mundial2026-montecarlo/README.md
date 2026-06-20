# Mundial 2026 — Simulación Monte Carlo

Simulación de Monte Carlo del **Mundial FIFA 2026** (48 selecciones) que estima
**en cuántos escenarios sale campeona cada selección**, partiendo de:

1. el **ELO** de cada selección, y
2. las **tablas de posiciones actuales** de los 12 grupos (snapshot en curso del torneo).

Se simula miles de veces el resto del torneo (partidos de grupo que faltan +
toda la llave de eliminación) y se cuenta cuántas veces gana cada equipo.

> Solo usa la **biblioteca estándar de Python** (sin numpy/pandas), en línea con
> la filosofía de `montecarlo-calc`.

---

## Uso

```bash
python3 run.py                       # 20.000 simulaciones (por defecto)
python3 run.py -n 100000 --seed 2026 # más simulaciones = menos ruido de muestreo
python3 run.py -n 50000 --out resultados.csv --top 30
```

Opciones:

| flag | qué hace | default |
|------|----------|---------|
| `-n, --num`  | cantidad de escenarios simulados | `20000` |
| `--seed`     | semilla del RNG (reproducibilidad) | aleatoria |
| `--base`     | goles esperados base por equipo | `1.35` |
| `--home-adv` | bonus de ELO por localía (sedes USA/México/Canadá) | `60` |
| `--out`      | CSV con los resultados completos | — |
| `--top`      | cuántas selecciones mostrar en pantalla | `20` |

---

## Modelo

**Por partido** se usa un modelo Poisson cuyas medias salen de la diferencia de ELO:

```
λ_A = base · 10^( (ELO_A − ELO_B) / 800 )
λ_B = base · 10^(−(ELO_A − ELO_B) / 800 )
goles_A ~ Poisson(λ_A),   goles_B ~ Poisson(λ_B)
```

Esto hace que el cociente de goles esperados sea `10^(ΔELO/400)` (la forma clásica
del ELO) y mantiene el total de goles aproximadamente constante. Las sedes
(USA, México, Canadá) reciben un bonus de ELO por localía **solo cuando juegan en su
propio país**: en grupos siempre son locales; en la eliminación, la sede de cada partido
está fijada por el calendario oficial (`venue` en `bracket.json`), de modo que un anfitrión
puede jugar fuera de su país (la final y todo desde cuartos se juega en EE. UU.).

**Fase de grupos:** se arranca del estado **actual** (puntos, GF, GA del snapshot) y
se simulan solo los partidos que faltan. El orden final de cada grupo se resuelve por
puntos → diferencia de gol → goles a favor → ELO. Clasifican los 2 primeros de cada
grupo + los **8 mejores terceros**.

**Eliminación directa:** se arma la **Ronda de 32 con la plantilla oficial 2026**
(`data/bracket.json`, M73–M88) y se juega la llave hasta la final (M103). Los 8
terceros se asignan a sus ranuras respetando los grupos admitidos por cada una
(matching bipartito). En la llave, el empate se resuelve por penales con
probabilidad derivada del ELO.

---

## Datos (`data/`)

| archivo | contenido |
|---------|-----------|
| `elo.csv`      | ELO de las 48 selecciones (escala clásica de worldfootballrankings; los equipos sin dato público están marcados `estimado`) |
| `groups.csv`   | snapshot de los 12 grupos: partidos jugados, puntos, GF, GA |
| `fixtures.csv` | partidos de grupo que **faltan** jugar |
| `bracket.json` | plantilla oficial de la Ronda de 32 y el árbol hasta la final |

Para **actualizar** la simulación a medida que avanza el Mundial: editá `groups.csv`
(puntos/goles) y sacá de `fixtures.csv` los partidos ya jugados. El modelo se adapta solo.

**Fuentes del snapshot (≈ 20-jun-2026):** tablas de [CBS Sports](https://www.cbssports.com/soccer/news/world-cup-group-standings-table-results/)
y [NBC Sports](https://www.nbcsports.com/soccer/news/2026-world-cup-group-stage-table-full-standings-for-all-12-groups);
ELO de [worldfootballrankings.com](https://worldfootballrankings.com/rankings);
bracket de [worldcuppass.com](https://worldcuppass.com/world-cup-2026-round-of-32/).

---

## Resultado (1.000.000 de escenarios, `--seed 2026`)

| # | Selección | ELO | P(campeón) | P(final) | P(semi) |
|---|-----------|-----|-----------:|---------:|--------:|
| 1 | Argentina | 1889 | **28.6 %** | 46.7 % | 68.0 % |
| 2 | France | 1887 | **25.7 %** | 43.3 % | 62.2 % |
| 3 | Spain | 1856 | **14.3 %** | 26.2 % | 47.5 % |
| 4 | England | 1848 | **12.0 %** | 23.4 % | 44.7 % |
| 5 | USA | 1710 | 3.8 % | 10.2 % | 27.5 % |
| 6 | Morocco | 1770 | 3.0 % | 8.2 % | 20.2 % |
| 7 | Brazil | 1772 | 2.9 % | 8.1 % | 19.7 % |
| 8 | Portugal | 1755 | 2.1 % | 6.0 % | 15.5 % |
| 9 | Netherlands | 1764 | 2.0 % | 5.6 % | 13.5 % |
| 10 | Germany | 1744 | 1.4 % | 4.5 % | 11.7 % |
| 11 | Mexico | 1722 | 1.3 % | 4.6 % | 16.1 % |

En total **33 selecciones** salieron campeonas en al menos un escenario. El detalle
completo de las 48 (campeón / final / semi) está en [`resultados_1M.csv`](resultados_1M.csv).
La columna `P(semi)` deja a la vista que la **grilla de eliminatorias completa** (Ronda de
32 → octavos → cuartos → semi → final) entra en el cómputo, no solo la final.

> **Localía geográfica:** la ventaja de sede solo se aplica cuando un anfitrión juega en su
> país. Por eso México (local en R32/octavos pero no en las rondas finales, todas en EE. UU.)
> llega seguido a instancias intermedias —P(semi) ≈ 16 %— pero su P(campeón) cae a ~1.3 %.

---

## Notebook / paper (`mundial2026.ipynb`)

[`mundial2026.ipynb`](mundial2026.ipynb) está redactado como un **paper académico**: resumen,
introducción, datos, metodología formal (ELO, modelo Poisson, estructura del torneo y
**estimador de Monte Carlo con error estándar e intervalos de confianza**), resultados con
**análisis de convergencia** y **de sensibilidad**, discusión, conclusiones y referencias.
Corre **1.000.000 de escenarios** y produce las figuras. Para abrirlo:

```bash
uv sync --extra notebook         # instala matplotlib/pandas/jupyter (solo para el notebook)
uv run --extra notebook jupyter lab mundial2026.ipynb
```

> El **motor** (`wcsim.py`) sigue siendo stdlib puro; estas dependencias son solo para la
> presentación. El entorno queda fijado en `uv.lock`. Para regenerar el notebook desde cero:
> `uv run --extra notebook python _build_notebook.py`.

### Figuras

![Probabilidad de campeonato top-15 con IC 95%](charts/03_campeon_top15.png)

![Avance por la grilla de eliminatorias](charts/04_avance_grilla.png)

![Convergencia del estimador de Monte Carlo](charts/06_convergencia.png)

### Apéndice A — robustez al modelo de goles

El baseline (Ec. 3) prioriza calibrar el ELO, pero como contrapartida los goles totales
**explotan** en partidos desparejos (~7.6 esperados en Argentina–Haití). El apéndice evalúa una
variante con **total de goles fijo** (`MatchModel(total_goals=T)`) que reparte las intensidades
preservando el puntaje esperado del ELO (la diferencia de goles sigue una **Skellam**). Resultado:
el *ordenamiento* de favoritas es robusto, pero las *magnitudes* se mueven —acotar el total
transfiere ~4 pp de Argentina/Francia al pelotón medio—, así que la hipótesis de goles **no es
inocua** y el modelo realista vive entre ambos regímenes (lo fija una calibración tipo Dixon-Coles).

![Goles totales esperados: baseline vs total fijo](charts/07_goles_totales.png)

![Sensibilidad de P(campeón) al modelo de goles](charts/08_sensibilidad_goles.png)

---

## Limitaciones / supuestos

- El ELO es un proxy de fuerza; no modela lesiones, suspensiones ni el estado de forma.
- Goles modelados como Poisson **independientes** (sin correlación ni efecto de marcador).
- La asignación de terceros usa un matching válido respetando los grupos admitidos,
  no la tabla FIFA exacta de 495 combinaciones (efecto de segundo orden sobre el campeón).
- Algunos ELO de selecciones menores están **estimados** (ver columna `fuente` en `elo.csv`).
