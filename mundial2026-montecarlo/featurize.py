#!/usr/bin/env python3
"""
Hito 1 del TP de Aprendizaje de Maquina: construye la tabla de entrenamiento.

Entrada:  data/elo_prematch.parquet  (serie de ELO PRE-PARTIDO de todo el
          historico, generada por elo_history.py).
Salida:   data/ml_dataset.parquet    (una fila por partido del rango de modelado,
          con las features de la seccion 6 del diseno, la etiqueta {H,D,A} y la
          columna de split train/test).

Regla de split (ciclo mundialista, DISENO_TP_ML.md sec. 7):
  - train: partidos 2022-11-20 .. 2026-06-10  (ciclo Qatar 2022 -> vispera 2026)
  - test:  Mundial 2026 real (>= 2026-06-11, torneo "FIFA World Cup")
  - el resto del historico NO se emite; solo se usa para calcular los rezagos
    (forma, descanso, h2h) sin fuga temporal.

Sin fuga: cada feature de un partido mira SOLO partidos anteriores a el. El loop
es cronologico y actualiza el estado por equipo DESPUES de emitir la fila.

  uv run python featurize.py
"""
import os
from collections import defaultdict, deque

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "elo_prematch.parquet")
OUT = os.path.join(HERE, "data", "ml_dataset.parquet")

TRAIN_DESDE, TRAIN_HASTA = "2022-11-20", "2026-06-10"  # ciclo Qatar 2022 -> vispera
TEST_DESDE = "2026-06-11"                               # arranque del Mundial 2026
FORM_N = 5      # ventana de forma (ultimos N partidos)
H2H_N = 5       # ventana de head-to-head reciente

# Marcadores de confederacion: el equipo se asigna a la confederacion cuyos
# torneos continentales jugo mas veces. Friendly / World Cup / Olympics son
# neutrales (no asignan). Frases completas para no confundir "UEFA Nations
# League" con "CONCACAF Nations League".
CONF_MARKERS = [
    ("UEFA",     ["uefa euro", "uefa nations league"]),
    ("CONMEBOL", ["copa américa", "copa america"]),
    ("CAF",      ["african cup of nations"]),
    ("AFC",      ["afc asian cup"]),
    ("CONCACAF", ["concacaf", "gold cup"]),
    ("OFC",      ["oceania nations cup"]),
]


def tournament_weight(t):
    """Ordinal de importancia/intensidad: amistoso < eliminatoria < copa < Mundial."""
    t = t.lower()
    if t == "friendly":
        return 1.0
    if "world cup" in t and "qual" not in t:
        return 4.0
    if "qualif" in t:
        return 2.0
    return 3.0   # campeonatos continentales y demas copas


def is_world_cup(t):
    t = t.lower()
    return "world cup" in t and "qual" not in t


def infer_confederations(rows):
    """Confederacion por equipo via mayoria de torneos continentales jugados."""
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        t = r["tournament"].lower()
        for conf, markers in CONF_MARKERS:
            if any(m in t for m in markers):
                counts[r["home_team"]][conf] += 1
                counts[r["away_team"]][conf] += 1
                break
    conf = {}
    for team, c in counts.items():
        conf[team] = max(c.items(), key=lambda kv: kv[1])[0]
    return conf   # equipos sin torneo continental -> ausentes (default "OTH")


def _days_between(d_lo, d_hi):
    """Dias entre dos fechas 'YYYY-MM-DD' (aritmetica de calendario simple)."""
    from datetime import date
    a = date(*map(int, d_lo.split("-")))
    b = date(*map(int, d_hi.split("-")))
    return (b - a).days


def _avg(seq):
    seq = list(seq)
    return sum(seq) / len(seq) if seq else None


def split_of(date, tournament):
    if TRAIN_DESDE <= date <= TRAIN_HASTA:
        return "train"
    if date >= TEST_DESDE and is_world_cup(tournament):
        return "test"
    return None


def featurize(rows):
    rows = sorted(rows, key=lambda r: r["date"])   # ya viene ordenado; por las dudas
    conf = infer_confederations(rows)

    form = defaultdict(lambda: deque(maxlen=FORM_N))   # team -> (gd, pts) recientes
    last_date = {}                                       # team -> fecha ultimo partido
    h2h = defaultdict(lambda: deque(maxlen=H2H_N))      # par ordenado -> ganador|'D'

    out = []
    for r in rows:
        date, home, away = r["date"], r["home_team"], r["away_team"]
        eh, ea = r["elo_home_pre"], r["elo_away_pre"]
        gh, ga = r["home_score"], r["away_score"]
        neutral = bool(r["neutral"])

        keep = split_of(date, r["tournament"])
        if keep is not None:
            # --- features (solo con info previa al partido) ---
            fh, fa = form[home], form[away]
            pair = tuple(sorted((home, away)))
            recent = h2h[pair]
            h2h_recent = sum(1 if w == home else (-1 if w == away else 0) for w in recent)

            label = "H" if gh > ga else ("A" if ga > gh else "D")
            out.append({
                "date": date, "home_team": home, "away_team": away,
                "split": keep, "label": label,
                "delta_elo": round(eh - ea, 2),
                "elo_home": eh, "elo_away": ea,
                "is_home": 0 if neutral else 1,
                "neutral": neutral,
                "form_gd_5_home": _avg(gd for gd, _ in fh),
                "form_gd_5_away": _avg(gd for gd, _ in fa),
                "form_pts_5_home": _avg(pts for _, pts in fh),
                "form_pts_5_away": _avg(pts for _, pts in fa),
                "rest_days_home": _days_between(last_date[home], date) if home in last_date else None,
                "rest_days_away": _days_between(last_date[away], date) if away in last_date else None,
                "tournament_weight": tournament_weight(r["tournament"]),
                "confed_home": conf.get(home, "OTH"),
                "confed_away": conf.get(away, "OTH"),
                "h2h_recent": h2h_recent,
            })

        # --- actualizar estado DESPUES de emitir (sin fuga) ---
        gd_h = gh - ga
        pts_h = 3 if gh > ga else (1 if gh == ga else 0)
        pts_a = 3 if ga > gh else (1 if ga == gh else 0)
        form[home].append((gd_h, pts_h))
        form[away].append((-gd_h, pts_a))
        last_date[home] = last_date[away] = date
        winner = home if gh > ga else (away if ga > gh else "D")
        h2h[tuple(sorted((home, away)))].append(winner)

    return out


def write_parquet(rows, path):
    if not rows:
        raise SystemExit("featurize: sin filas en el rango de modelado")
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    pq.write_table(pa.table(cols), path)


def main():
    rows = pq.read_table(SRC).to_pylist()
    ds = featurize(rows)
    write_parquet(ds, OUT)

    n = len(ds)
    n_train = sum(1 for r in ds if r["split"] == "train")
    n_test = sum(1 for r in ds if r["split"] == "test")
    print(f"data/ml_dataset.parquet  ({n} filas: {n_train} train, {n_test} test)")

    from collections import Counter
    for sp in ("train", "test"):
        sub = [r for r in ds if r["split"] == sp]
        bal = Counter(r["label"] for r in sub)
        pct = {k: f"{100*bal[k]/len(sub):.0f}%" for k in ("H", "D", "A")}
        print(f"  {sp}: balance H/D/A = {pct}")

    # nulos por feature (esperables al inicio de la serie; en train deberian ser ~0)
    feats = [k for k in ds[0] if k not in ("date", "home_team", "away_team",
                                           "split", "label", "neutral")]
    train = [r for r in ds if r["split"] == "train"]
    nul = {f: sum(1 for r in train if r[f] is None) for f in feats}
    nul = {f: c for f, c in nul.items() if c}
    print(f"  nulos en train: {nul if nul else 'ninguno'}")
    print(f"  confederaciones: {dict(Counter(r['confed_home'] for r in ds))}")


if __name__ == "__main__":
    main()
