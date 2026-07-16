#!/usr/bin/env python3
"""
Reconstruccion del ELO pre-partido historico y recalibracion SIN proxy.

Corre el algoritmo de los World Football Elo Ratings (eloratings.net) hacia
adelante sobre todo el historico de partidos internacionales:

    R' = R + K * G * (W - We)
    We = 1 / (1 + 10^(-(R_local + ventaja_local - R_visita)/400))
    K  : 60 Mundial, 50 finales continentales, 40 clasificatorias/Nations,
         30 otros torneos, 20 amistosos
    G  : 1 (margen<=1), 1.5 (margen=2), (11+margen)/8 (margen>=3)
    ventaja_local = 100 si no es cancha neutral, 0 si neutral

Con esto se obtiene, para CADA partido, el ELO de ambos equipos JUSTO ANTES de
jugarlo (sin el sesgo de usar el ELO actual como proxy, que atenuaba beta1 por
dilucion de regresion en el Apendice B). Luego se reajusta la regresion de
Poisson de los goles sobre esa diferencia de ELO pre-partido.

Salidas:
  data/elo_reconstructed.parquet  ELO reconstruido de las 48 selecciones (pre-Mundial)
  data/calibration_prematch.json   calibracion MLE con ELO pre-partido (sin proxy)

Requiere numpy. Fuente del historico: github.com/martj42/international_results
"""
import json
import os
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import wcsim
from calibrate import cargar_historico, poisson_irls, loglik, ALIAS, DESDE

# Nombre en groups.parquet (mi) -> nombre en el dataset (martj42)
MI_A_DATASET = {
    "USA": "United States",
    "Turkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Curacao": "Curaçao",
}


def k_factor(tournament):
    t = tournament.lower()
    if t == "friendly":
        return 20
    if "world cup" in t and "qual" not in t:
        return 60
    if "qualif" in t:
        return 40
    if any(s in t for s in ["uefa euro", "copa am", "african cup", "afc asian",
                            "gold cup", "nations cup", "confederations"]):
        return 50
    if "nations league" in t:
        return 40
    return 30


def goal_mult(margin):
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0


def reconstruir(filas):
    """Pasada forward. Devuelve (ratings_finales, muestras_prepartido)."""
    R = defaultdict(lambda: 1500.0)
    filas = sorted(filas, key=lambda r: r["date"])
    # (date, home, away, R_home_pre, R_away_pre, neutral, gh, ga, tournament)
    muestras = []
    for r in filas:
        if r["home_score"] is None or r["away_score"] is None:
            continue
        gh, ga = int(r["home_score"]), int(r["away_score"])
        h, a = r["home_team"], r["away_team"]
        neutral = bool(r["neutral"])
        rh, ra = R[h], R[a]
        muestras.append((r["date"], h, a, rh, ra, neutral, gh, ga, r["tournament"]))
        # actualizacion ELO
        dr = (rh + (0 if neutral else 100)) - ra
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        chg = k_factor(r["tournament"]) * goal_mult(abs(gh - ga)) * (w - we)
        R[h] = rh + chg
        R[a] = ra - chg
    return R, muestras


def escribir_prematch(muestras, path):
    """Vuelca la serie de ELO PRE-PARTIDO de cada partido jugado del historico
    (una fila por partido). Es la entrada del TP de Aprendizaje de Maquina:
    featurize.py la une con history.parquet (mismos nombres martj42) para armar
    la tabla de entrenamiento sin fuga temporal."""
    cols = {
        "date":         [m[0] for m in muestras],
        "home_team":    [m[1] for m in muestras],
        "away_team":    [m[2] for m in muestras],
        "elo_home_pre": [round(m[3], 2) for m in muestras],
        "elo_away_pre": [round(m[4], 2) for m in muestras],
        "neutral":      [bool(m[5]) for m in muestras],
        "home_score":   [int(m[6]) for m in muestras],
        "away_score":   [int(m[7]) for m in muestras],
        "tournament":   [m[8] for m in muestras],
    }
    pq.write_table(pa.table(cols), path)


def main():
    elo_actual = wcsim.load_all()["elo"]
    filas = cargar_historico()
    R, muestras = reconstruir(filas)
    print(f"Procesados {len(muestras)} partidos jugados; {len(R)} selecciones con ELO reconstruido.")

    # --- 1) ELO reconstruido de las 48 selecciones (pre-Mundial) ---
    here = os.path.dirname(os.path.abspath(__file__))

    # --- 1b) Serie de ELO pre-partido de TODO el historico (insumo del TP ML) ---
    prematch_path = os.path.join(here, "data", "elo_prematch.parquet")
    escribir_prematch(muestras, prematch_path)
    print(f"  Serie ELO pre-partido -> data/elo_prematch.parquet  ({len(muestras)} filas)")
    rec_rows, faltan = [], []
    corr_pairs = []
    for team in elo_actual:                       # las 48 del torneo
        dsname = MI_A_DATASET.get(team, team)
        if dsname in R:
            rec_rows.append((team, round(R[dsname], 1)))
            corr_pairs.append((R[dsname], elo_actual[team]))
        else:
            faltan.append((team, dsname))
    rec_rows.sort(key=lambda x: -x[1])
    pq.write_table(
        pa.table({"team": [t for t, _ in rec_rows], "elo": [e for _, e in rec_rows]}),
        os.path.join(here, "data", "elo_reconstructed.parquet"))
    if faltan:
        print("  OJO sin match en dataset:", faltan)
    rec = np.array([p[0] for p in corr_pairs]); cur = np.array([p[1] for p in corr_pairs])
    print(f"  Validacion vs worldfootballrankings: corr = {np.corrcoef(rec, cur)[0,1]:.3f}  (n={len(rec)})")

    # --- 2) Recalibracion MLE con ELO PRE-PARTIDO (sin proxy), ventana >= DESDE ---
    X, y, npart = [], [], 0
    for date, _h, _a, rh, ra, neutral, gh, ga, _tour in muestras:
        if date < DESDE:
            continue
        d = rh - ra
        local_h = 0.0 if neutral else 1.0
        X.append([1.0, d, local_h]); y.append(gh)
        X.append([1.0, -d, 0.0]);    y.append(ga)
        npart += 1
    X = np.array(X); y = np.array(y, dtype=float)
    beta, se = poisson_irls(X, y)
    b0, b1, b2 = beta
    ln10 = np.log(10.0)
    mu, escala, h_elo = float(np.exp(b0)), float(ln10 / b1), float(b2 / b1)
    z = 1.959963985
    print(f"\nRecalibracion con ELO pre-partido (>= {DESDE}, TODOS los equipos): {npart} partidos")
    print(f"  b1 (dELO) = {b1:.6f} ± {z*se[1]:.6f}")
    print(f"  mu      = {mu:.3f}   (proxy Apendice B: 1.21 | a mano: 1.35)")
    print(f"  escala  = {escala:.0f}   (proxy Apendice B: 1281 | a mano: 800)")
    print(f"  h (ELO) = {h_elo:.1f}   (proxy Apendice B: 86.7 | a mano: 60)")

    out = {
        "fuente": "reconstruccion ELO eloratings sobre martj42/international_results",
        "desde": DESDE, "n_partidos": npart, "n_obs": int(len(y)),
        "beta": {"b0": float(b0), "b1": float(b1), "b2": float(b2)},
        "se": {"b0": float(se[0]), "b1": float(se[1]), "b2": float(se[2])},
        "mu": mu, "escala": escala, "home_adv_elo": h_elo,
        "loglik": loglik(X, y, beta),
        "corr_vs_wfr": float(np.corrcoef(rec, cur)[0, 1]),
    }
    with open(os.path.join(here, "data", "calibration_prematch.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nEscrito data/elo_reconstructed.parquet y data/calibration_prematch.json")


if __name__ == "__main__":
    main()
