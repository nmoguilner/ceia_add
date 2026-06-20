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
  data/elo_reconstructed.csv   ELO reconstruido de las 48 selecciones (pre-Mundial)
  data/calibration_prematch.json   calibracion MLE con ELO pre-partido (sin proxy)

Requiere numpy. Fuente del historico: github.com/martj42/international_results
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

import wcsim
from calibrate import cargar_historico, poisson_irls, loglik, ALIAS, DESDE

# Nombre en groups.csv (mi) -> nombre en el dataset (martj42)
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
    muestras = []  # (date, R_home_pre, R_away_pre, neutral, gh, ga)
    for r in filas:
        if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
            continue
        try:
            gh, ga = int(r["home_score"]), int(r["away_score"])
        except ValueError:
            continue
        h, a = r["home_team"], r["away_team"]
        neutral = r["neutral"].strip().upper() == "TRUE"
        rh, ra = R[h], R[a]
        muestras.append((r["date"], rh, ra, neutral, gh, ga))
        # actualizacion ELO
        dr = (rh + (0 if neutral else 100)) - ra
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        chg = k_factor(r["tournament"]) * goal_mult(abs(gh - ga)) * (w - we)
        R[h] = rh + chg
        R[a] = ra - chg
    return R, muestras


def main():
    elo_actual = wcsim.load_all()["elo"]
    filas = cargar_historico()
    R, muestras = reconstruir(filas)
    print(f"Procesados {len(muestras)} partidos jugados; {len(R)} selecciones con ELO reconstruido.")

    # --- 1) ELO reconstruido de las 48 selecciones (pre-Mundial) ---
    here = os.path.dirname(os.path.abspath(__file__))
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
    with open(os.path.join(here, "data", "elo_reconstructed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["team", "elo"]); w.writerows(rec_rows)
    if faltan:
        print("  OJO sin match en dataset:", faltan)
    rec = np.array([p[0] for p in corr_pairs]); cur = np.array([p[1] for p in corr_pairs])
    print(f"  Validacion vs worldfootballrankings: corr = {np.corrcoef(rec, cur)[0,1]:.3f}  (n={len(rec)})")

    # --- 2) Recalibracion MLE con ELO PRE-PARTIDO (sin proxy), ventana >= DESDE ---
    X, y, npart = [], [], 0
    for date, rh, ra, neutral, gh, ga in muestras:
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
    print("\nEscrito data/elo_reconstructed.csv y data/calibration_prematch.json")


if __name__ == "__main__":
    main()
