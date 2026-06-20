#!/usr/bin/env python3
"""
Calibracion por Maxima Verosimilitud (MLE) del modelo de goles.

Ajusta una REGRESION DE POISSON de los goles marcados contra la diferencia de ELO
y un indicador de localia, sobre el historico de partidos internacionales:

    log E[goles] = b0 + b1 * dELO + b2 * local

equivalente al modelo del paper  lambda = mu * 10^(dELO/escala)  con
    mu     = exp(b0)            (goles esperados base, partido parejo y neutral)
    escala = ln(10) / b1        (cuantos puntos de ELO valen un factor 10 de goles)
    h      = b2 / b1            (ventaja de localia, EXPRESADA EN PUNTOS DE ELO)

Se usa el ELO actual (elo.csv) como proxy de fuerza, por lo que el ajuste se
restringe a partidos recientes (>= 2023) entre las selecciones con ELO conocido,
donde el proxy es razonable. Errores estandar por la informacion de Fisher.

Fuente del historico: https://github.com/martj42/international_results
Requiere numpy (uv sync --extra notebook). Escribe data/calibration.json.
"""
import csv
import io
import json
import os
import urllib.request

import numpy as np

import wcsim

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DESDE = "2023-01-01"

# Nombres del dataset (martj42) -> nombres de elo.csv
ALIAS = {
    "United States": "USA",
    "Turkey": "Turkiye",
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Republic of Ireland": "Ireland",
}


def cargar_historico(path_cache="/tmp/intl_results.csv"):
    if not os.path.exists(path_cache):
        print("Descargando historico de partidos...")
        urllib.request.urlretrieve(RESULTS_URL, path_cache)
    with open(path_cache, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def construir_dataset(filas, elo):
    """Formato largo: una fila por equipo-en-partido. Devuelve X, y."""
    X, y = [], []
    n_part = 0
    for r in filas:
        if r["date"] < DESDE:
            continue
        if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
            continue
        h = ALIAS.get(r["home_team"], r["home_team"])
        a = ALIAS.get(r["away_team"], r["away_team"])
        if h not in elo or a not in elo:
            continue
        try:
            gh, ga = int(r["home_score"]), int(r["away_score"])
        except ValueError:
            continue
        neutral = r["neutral"].strip().upper() == "TRUE"
        d = elo[h] - elo[a]
        local_h = 0.0 if neutral else 1.0
        # fila del equipo local
        X.append([1.0, d, local_h]); y.append(gh)
        # fila del equipo visitante
        X.append([1.0, -d, 0.0]); y.append(ga)
        n_part += 1
    return np.array(X), np.array(y, dtype=float), n_part


def poisson_irls(X, y, iters=50, tol=1e-10):
    """MLE de regresion de Poisson por Newton-Raphson / IRLS."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3))
    for _ in range(iters):
        eta = X @ beta
        lam = np.exp(np.clip(eta, -20, 20))
        grad = X.T @ (y - lam)               # gradiente del log-likelihood
        H = X.T @ (X * lam[:, None])         # -Hessiano = X' diag(lam) X
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    cov = np.linalg.inv(X.T @ (X * np.exp(np.clip(X @ beta, -20, 20))[:, None]))
    se = np.sqrt(np.diag(cov))
    return beta, se


def loglik(X, y, beta):
    from math import lgamma
    lam = np.exp(np.clip(X @ beta, -20, 20))
    return float(np.sum(y * np.log(lam) - lam - np.array([lgamma(v + 1) for v in y])))


def main():
    elo = wcsim.load_all()["elo"]
    filas = cargar_historico()
    X, y, n_part = construir_dataset(filas, elo)
    print(f"Partidos usados: {n_part}  (>= {DESDE}, ambos equipos con ELO conocido)")
    print(f"Observaciones (equipo-partido): {len(y)}  | goles promedio: {y.mean():.3f}")

    beta, se = poisson_irls(X, y)
    b0, b1, b2 = beta
    ln10 = np.log(10.0)
    mu = float(np.exp(b0))
    escala = float(ln10 / b1)
    h_elo = float(b2 / b1)
    z = 1.959963985

    print("\n=== Coeficientes (MLE, regresion de Poisson) ===")
    nombres = ["b0 (intercept)", "b1 (dELO)", "b2 (local)"]
    for nm, bb, ss in zip(nombres, beta, se):
        print(f"  {nm:16} = {bb: .6f}  ± {z*ss:.6f}  (IC95)")
    print("\n=== Parametros del modelo derivados ===")
    print(f"  mu      = {mu:.3f}   (paper: 1.35 fijado a mano)")
    print(f"  escala  = {escala:.1f}   (paper: 800 fijado a mano)")
    print(f"  h (ELO) = {h_elo:.1f}   (paper: 60 fijado a mano)")

    out = {
        "fuente": RESULTS_URL,
        "desde": DESDE,
        "n_partidos": n_part,
        "n_obs": int(len(y)),
        "beta": {"b0": float(b0), "b1": float(b1), "b2": float(b2)},
        "se": {"b0": float(se[0]), "b1": float(se[1]), "b2": float(se[2])},
        "mu": mu, "escala": escala, "home_adv_elo": h_elo,
        "loglik": loglik(X, y, beta),
    }
    dpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nEscrito {dpath}")


if __name__ == "__main__":
    main()
