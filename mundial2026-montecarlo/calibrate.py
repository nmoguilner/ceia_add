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

Lee data/history.parquet (generado por convert_to_parquet.py; fuente martj42).
Requiere numpy + pyarrow (uv sync --extra notebook). Escribe data/calibration.json.
"""
import json
import os

import numpy as np

import wcsim

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


def cargar_historico():
    here = os.path.dirname(os.path.abspath(__file__))
    return wcsim.read_parquet(os.path.join(here, "data", "history.parquet"))


def construir_dataset(filas, elo):
    """Formato largo: una fila por equipo-en-partido. Devuelve X, y."""
    X, y = [], []
    n_part = 0
    for r in filas:
        if r["date"] < DESDE:
            continue
        if r["home_score"] is None or r["away_score"] is None:
            continue
        h = ALIAS.get(r["home_team"], r["home_team"])
        a = ALIAS.get(r["away_team"], r["away_team"])
        if h not in elo or a not in elo:
            continue
        gh, ga = int(r["home_score"]), int(r["away_score"])
        neutral = bool(r["neutral"])
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


def construir_pares(filas, elo, beta):
    """(la, lb, gh, ga) por partido con los lambda del modelo ya ajustado.
    Necesario para Dixon-Coles, que depende del marcador CONJUNTO (no separable)."""
    b0, b1, b2 = beta
    pares = []
    for r in filas:
        if r["date"] < DESDE:
            continue
        if r["home_score"] is None or r["away_score"] is None:
            continue
        h = ALIAS.get(r["home_team"], r["home_team"])
        a = ALIAS.get(r["away_team"], r["away_team"])
        if h not in elo or a not in elo:
            continue
        gh, ga = int(r["home_score"]), int(r["away_score"])
        d = elo[h] - elo[a]
        local_h = 0.0 if bool(r["neutral"]) else 1.0
        la = float(np.exp(b0 + b1 * d + b2 * local_h))
        lb = float(np.exp(b0 - b1 * d))
        pares.append((la, lb, gh, ga))
    return pares


def _tau(x, y, la, lb, rho):
    if x == 0 and y == 0:
        return 1.0 - la * lb * rho
    if x == 0 and y == 1:
        return 1.0 + la * rho
    if x == 1 and y == 0:
        return 1.0 + lb * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def loglik_rho(pares, rho):
    """Log-verosimilitud que aporta rho: solo el termino tau (lo demas es cte).
    Maximizarla equivale al estimador Dixon-Coles de 2 etapas (lambda fijos)."""
    s = 0.0
    for la, lb, gh, ga in pares:
        t = _tau(gh, ga, la, lb, rho)
        if t <= 0.0:
            return -1e18
        s += np.log(t)
    return s


def estimar_rho(pares, lo=-0.4, hi=0.4):
    """Maximiza loglik_rho por busqueda de la seccion aurea en [lo, hi]."""
    import math
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = loglik_rho(pares, c), loglik_rho(pares, d)
    for _ in range(200):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo); fc = loglik_rho(pares, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo); fd = loglik_rho(pares, d)
        if hi - lo < 1e-7:
            break
    rho = 0.5 * (lo + hi)
    return rho, loglik_rho(pares, rho)


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

    # --- Segunda etapa: dependencia de marcadores bajos (Dixon-Coles) ---
    pares = construir_pares(filas, elo, beta)
    rho, gain = estimar_rho(pares)
    print("\n=== Dixon-Coles (2da etapa, lambda fijos) ===")
    print(f"  rho     = {rho:+.4f}   (rho<0 => sube la masa de empates)")
    print(f"  ganancia logLik sobre el historico (n={len(pares)}): +{gain:.1f}")

    out = {
        "rho": rho,
        "rho_loglik_gain": gain,
        "fuente": "data/history.parquet (martj42/international_results)",
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
