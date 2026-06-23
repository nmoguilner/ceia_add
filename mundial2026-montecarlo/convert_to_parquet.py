#!/usr/bin/env python3
"""
Genera/actualiza los datos del proyecto en formato Parquet.

- Descarga el historico de partidos (martj42) y lo guarda como data/history.parquet
  (columnas tipadas; scores como enteros nullables, neutral como bool).
- Convierte las fuentes CSV editables (elo, groups, fixtures) a Parquet.
- Convierte, si existen, resultados_1M y elo_reconstructed.

Las fuentes CSV se conservan como copia editable a mano (data/sources/*.csv): para
actualizar el torneo se edita el CSV y se vuelve a correr este script. Los Parquet
son el formato canonico que leen wcsim.py / calibrate.py / elo_history.py.

Requiere pandas + pyarrow (uv sync --extra notebook).
"""
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SRC = os.path.join(DATA, "sources")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def _write(df, name):
    path = os.path.join(DATA, name)
    df.to_parquet(path, index=False)
    print(f"  {name:28} {len(df):>6} filas")


def convertir_fuentes():
    # elo
    elo = pd.read_csv(os.path.join(SRC, "elo.csv"))
    elo["elo"] = elo["elo"].astype("float64")
    _write(elo, "elo.parquet")
    # groups
    g = pd.read_csv(os.path.join(SRC, "groups.csv"))
    for c in ["played", "pts", "gf", "ga"]:
        g[c] = g[c].astype("int64")
    _write(g, "groups.parquet")
    # fixtures
    _write(pd.read_csv(os.path.join(SRC, "fixtures.csv")), "fixtures.parquet")
    # played (partidos ya disputados, para backtest)
    played_csv = os.path.join(SRC, "played.csv")
    if os.path.exists(played_csv):
        p = pd.read_csv(played_csv)
        for c in ["gh", "ga"]:
            p[c] = p[c].astype("int64")
        _write(p, "played.parquet")


def convertir_historico():
    cache = "/tmp/intl_results.csv"
    if not os.path.exists(cache):
        print("Descargando historico...")
        urllib.request.urlretrieve(RESULTS_URL, cache)
    df = pd.read_csv(cache, dtype=str)
    df = df[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]].copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["neutral"] = df["neutral"].str.strip().str.upper().eq("TRUE")
    _write(df, "history.parquet")


def convertir_derivados():
    for csv_name, pq_name in [("../resultados_1M.csv", "../resultados_1M.parquet"),
                              ("elo_reconstructed.csv", "elo_reconstructed.parquet")]:
        src = os.path.join(DATA, csv_name)
        if os.path.exists(src):
            _write(pd.read_csv(src), pq_name)


def main():
    print("Generando Parquet en data/ ...")
    convertir_fuentes()
    convertir_historico()
    convertir_derivados()
    print("Listo.")


if __name__ == "__main__":
    main()
