#!/usr/bin/env python3
"""
Descarga las formaciones (XI titular) de torneos internacionales recientes desde
StatsBomb Open Data y arma el dataset base para el modelo de formaciones (Fase 1).

Fuente: https://github.com/statsbomb/open-data  (gratis, con atribucion).
Para cada partido obtiene quien fue TITULAR en cada equipo (start_reason
'Starting XI'), su posicion y minutos. Salidas:

  data/sb_matches.parquet   un registro por partido (equipos, resultado, torneo)
  data/lineups.parquet      un registro por (partido, equipo, jugador titular)

Requiere pandas + pyarrow (uv sync --extra notebook).
"""
import json
import os
import urllib.request

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = "/tmp/statsbomb"
HERE = os.path.dirname(os.path.abspath(__file__))

# (nombre, competition_id, season_id) — torneos de selecciones recientes
TORNEOS = [
    ("FIFA World Cup 2022", 43, 106),
    ("FIFA World Cup 2018", 43, 3),
    ("UEFA Euro 2024", 55, 282),
    ("UEFA Euro 2020", 55, 43),
    ("Copa America 2024", 223, 282),
    ("African Cup of Nations 2023", 1267, 107),
]


def _get(url, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def starters(lineup_team):
    """Devuelve [(player_name, position, minutos)] de los titulares de un equipo."""
    out = []
    for p in lineup_team["lineup"]:
        pos = p.get("positions") or []
        start = next((x for x in pos if x.get("start_reason") == "Starting XI"), None)
        if start is None:
            continue
        name = p.get("player_nickname") or p["player_name"]
        out.append((name, start.get("position", ""), _mins(pos)))
    return out


def _mins(positions):
    total = 0
    for x in positions:
        try:
            fr = x["from"]; to = x.get("to") or "90:00"
            total += (_sec(to) - _sec(fr)) // 60
        except Exception:
            pass
    return total


def _sec(mmss):
    m, s = mmss.split(":")[:2]
    return int(m) * 60 + int(s)


def main():
    match_rows, lineup_rows = [], []
    for nombre, comp, season in TORNEOS:
        matches = _get(f"{BASE}/matches/{comp}/{season}.json",
                       f"{CACHE}/matches_{comp}_{season}.json")
        n_lu = 0
        for m in matches:
            mid = m["match_id"]
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            match_rows.append({
                "match_id": mid, "date": m["match_date"], "torneo": nombre,
                "home_team": home, "away_team": away,
                "home_score": m["home_score"], "away_score": m["away_score"],
            })
            try:
                lu = _get(f"{BASE}/lineups/{mid}.json", f"{CACHE}/lineup_{mid}.json")
            except Exception:
                continue
            for team in lu:
                tname = team["team_name"]
                for pname, pos, mins in starters(team):
                    lineup_rows.append({
                        "match_id": mid, "team": tname, "player": pname,
                        "position": pos, "minutes": mins,
                    })
            n_lu += 1
        print(f"  {nombre:32} {len(matches):>3} partidos, {n_lu:>3} con formacion")

    mdf = pd.DataFrame(match_rows)
    ldf = pd.DataFrame(lineup_rows)
    mdf.to_parquet(os.path.join(HERE, "data", "sb_matches.parquet"), index=False)
    ldf.to_parquet(os.path.join(HERE, "data", "lineups.parquet"), index=False)
    print(f"\nTotal: {len(mdf)} partidos, {len(ldf)} titulares "
          f"({ldf['player'].nunique()} jugadores distintos, {ldf['team'].nunique()} selecciones)")
    print("Escrito data/sb_matches.parquet y data/lineups.parquet")


if __name__ == "__main__":
    main()
