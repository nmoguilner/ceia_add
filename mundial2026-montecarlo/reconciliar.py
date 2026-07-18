#!/usr/bin/env python3
"""Reconcilia groups.csv y fixtures.csv a partir del log played.csv.

- groups.csv se RECONSTRUYE sumando todos los partidos de played.csv (played,
  pts, gf, ga por equipo) -> queda siempre consistente con el log.
- fixtures.csv conserva solo los partidos que NO estan en played.csv (los que
  el simulador todavia tiene que jugar).
Usa solo csv de stdlib. El team->grupo se deriva de played + fixtures.
"""
import csv, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sources")


def read_csv(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


played = read_csv("played.csv")
fixtures = read_csv("fixtures.csv")

# team -> grupo (de cualquier partido donde aparezca)
group_of = {}
for r in played + fixtures:
    group_of[r["home"]] = r["group"]
    group_of[r["away"]] = r["group"]

# acumular standings desde el log
st = {t: {"group": g, "played": 0, "pts": 0, "gf": 0, "ga": 0} for t, g in group_of.items()}
for r in played:
    h, a, gh, ga = r["home"], r["away"], int(r["gh"]), int(r["ga"])
    for team, gf, gc in ((h, gh, ga), (a, ga, gh)):
        s = st[team]
        s["played"] += 1; s["gf"] += gf; s["ga"] += gc
        s["pts"] += 3 if gf > gc else (1 if gf == gc else 0)

# escribir groups.csv ordenado por grupo y posicion (pts, dg, gf)
rows = sorted(st.items(), key=lambda kv: (kv[1]["group"],
              -kv[1]["pts"], -(kv[1]["gf"] - kv[1]["ga"]), -kv[1]["gf"]))
with open(os.path.join(SRC, "groups.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["group", "team", "played", "pts", "gf", "ga"])
    for team, s in rows:
        w.writerow([s["group"], team, s["played"], s["pts"], s["gf"], s["ga"]])

# fixtures.csv: dejar solo lo no jugado
played_key = {(r["group"], r["home"], r["away"]) for r in played}
pendientes = [r for r in fixtures if (r["group"], r["home"], r["away"]) not in played_key]
with open(os.path.join(SRC, "fixtures.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "group", "home", "away"])
    for r in pendientes:
        w.writerow([r["date"], r["group"], r["home"], r["away"]])

# resumen
by_played = {}
for _t, s in rows:
    by_played.setdefault(s["played"], []).append(s["group"])
print("groups.csv reconstruido:", len(rows), "equipos")
for pj in sorted(by_played):
    grupos = sorted(set(by_played[pj]))
    print(f"  PJ={pj}: grupos {','.join(grupos)} ({len(by_played[pj])} equipos)")
print("fixtures.csv pendientes:", len(pendientes))
for r in pendientes:
    print(f"  {r['date']}  G{r['group']}  {r['home']} vs {r['away']}")
