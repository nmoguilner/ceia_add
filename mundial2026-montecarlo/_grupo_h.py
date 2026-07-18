#!/usr/bin/env python3
"""Probabilidad de avance de fase de grupos (P(1ro), P(2do), P(3ro clasif.),
P(clasifica)) para selecciones puntuales. Reusa el motor wcsim sin tocarlo.

Uso: uv run python _grupo_h.py -n 200000 --seed 42 --calibrated Uruguay Spain
"""
import argparse, json, os, random, collections
import wcsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*", default=["Uruguay", "Spain"])
    ap.add_argument("-n", "--num", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()

    base, home_adv, scale, rho = 1.35, 60.0, 800.0, 0.0
    if args.calibrated:
        here = os.path.dirname(os.path.abspath(__file__))
        cal = json.load(open(os.path.join(here, "data", "calibration.json"), encoding="utf-8"))
        base, scale, home_adv, rho = cal["mu"], cal["escala"], cal["home_adv_elo"], cal.get("rho", 0.0)

    data = wcsim.load_all()
    model = wcsim.MatchModel(data["elo"], base=base, home_adv=home_adv, scale=scale, rho=rho)
    rng = random.Random(args.seed)

    group_of = {t["team"]: g for g, ts in data["groups"].items() for t in ts}
    teams = args.teams or ["Uruguay", "Spain"]
    cnt = {t: collections.Counter() for t in teams}

    for _ in range(args.num):
        standings, best_thirds = wcsim.simulate_group_stage(
            data["groups"], data["fixtures"], model, rng)
        third_qual = {team for (_g, team, _st) in best_thirds}
        for t in teams:
            order = standings[group_of[t]]
            pos = order.index(t)
            if pos == 0:
                cnt[t]["1ro"] += 1; cnt[t]["clasifica"] += 1
            elif pos == 1:
                cnt[t]["2do"] += 1; cnt[t]["clasifica"] += 1
            elif pos == 2 and t in third_qual:
                cnt[t]["3ro_clasif"] += 1; cnt[t]["clasifica"] += 1
            else:
                cnt[t]["eliminado"] += 1

    n = args.num
    print(f"\nAvance de fase de grupos sobre {n:,} simulaciones "
          f"({'calibrado MLE' if args.calibrated else 'baseline'}):\n")
    print(f"{'Equipo':<12}{'Grupo':>6}{'1ro':>9}{'2do':>9}{'3ro✓':>9}{'CLASIFICA':>12}")
    print("-" * 57)
    for t in teams:
        c = cnt[t]
        print(f"{t:<12}{group_of[t]:>6}"
              f"{c['1ro']/n*100:>8.1f}%{c['2do']/n*100:>8.1f}%"
              f"{c['3ro_clasif']/n*100:>8.1f}%{c['clasifica']/n*100:>11.1f}%")


if __name__ == "__main__":
    main()
