#!/usr/bin/env python3
"""Distribucion del rival de Argentina en Ronda de 32 + P(posicion de grupo).
Reusa el motor wcsim. Uso: uv run python _rival_arg.py -n 200000 --calibrated"""
import argparse, json, os, random, collections
import wcsim


def main():
    ap = argparse.ArgumentParser()
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
    r32 = data["bracket"]["r32"]

    arg_pos = collections.Counter()         # posicion de Argentina en grupo J
    rival = collections.Counter()           # rival en R32 (si Arg clasifica)
    h2 = collections.Counter()              # quien queda 2do en grupo H
    rival_given_pos = collections.defaultdict(collections.Counter)

    n = args.num
    for _ in range(n):
        standings, best_thirds = wcsim.simulate_group_stage(
            data["groups"], data["fixtures"], model, rng)
        third_qual = {team for (_g, team, _st) in best_thirds}
        third_slot = wcsim.assign_thirds(best_thirds, r32)

        # posicion de Argentina
        order_j = standings["J"]
        pos = order_j.index("Argentina")
        arg_pos[["1ro", "2do", "3ro", "4to"][pos]] += 1
        h2[standings["H"][1]] += 1   # 2do del grupo H

        # clasifica Argentina?
        clasif = pos <= 1 or (pos == 2 and "Argentina" in third_qual)
        if not clasif:
            rival["(Arg no clasifica)"] += 1
            continue
        # encontrar el partido de R32 donde juega Argentina y su rival
        rv = None
        for m in r32:
            a = wcsim.resolve_slot(m["a"], standings, third_slot, m["match"])
            b = wcsim.resolve_slot(m["b"], standings, third_slot, m["match"])
            if a == "Argentina":
                rv = b; break
            if b == "Argentina":
                rv = a; break
        rival[rv] += 1
        rival_given_pos[["1ro", "2do", "3ro"][pos]][rv] += 1

    print(f"\n== Argentina, posicion final en grupo J ({n:,} sims) ==")
    for k in ["1ro", "2do", "3ro", "4to"]:
        print(f"  {k}: {arg_pos[k]/n*100:5.1f}%")

    print(f"\n== Quien queda 2do en grupo H (= rival de Arg si Arg sale 1ro) ==")
    for t, c in h2.most_common():
        print(f"  {t:<14}{c/n*100:5.1f}%")

    print(f"\n== Rival de Argentina en Ronda de 32 ==")
    for t, c in rival.most_common():
        print(f"  {t:<18}{c/n*100:5.1f}%")

    es_uru = rival["Spain"] + rival["Uruguay"]
    print(f"\n  -> P(rival = Espana o Uruguay) = {es_uru/n*100:.1f}%")
    print(f"  -> P(rival = NI Espana NI Uruguay, clasificando) = "
          f"{(sum(rival.values())-es_uru-rival['(Arg no clasifica)'])/n*100:.1f}%")


if __name__ == "__main__":
    main()
