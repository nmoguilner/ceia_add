#!/usr/bin/env python3
"""
Comparacion A/B del torneo: Monte Carlo con motor ELO vs motor ML sobre el mismo
snapshot. Reporta P(campeon) de cada seleccion con ambos motores y, si existe
data/market.json, la cercania al mercado (MAE + Spearman) -> metrica de negocio.

  uv run --extra ml python evaluate.py -n 50000 --seed 42

Salida: data/ml_champions.json (P(campeon) ML vs ELO [vs mercado]).
"""
import argparse
import json
import os
import random

import wcsim
from mlmodel import MLMatchModel

HERE = os.path.dirname(os.path.abspath(__file__))


def run_mc(data, model, n, seed):
    rng = random.Random(seed)
    champ = {t: 0 for t in data["elo"]}
    for _ in range(n):
        standings, best_thirds = wcsim.simulate_group_stage(
            data["groups"], data["fixtures"], model, rng)
        champion, _f, _s = wcsim.simulate_knockout(
            standings, best_thirds, data["bracket"], model, rng)
        champ[champion] += 1
    return {t: c / n for t, c in champ.items()}


def elo_model(data):
    cal = json.load(open(os.path.join(HERE, "data", "calibration.json"), encoding="utf-8"))
    return wcsim.MatchModel(dict(data["elo"]), base=cal["mu"], scale=cal["escala"],
                            home_adv=cal["home_adv_elo"], rho=cal.get("rho", 0.0))


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    data = wcsim.load_all()
    teams = list(data["elo"].keys())

    print(f"Monte Carlo A/B: {args.num:,} torneos por motor (seed={args.seed})")
    print("  motor ELO (calibrado MLE + Dixon-Coles)...")
    p_elo = run_mc(data, elo_model(data), args.num, args.seed)
    print("  motor ML (clasificador + tabla precomputada)...")
    ml = MLMatchModel(dict(data["elo"])).precompute(teams)
    p_ml = run_mc(data, ml, args.num, args.seed)

    # referencia de mercado (opcional)
    market_path = os.path.join(HERE, "data", "market.json")
    market = json.load(open(market_path, encoding="utf-8")) if os.path.exists(market_path) else None

    order = sorted(teams, key=lambda t: -p_ml[t])
    print(f"\n{'#':>2}  {'Seleccion':<22}{'P(camp) ML':>11}{'P(camp) ELO':>12}" +
          ("{:>12}".format("Mercado") if market else ""))
    print("-" * (59 + (12 if market else 0)))
    for i, t in enumerate(order[:args.top], 1):
        line = f"{i:>2}  {t:<22}{p_ml[t]*100:>10.2f}%{p_elo[t]*100:>11.2f}%"
        if market:
            line += f"{market.get(t, 0.0)*100:>11.2f}%"
        print(line)

    out = {"n": args.num, "seed": args.seed,
           "p_champion_ml": p_ml, "p_champion_elo": p_elo}

    # metrica de negocio
    if market:
        common = [t for t in teams if t in market]
        for tag, p in [("ML", p_ml), ("ELO", p_elo)]:
            xs = [p[t] for t in common]
            ms = [market[t] for t in common]
            mae = sum(abs(a - b) for a, b in zip(xs, ms)) / len(common)
            rho = spearman(xs, ms)
            print(f"\nCercania al mercado ({tag}): MAE={mae*100:.2f} pts  Spearman={rho:.3f}")
            out.setdefault("market", {})[tag] = {"mae": mae, "spearman": rho}
    else:
        print("\n[H2] data/market.json ausente: falta snapshotear el vector de "
              "mercado (cuotas outright, misma fecha) para medir la metrica de negocio.")

    # concordancia ML vs ELO entre si
    xs = [p_ml[t] for t in teams]
    ys = [p_elo[t] for t in teams]
    mae = sum(abs(a - b) for a, b in zip(xs, ys)) / len(teams)
    print(f"\nML vs ELO: MAE={mae*100:.2f} pts  Spearman={spearman(xs, ys):.3f}")
    out["ml_vs_elo"] = {"mae": mae, "spearman": spearman(xs, ys)}

    json.dump(out, open(os.path.join(HERE, "data", "ml_champions.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nGuardado -> data/ml_champions.json")


if __name__ == "__main__":
    main()
