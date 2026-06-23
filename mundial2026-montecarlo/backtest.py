"""Backtest vuelta a vuelta del modelo Poisson(ELO) sobre los partidos jugados.

Lee los partidos de data/played.parquet (fuente editable: data/sources/played.csv)
y para cada uno computa la probabilidad analitica W/D/L con el mismo modelo del
motor, comparandola con el resultado real. Reporta acierto, Brier (3-via) y
log-loss agrupados por vuelta, y opcionalmente genera una figura de calibracion.

  uv run python backtest.py                       # baseline (a mano)
  uv run python backtest.py --calibrated          # MLE (Apendice B)
  uv run python backtest.py --detail              # detalle partido por partido
  uv run --extra notebook python backtest.py --plot  # figura de calibracion
"""

import argparse
import json
import math
import os
from collections import defaultdict

from wcsim import MatchModel, _data_dir, load_elo, read_parquet


def load_played(path):
    return [(r["date"], r["group"], r["home"], r["away"], int(r["gh"]), int(r["ga"]))
            for r in read_parquet(path)]


def wdl(model, home, away, maxg=16):
    """P(home), P(empate), P(away) analitico bajo dos Poisson(la), Poisson(lb)."""
    la, lb = model.lambdas(home, away)
    pa = [math.exp(-la) * la ** k / math.factorial(k) for k in range(maxg)]
    pb = [math.exp(-lb) * lb ** k / math.factorial(k) for k in range(maxg)]
    p_h = sum(pa[i] * pb[j] for i in range(maxg) for j in range(maxg) if i > j)
    p_d = sum(pa[i] * pb[i] for i in range(maxg))
    p_a = sum(pa[i] * pb[j] for i in range(maxg) for j in range(maxg) if i < j)
    return p_h, p_d, p_a


def matchday(group, date):
    if group in "ABCD":
        return "M1" if date <= "2026-06-13" else "M2"
    return "M1" if date <= "2026-06-17" else "M2"


def build_model(calibrated):
    elo = load_elo(os.path.join(_data_dir(), "elo.parquet"))
    if calibrated:
        with open(os.path.join(_data_dir(), "calibration.json")) as f:
            cal = json.load(f)
        return MatchModel(elo, base=cal["mu"], home_adv=cal["home_adv_elo"], scale=cal["escala"])
    return MatchModel(elo, base=1.35, home_adv=60.0, scale=800.0)


def evaluate(rows_played, model):
    rows = []
    for date, g, home, away, gh, gg in rows_played:
        p_h, p_d, p_a = wdl(model, home, away)
        if gh > gg:
            outcome, p_actual, obs = "H", p_h, (1, 0, 0)
        elif gg > gh:
            outcome, p_actual, obs = "A", p_a, (0, 0, 1)
        else:
            outcome, p_actual, obs = "D", p_d, (0, 1, 0)
        brier = sum((p - o) ** 2 for p, o in zip((p_h, p_d, p_a), obs))
        logl = -math.log(max(p_actual, 1e-9))
        pred = max(["H", "D", "A"], key=lambda x: {"H": p_h, "D": p_d, "A": p_a}[x])
        rows.append({
            "date": date, "g": g, "home": home, "away": away,
            "gh": gh, "gg": gg, "outcome": outcome,
            "p_h": p_h, "p_d": p_d, "p_a": p_a,
            "pred": pred, "correct": pred == outcome,
            "brier": brier, "logl": logl, "md": matchday(g, date),
        })
    return rows


def calibration_bins(rows, n_bins=5):
    """Apila las 3*N predicciones (H/D/A) en n_bins; devuelve (centro, prob_media, freq_real, n_en_bin)."""
    pairs = []  # (probabilidad predicha, fue 1?)
    for r in rows:
        for prob, label in [(r["p_h"], "H"), (r["p_d"], "D"), (r["p_a"], "A")]:
            pairs.append((prob, int(r["outcome"] == label)))
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    out = []
    for i, b in enumerate(bins):
        if not b:
            continue
        mean_p = sum(p for p, _ in b) / len(b)
        freq = sum(y for _, y in b) / len(b)
        out.append((0.5 * (edges[i] + edges[i + 1]), mean_p, freq, len(b)))
    return out


def plot_calibration(rows_baseline, rows_calibrated, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibracion perfecta")

    for tag, rows, color in [("baseline (a mano)", rows_baseline, "C0"),
                             ("calibrado (MLE)", rows_calibrated, "C1")]:
        bins = calibration_bins(rows, n_bins=5)
        if not bins:
            continue
        xs = [mp for _c, mp, _f, _n in bins]
        ys = [fr for _c, _mp, fr, _n in bins]
        ns = [n for _c, _mp, _f, n in bins]
        sizes = [max(40, 6 * n) for n in ns]
        ax.scatter(xs, ys, s=sizes, alpha=0.55, color=color, edgecolor="black",
                   linewidth=0.5, label=tag)
        ax.plot(xs, ys, color=color, alpha=0.6, lw=1.2)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad predicha (media del bin)")
    ax.set_ylabel("Frecuencia observada en el bin")
    ax.set_title("Calibracion del modelo sobre partidos jugados\n"
                 "(132 pares H/D/A; tamano = #predicciones en el bin)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura -> {out_path}")


def summary(rows, label):
    print(f"\n== {label} ==")
    blocks = [
        ("M1 grupos A-D",  lambda r: r["md"] == "M1" and r["g"] in "ABCD"),
        ("M1 grupos E-L",  lambda r: r["md"] == "M1" and r["g"] in "EFGHIJKL"),
        ("M2 grupos A-D",  lambda r: r["md"] == "M2" and r["g"] in "ABCD"),
        ("M2 grupos E-J",  lambda r: r["md"] == "M2" and r["g"] in "EFGHIJ"),
        ("TODOS",          lambda r: True),
    ]
    print(f"{'bloque':<20} {'N':>4} {'acierto':>9} {'Brier':>8} {'logLoss':>9}")
    print("-" * 60)
    for name, pred in blocks:
        sub = [r for r in rows if pred(r)]
        if not sub:
            continue
        acc = sum(r["correct"] for r in sub) / len(sub)
        brier = sum(r["brier"] for r in sub) / len(sub)
        logl = sum(r["logl"] for r in sub) / len(sub)
        print(f"{name:<20} {len(sub):>4} {100*acc:>8.1f}% {brier:>8.3f} {logl:>9.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated", action="store_true",
                        help="Usa solo el modelo calibrado por MLE (default: imprime ambos)")
    parser.add_argument("--detail", action="store_true", help="Imprime cada partido (baseline)")
    parser.add_argument("--plot", action="store_true",
                        help="Genera charts/13_backtest_calibracion.png (requiere matplotlib)")
    parser.add_argument("--out", default=None,
                        help="Volcado de predicciones (CSV/Parquet); por defecto no escribe")
    args = parser.parse_args()

    played = load_played(os.path.join(_data_dir(), "played.parquet"))
    model_b = build_model(False)
    model_c = build_model(True)
    rows_b = evaluate(played, model_b)
    rows_c = evaluate(played, model_c)

    if args.detail:
        print("Detalle (modelo BASELINE):")
        print(f"{'fecha':<11} {'g':<2} {'partido':<38} {'real':<5} {'P(H/D/A) %':<15} {'pred':<6} {'brier':<7} {'logL':<6}")
        print("-" * 100)
        for r in rows_b:
            match = f"{r['home']} {r['gh']}-{r['gg']} {r['away']}"[:38]
            probs = f"{100*r['p_h']:3.0f}/{100*r['p_d']:3.0f}/{100*r['p_a']:3.0f}"
            tick = "OK" if r["correct"] else ".."
            print(f"{r['date']:<11} {r['g']:<2} {match:<38} {r['outcome']:<5} {probs:<15} {tick} {r['pred']:<2}   {r['brier']:.3f}  {r['logl']:.3f}")

    if not args.calibrated:
        summary(rows_b, "BASELINE (a mano) -- base=1.35 scale=800 h=60")
    summary(rows_c, "CALIBRADO (MLE) -- base=1.21 scale=1281 h=86.7")

    # Referencias
    n = len(rows_b)
    fH = sum(1 for r in rows_b if r["outcome"] == "H") / n
    fD = sum(1 for r in rows_b if r["outcome"] == "D") / n
    fA = sum(1 for r in rows_b if r["outcome"] == "A") / n
    brier_unif = 2/3
    brier_freq = sum(fz * ((1 - fz) ** 2 + sum(fy ** 2 for fy in (fH, fD, fA) if fy != fz))
                     for fz in (fH, fD, fA))
    logl_unif = math.log(3.0)
    logl_freq = -sum(fz * math.log(max(fz, 1e-9)) for fz in (fH, fD, fA))
    print("\nReferencias")
    print(f"  Frecuencia empirica       H={100*fH:.0f}%  D={100*fD:.0f}%  A={100*fA:.0f}%")
    print(f"  Uniforme (1/3,1/3,1/3)    Brier={brier_unif:.3f}  logLoss={logl_unif:.3f}")
    print(f"  Frecuencia base           Brier={brier_freq:.3f}  logLoss={logl_freq:.3f}")

    # Top y bottom (baseline)
    rs = sorted(rows_b, key=lambda r: r["logl"])
    print("\nTop-5 mejores predicciones (baseline):")
    for r in rs[:5]:
        probs = f"{100*r['p_h']:.0f}/{100*r['p_d']:.0f}/{100*r['p_a']:.0f}"
        print(f"  {r['home']} {r['gh']}-{r['gg']} {r['away']:<22} real={r['outcome']} P(H/D/A)={probs}  logL={r['logl']:.2f}")
    print("\nTop-5 sorpresas (baseline):")
    for r in rs[-5:][::-1]:
        probs = f"{100*r['p_h']:.0f}/{100*r['p_d']:.0f}/{100*r['p_a']:.0f}"
        print(f"  {r['home']} {r['gh']}-{r['gg']} {r['away']:<22} real={r['outcome']} P(H/D/A)={probs}  logL={r['logl']:.2f}")

    if args.plot:
        charts = os.path.join(os.path.dirname(_data_dir()), "charts")
        os.makedirs(charts, exist_ok=True)
        plot_calibration(rows_b, rows_c, os.path.join(charts, "13_backtest_calibracion.png"))

    if args.out:
        import pandas as pd
        df = pd.DataFrame(rows_b)
        df["modelo"] = "baseline"
        df_c = pd.DataFrame(rows_c)
        df_c["modelo"] = "calibrado"
        df_full = pd.concat([df, df_c], ignore_index=True)
        if args.out.endswith(".csv"):
            df_full.to_csv(args.out, index=False)
        else:
            df_full.to_parquet(args.out, index=False)
        print(f"Predicciones -> {args.out}")


if __name__ == "__main__":
    main()
