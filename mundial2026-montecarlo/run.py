#!/usr/bin/env python3
"""
CLI de la simulacion Monte Carlo del Mundial 2026.

Uso:
    python3 run.py                  # 20.000 simulaciones
    python3 run.py -n 100000        # mas simulaciones (mas preciso)
    python3 run.py -n 50000 --seed 42 --out resultados.csv
"""

import argparse
import csv
import sys

import wcsim


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo Mundial 2026 (ELO + tablas actuales)")
    ap.add_argument("-n", "--num", type=int, default=20000, help="cantidad de simulaciones")
    ap.add_argument("--seed", type=int, default=None, help="semilla del RNG (reproducibilidad)")
    ap.add_argument("--base", type=float, default=1.35, help="goles esperados base por equipo")
    ap.add_argument("--home-adv", type=float, default=60.0, help="bonus de ELO por localia (sedes)")
    ap.add_argument("--out", type=str, default=None, help="ruta CSV para volcar todos los resultados")
    ap.add_argument("--top", type=int, default=20, help="cuantas selecciones mostrar en pantalla")
    args = ap.parse_args()

    print(f"Simulando {args.num:,} escenarios del Mundial 2026 "
          f"(seed={args.seed}, base={args.base}, localia=+{args.home_adv} ELO)...")
    results, n = wcsim.run(
        n=args.num, seed=args.seed, base=args.base, home_adv=args.home_adv,
        progress=max(args.num // 10, 1),
    )

    champs = [r for r in results if r["titles"] > 0]
    print(f"\nResultado sobre {n:,} torneos simulados — "
          f"{len(champs)} selecciones salieron campeonas al menos una vez.\n")
    print(f"{'#':>2}  {'Seleccion':<24}{'ELO':>7}  {'Titulos':>9}  "
          f"{'P(campeon)':>11}  {'P(final)':>9}  {'P(semi)':>8}")
    print("-" * 80)
    for i, r in enumerate(results[:args.top], 1):
        print(f"{i:>2}  {r['team']:<24}{r['elo']:>7.0f}  {r['titles']:>9,}  "
              f"{r['p_champion']*100:>10.2f}%  {r['p_final']*100:>8.1f}%  {r['p_semi']*100:>7.1f}%")

    tail = [r for r in results[args.top:] if r["titles"] > 0]
    if tail:
        print("\nResto que ganó algún escenario:")
        for r in tail:
            print(f"    {r['team']:<24}{r['titles']:>8,}  {r['p_champion']*100:>7.3f}%")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["team", "elo", "titles", "p_champion",
                                              "finals", "p_final", "semis", "p_semi"])
            w.writeheader()
            w.writerows(results)
        print(f"\nResultados completos escritos en {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
