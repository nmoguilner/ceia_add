"""Pronostico condicionado al estado REAL del torneo (eliminatorias en curso).

Fija los resultados ya disputados (data/knockout_played.parquet, M73-M100) y
computa las probabilidades de los partidos que faltan (semis M101/M102 y final
M103) con el mismo modelo del motor. Como quedan solo 3 partidos, P(campeon) se
calcula en forma ANALITICA (exacta bajo el modelo):

  P(avanza A vs B) = P(gana en 90') + P(empate) * P(gana definicion | ELO)

con la matriz de marcadores de wcsim (Dixon-Coles si rho != 0). Se reporta:

  - baseline (a mano):        base=1.35, escala=800,  h=60,   rho=0
  - calibrado (MLE, Ap. B):   mu/escala/h/rho de data/calibration.json
  - calibrado + ELO dinamico: idem, con el ELO evolucionado partido a partido
    por los 100 resultados reales del torneo (World Football Elo, K=60)

Tambien evalua la calidad predictiva sobre las eliminatorias ya jugadas
(P(avanzar) del ganador real: acierto, Brier 2-via y log-loss).

  uv run python pronostico.py
"""

import json
import os

from wcsim import (HOSTS, HOST_COUNTRY, MatchModel, _data_dir, load_bracket,
                   load_elo, read_parquet)


def load_knockout(path):
    return sorted(read_parquet(path), key=lambda r: (r["date"], r["match"]))


def advance_prob(model, a, b, venue=None):
    """P(a elimina a b): gana en 90'/120' o empata y gana la definicion."""
    M = model.score_matrix(a, b, venue)
    n = len(M)
    p_win = sum(M[i][j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(M[i][i] for i in range(n))
    return p_win + p_draw * model.win_prob(a, b, venue)


def elo_dinamico(elo0, played, knockout, k=60.0):
    """ELO evolucionado por los resultados reales (grupos + eliminatorias).

    Para grupos (venue=None) update_elo solo bonifica al 'home' si es
    anfitrion: si el anfitrion figura como visitante se invierte el orden
    (la formula es simetrica), para no perderle la localia."""
    model = MatchModel(dict(elo0))
    rows = ([(r["date"], r["home"], r["away"], int(r["gh"]), int(r["ga"]), None)
             for r in played] +
            [(r["date"], r["home"], r["away"], int(r["gh"]), int(r["ga"]), r["venue"])
             for r in knockout])
    for _date, h, a, gh, ga, venue in sorted(rows, key=lambda r: r[0]):
        if venue is None and a in HOSTS and h not in HOSTS:
            h, a, gh, ga = a, h, ga, gh
        model.update_elo(h, a, gh, ga, venue, k=k)
    return model.elo


def bracket_restante(bracket, knockout):
    """Resuelve el arbol con los ganadores reales; devuelve los partidos que
    faltan como (match, venue, equipo_a, equipo_b) ya resueltos o simbolicos."""
    winner = {r["match"]: r["winner"] for r in knockout}
    pend = []
    for node in bracket["tree"]:
        m = node["match"]
        if m in winner:
            continue
        a = winner.get(node["a"], f"W{node['a']}")
        b = winner.get(node["b"], f"W{node['b']}")
        pend.append((m, node.get("venue"), a, b))
    return pend


def forecast_final4(model, bracket, knockout):
    """P(gana semi / llega a final / campeon) para los 4 semifinalistas.

    Asume que faltan exactamente las dos semis y la final (estado al 13-jul)."""
    winner = {r["match"]: r["winner"] for r in knockout}
    tree = {n["match"]: n for n in bracket["tree"]}
    final = tree[max(tree)]                       # M103
    sf1, sf2 = tree[final["a"]], tree[final["b"]]
    v_f = final.get("venue")

    out = {}
    for sf, other in ((sf1, sf2), (sf2, sf1)):
        a, b = winner[sf["a"]], winner[sf["b"]]
        oa, ob = winner[other["a"]], winner[other["b"]]
        v = sf.get("venue")
        p_a = advance_prob(model, a, b, v)
        p_oa = advance_prob(model, oa, ob, other.get("venue"))
        for team, p_semi in ((a, p_a), (b, 1.0 - p_a)):
            p_champ = p_semi * (p_oa * advance_prob(model, team, oa, v_f)
                                + (1.0 - p_oa) * advance_prob(model, team, ob, v_f))
            out[team] = {"rival_semi": b if team == a else a,
                         "p_final": p_semi, "p_champ": p_champ}
    return out


def forecast_final(model, bracket, knockout):
    """P(campeon) con las semis ya jugadas: solo falta la final (M103)."""
    winner = {r["match"]: r["winner"] for r in knockout}
    tree = {n["match"]: n for n in bracket["tree"]}
    final = tree[max(tree)]
    a, b = winner[final["a"]], winner[final["b"]]
    p_a = advance_prob(model, a, b, final.get("venue"))
    return {a: {"rival_semi": b, "p_final": 1.0, "p_champ": p_a},
            b: {"rival_semi": a, "p_final": 1.0, "p_champ": 1.0 - p_a}}


def eval_knockout(model, knockout, elo_seq=None):
    """Backtest 2-vias sobre eliminatorias jugadas: P(avanza) del lado 'home'.

    Con elo_seq (lista de dicts ELO pre-partido) se evalua con ELO dinamico."""
    rows = []
    for i, r in enumerate(knockout):
        if elo_seq is not None:
            model.elo = elo_seq[i]
        h, a, venue, w = r["home"], r["away"], r["venue"], r["winner"]
        p_h = advance_prob(model, h, a, venue)
        p_win = p_h if w == h else 1.0 - p_h
        rows.append({"match": r["match"], "stage": r["stage"], "home": h,
                     "away": a, "gh": r["gh"], "ga": r["ga"],
                     "pen": r["pen_h"] is not None, "winner": w,
                     "p_h": p_h, "p_winner": p_win,
                     "correct": p_win >= 0.5})
    return rows


def elo_seq_prematch(elo0, played, knockout, k=60.0):
    """ELO vigente ANTES de cada partido de eliminatorias (orden cronologico)."""
    model = MatchModel(dict(elo0))
    group_rows = sorted(
        [(r["date"], r["home"], r["away"], int(r["gh"]), int(r["ga"])) for r in played])
    for _d, h, a, gh, ga in group_rows:
        if a in HOSTS and h not in HOSTS:
            h, a, gh, ga = a, h, ga, gh
        model.update_elo(h, a, gh, ga, None, k=k)
    seq = []
    for r in knockout:
        seq.append(dict(model.elo))
        model.update_elo(r["home"], r["away"], int(r["gh"]), int(r["ga"]),
                         r["venue"], k=k)
    return seq


def _print_forecast(tag, fc, elo):
    print(f"\n== {tag} ==")
    print(f"{'seleccion':<12} {'ELO':>6} {'rival':<12} {'P(final)':>9} {'P(campeon)':>11}")
    print("-" * 55)
    for team, d in sorted(fc.items(), key=lambda kv: -kv[1]["p_champ"]):
        print(f"{team:<12} {elo.get(team, 0):>6.0f} {d['rival_semi']:<12} "
              f"{100*d['p_final']:>8.1f}% {100*d['p_champ']:>10.1f}%")


def _print_backtest(tag, rows):
    import math
    acc = sum(r["correct"] for r in rows) / len(rows)
    # Brier 2-vias por lado: con prob p para el ganador real, el partido aporta
    # (1-p)^2 + ((1-p)-0)^2 = 2(1-p)^2; promediado por lado queda (1-p)^2.
    brier = sum((1.0 - r["p_winner"]) ** 2 for r in rows) / len(rows)
    logl = sum(-math.log(max(r["p_winner"], 1e-9)) for r in rows) / len(rows)
    print(f"{tag:<28} N={len(rows):>3}  acierto={100*acc:5.1f}%  "
          f"Brier={brier:.3f}  logLoss={logl:.3f}")


def main():
    d = _data_dir()
    elo0 = load_elo(os.path.join(d, "elo.parquet"))
    bracket = load_bracket(os.path.join(d, "bracket.json"))
    knockout = load_knockout(os.path.join(d, "knockout_played.parquet"))
    played = read_parquet(os.path.join(d, "played.parquet"))
    with open(os.path.join(d, "calibration.json")) as f:
        cal = json.load(f)

    pend = bracket_restante(bracket, knockout)
    print(f"Estado real del torneo ({len(knockout)} eliminatorias jugadas). Falta jugar:")
    for m, venue, a, b in pend:
        print(f"  M{m} ({venue}): {a} vs {b}")
    solo_final = len(pend) == 1

    variantes = [
        ("BASELINE (a mano)  base=1.35 escala=800 h=60",
         MatchModel(dict(elo0)), elo0),
        (f"CALIBRADO (MLE)  mu={cal['mu']:.2f} escala={cal['escala']:.0f} "
         f"h={cal['home_adv_elo']:.0f} rho={cal['rho']:.3f}",
         MatchModel(dict(elo0), base=cal["mu"], home_adv=cal["home_adv_elo"],
                    scale=cal["escala"], rho=cal["rho"]), elo0),
    ]
    elo_dyn = elo_dinamico(elo0, played, knockout)
    variantes.append((
        "CALIBRADO + ELO DINAMICO (100 partidos reales, K=60)",
        MatchModel(dict(elo_dyn), base=cal["mu"], home_adv=cal["home_adv_elo"],
                   scale=cal["escala"], rho=cal["rho"]), elo_dyn))

    for tag, model, elo in variantes:
        fc = (forecast_final if solo_final else forecast_final4)(model, bracket, knockout)
        _print_forecast(tag, fc, elo)

    print("\nELO dinamico vs inicial (los 4 semifinalistas):")
    for t in sorted(elo_dyn, key=lambda t: -elo_dyn[t])[:8]:
        print(f"  {t:<12} {elo0.get(t, 0):>6.0f} -> {elo_dyn[t]:>6.0f}  "
              f"({elo_dyn[t] - elo0.get(t, 0):+.0f})")

    print("\n== Backtest eliminatorias (P(avanzar) del ganador real, 2-vias) ==")
    m_b = MatchModel(dict(elo0))
    m_c = MatchModel(dict(elo0), base=cal["mu"], home_adv=cal["home_adv_elo"],
                     scale=cal["escala"], rho=cal["rho"])
    rows_b = eval_knockout(m_b, knockout)
    rows_c = eval_knockout(m_c, knockout)
    seq = elo_seq_prematch(elo0, played, knockout)
    m_d = MatchModel(dict(elo0), base=cal["mu"], home_adv=cal["home_adv_elo"],
                     scale=cal["escala"], rho=cal["rho"])
    rows_d = eval_knockout(m_d, knockout, elo_seq=seq)
    _print_backtest("baseline (a mano)", rows_b)
    _print_backtest("calibrado (MLE)", rows_c)
    _print_backtest("calibrado + ELO dinamico", rows_d)

    print("\nDetalle (calibrado + ELO dinamico):")
    print(f"{'M':>3} {'ronda':<4} {'partido':<42} {'gano':<12} {'P(pre)':>7} ")
    print("-" * 75)
    for r in rows_d:
        res = f"{r['home']} {r['gh']}-{r['ga']} {r['away']}" + (" (p)" if r["pen"] else "")
        tick = "OK" if r["correct"] else ".."
        print(f"{r['match']:>3} {r['stage']:<4} {res:<42} {r['winner']:<12} "
              f"{100*r['p_winner']:>6.1f}% {tick}")

    surpresas = sorted(rows_d, key=lambda r: r["p_winner"])[:5]
    print("\nTop-5 sorpresas de la eliminatoria (modelo dinamico):")
    for r in surpresas:
        print(f"  M{r['match']} {r['stage']}: {r['winner']} elimino a "
              f"{r['away'] if r['winner'] == r['home'] else r['home']} "
              f"con P(pre)={100*r['p_winner']:.1f}%")


if __name__ == "__main__":
    main()
