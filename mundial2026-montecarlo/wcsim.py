"""
Motor de simulacion Monte Carlo del Mundial 2026.

Modelo:
  - Cada partido se simula con dos Poisson independientes cuyas medias (goles
    esperados) salen de la diferencia de ELO entre los equipos.
  - Se arranca desde el estado ACTUAL de los 12 grupos (snapshot) y se simulan
    solo los partidos que faltan.
  - Se define la clasificacion (1ros, 2dos y 8 mejores 3ros), se arma la Ronda
    de 32 con la plantilla oficial y se juega la llave hasta la final.
  - En eliminacion directa, el empate se resuelve por penales con probabilidad
    derivada del ELO.

Solo usa la biblioteca estandar de Python (sin numpy/pandas), en linea con la
filosofia del proyecto montecarlo-calc.
"""

import csv
import json
import math
import os
import random

HOSTS = {"USA", "Mexico", "Canada"}  # sedes 2026: ventaja de localia

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _data_dir(base=None):
    base = base or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data")


def load_elo(path):
    elo = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            elo[row["team"]] = float(row["elo"])
    return elo


def load_groups(path):
    groups = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = row["group"]
            groups.setdefault(g, []).append({
                "team": row["team"],
                "played": int(row["played"]),
                "pts": int(row["pts"]),
                "gf": int(row["gf"]),
                "ga": int(row["ga"]),
            })
    return groups


def load_fixtures(path):
    fx = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fx.append((row["group"], row["home"], row["away"]))
    return fx


def load_bracket(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all(base=None):
    d = _data_dir(base)
    return {
        "elo": load_elo(os.path.join(d, "elo.csv")),
        "groups": load_groups(os.path.join(d, "groups.csv")),
        "fixtures": load_fixtures(os.path.join(d, "fixtures.csv")),
        "bracket": load_bracket(os.path.join(d, "bracket.json")),
    }


# ---------------------------------------------------------------------------
# Modelo de partido
# ---------------------------------------------------------------------------

class MatchModel:
    """Convierte ELO -> goles esperados -> resultado simulado."""

    def __init__(self, elo, base=1.35, home_adv=60.0, scale=800.0):
        self.elo = elo
        self.base = base          # goles esperados base por equipo
        self.home_adv = home_adv  # bonus de ELO para equipos anfitriones
        self.scale = scale        # 800 => relacion de lambdas = 10^(dELO/400)

    def _eff_elo(self, team):
        e = self.elo.get(team, 1450.0)
        return e + (self.home_adv if team in HOSTS else 0.0)

    def lambdas(self, a, b):
        diff = self._eff_elo(a) - self._eff_elo(b)
        la = self.base * 10.0 ** (diff / self.scale)
        lb = self.base * 10.0 ** (-diff / self.scale)
        return la, lb

    def win_prob(self, a, b):
        """Prob. de que a venza a b a 1 partido (sin empate), via ELO clasico."""
        diff = self._eff_elo(a) - self._eff_elo(b)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def play_group(self, a, b, rng):
        la, lb = self.lambdas(a, b)
        return _poisson(la, rng), _poisson(lb, rng)

    def play_knockout(self, a, b, rng):
        """Devuelve el equipo ganador (empate -> penales segun ELO)."""
        la, lb = self.lambdas(a, b)
        ga, gb = _poisson(la, rng), _poisson(lb, rng)
        if ga > gb:
            return a
        if gb > ga:
            return b
        return a if rng.random() < self.win_prob(a, b) else b


def _poisson(lam, rng):
    """Muestreo de Poisson (algoritmo de Knuth); lam chico (~1-3)."""
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


# ---------------------------------------------------------------------------
# Fase de grupos
# ---------------------------------------------------------------------------

def _rank_key(team, st, elo):
    # Criterios: puntos, diferencia de gol, goles a favor, ELO (desempate final)
    return (st["pts"], st["gf"] - st["ga"], st["gf"], elo.get(team, 1450.0))


def simulate_group_stage(groups, fixtures, model, rng):
    # Estado mutable por equipo (parte del snapshot actual)
    state = {}
    group_of = {}
    for g, teams in groups.items():
        for st in teams:
            state[st["team"]] = {"pts": st["pts"], "gf": st["gf"], "ga": st["ga"]}
            group_of[st["team"]] = g

    for g, home, away in fixtures:
        gh, ga = model.play_group(home, away, rng)
        sh, sa = state[home], state[away]
        sh["gf"] += gh; sh["ga"] += ga
        sa["gf"] += ga; sa["ga"] += gh
        if gh > ga:
            sh["pts"] += 3
        elif ga > gh:
            sa["pts"] += 3
        else:
            sh["pts"] += 1; sa["pts"] += 1

    # Posiciones finales por grupo
    standings = {}
    thirds = []
    for g, teams in groups.items():
        ordered = sorted(
            (t["team"] for t in teams),
            key=lambda tm: _rank_key(tm, state[tm], model.elo),
            reverse=True,
        )
        standings[g] = ordered
        third = ordered[2]
        thirds.append((g, third, state[third]))

    # 8 mejores terceros
    thirds.sort(key=lambda x: (x[2]["pts"], x[2]["gf"] - x[2]["ga"], x[2]["gf"],
                               model.elo.get(x[1], 1450.0)), reverse=True)
    best_thirds = thirds[:8]
    return standings, best_thirds


# ---------------------------------------------------------------------------
# Asignacion de terceros a las ranuras de la Ronda de 32 (matching bipartito)
# ---------------------------------------------------------------------------

def assign_thirds(best_thirds, r32):
    """Asigna los 8 terceros clasificados a las 8 ranuras '3' respetando los
    grupos admitidos por cada ranura. Matching maximo (Kuhn)."""
    slots = []  # (match, set(grupos admitidos))
    for m in r32:
        for side in ("a", "b"):
            spec = m[side]
            if spec["type"] == 3:
                slots.append((m["match"], set(spec["groups"])))

    third_group = [g for (g, _team, _st) in best_thirds]  # grupo de cada tercero
    n = len(slots)
    # adyacencia: tercero i -> ranuras j compatibles
    adj = [[j for j, (_mt, allowed) in enumerate(slots) if third_group[i] in allowed]
           for i in range(len(third_group))]

    match_to_third = [-1] * n  # ranura j -> indice de tercero

    def try_assign(i, seen):
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if match_to_third[j] == -1 or try_assign(match_to_third[j], seen):
                    match_to_third[j] = i
                    return True
        return False

    for i in range(len(third_group)):
        try_assign(i, [False] * n)

    # Resultado: match -> equipo tercero. Fallback para no asignados.
    slot_team = {}
    unassigned_slots = []
    used = set()
    for j, (mt, _allowed) in enumerate(slots):
        i = match_to_third[j]
        if i == -1:
            unassigned_slots.append(mt)
        else:
            slot_team[mt] = best_thirds[i][1]
            used.add(i)
    leftover = [best_thirds[i][1] for i in range(len(third_group)) if i not in used]
    for mt, team in zip(unassigned_slots, leftover):
        slot_team[mt] = team
    return slot_team


# ---------------------------------------------------------------------------
# Eliminacion directa
# ---------------------------------------------------------------------------

def resolve_slot(spec, standings, third_slot_team, match_number):
    t = spec["type"]
    if t == 1:
        return standings[spec["group"]][0]
    if t == 2:
        return standings[spec["group"]][1]
    return third_slot_team[match_number]


def simulate_knockout(standings, best_thirds, bracket, model, rng):
    r32 = bracket["r32"]
    third_slot_team = assign_thirds(best_thirds, r32)

    winners = {}
    for m in r32:
        a = resolve_slot(m["a"], standings, third_slot_team, m["match"])
        b = resolve_slot(m["b"], standings, third_slot_team, m["match"])
        winners[m["match"]] = model.play_knockout(a, b, rng)

    for node in bracket["tree"]:
        a = winners[node["a"]]
        b = winners[node["b"]]
        winners[node["match"]] = model.play_knockout(a, b, rng)

    final_match = bracket["tree"][-1]
    champion = winners[final_match["match"]]
    finalists = (winners[final_match["a"]], winners[final_match["b"]])
    return champion, finalists


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run(n=20000, seed=None, base=1.35, home_adv=60.0, data_base=None, progress=None):
    data = load_all(data_base)
    model = MatchModel(data["elo"], base=base, home_adv=home_adv)
    rng = random.Random(seed)

    all_teams = list(data["elo"].keys())
    champ = {t: 0 for t in all_teams}
    final = {t: 0 for t in all_teams}

    for i in range(n):
        standings, best_thirds = simulate_group_stage(
            data["groups"], data["fixtures"], model, rng)
        champion, finalists = simulate_knockout(
            standings, best_thirds, data["bracket"], model, rng)
        champ[champion] += 1
        for f in finalists:
            final[f] += 1
        if progress and (i + 1) % progress == 0:
            print(f"  ... {i + 1}/{n} simulaciones", flush=True)

    results = []
    for t in all_teams:
        results.append({
            "team": t,
            "elo": data["elo"][t],
            "titles": champ[t],
            "p_champion": champ[t] / n,
            "finals": final[t],
            "p_final": final[t] / n,
        })
    results.sort(key=lambda r: r["titles"], reverse=True)
    return results, n
