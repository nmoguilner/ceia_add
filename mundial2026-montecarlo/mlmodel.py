#!/usr/bin/env python3
"""
MLMatchModel: motor de partido respaldado por el clasificador supervisado del TP,
inyectable en el Monte Carlo de wcsim en lugar del MatchModel (ELO-Poisson).

Cumple el contrato real que usa el simulador (DISENO_TP_ML.md sec. 12):
  - play_group(a, b, rng)            -> (gh, ga)   marcador (para desempates GF/GA)
  - play_knockout(a, b, rng, venue)  -> ganador    (cascada 90' -> alargue -> penales)
  - atributo .elo                    -> dict usado por el simulador en los desempates

Es ESTATICO: las features (ELO reconstruido, forma, h2h, confederacion) se toman
del estado actual del historico y se precomputa de una vez la tabla P(A/D/H) para
todos los cruces posibles; en el loop del MC solo se muestrea de la tabla. Llamar
predict_proba por partido seria inviable (millones de llamadas).

Cascada de eliminacion (sec. 12.1):
  - 90'    -> clasificador ML (H/D/A)
  - alargue-> si el ML dice empate, mini-partido: prob. de gol proporcional a la
              ventaja ELO; si sigue igualado, penales
  - penales-> casi moneda con sesgo leve por ELO (clamp ~55/45)
"""
import json
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import joblib

from wcsim import HOSTS, HOST_COUNTRY
from elo_history import k_factor, goal_mult, MI_A_DATASET
from calibrate import cargar_historico
from featurize import infer_confederations, FORM_N, H2H_N, tournament_weight

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "ml_match_model.joblib")
WC_TOURNAMENT_WEIGHT = tournament_weight("FIFA World Cup")  # 4.0
WC_REST_DAYS = 4.0      # los fixtures simulados no tienen calendario: descanso tipico de Mundial
PEN_CLAMP = (0.45, 0.55)  # sesgo maximo por ELO en la tanda de penales


def _forward_state(rows):
    """Una pasada cronologica que reconstruye, al estado ACTUAL del historico:
       R[team]       ELO reconstruido (eloratings, mismos nombres martj42)
       form[team]    deque de (gd, pts) de los ultimos FORM_N partidos
       last[team]    fecha del ultimo partido (no usado en simulacion, ver WC_REST_DAYS)
       h2h[par]      deque de ganadores recientes entre ambos
    """
    R = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: deque(maxlen=FORM_N))
    h2h = defaultdict(lambda: deque(maxlen=H2H_N))
    for r in sorted(rows, key=lambda r: r["date"]):
        if r["home_score"] is None or r["away_score"] is None:
            continue
        h, a = r["home_team"], r["away_team"]
        gh, ga = int(r["home_score"]), int(r["away_score"])
        neutral = bool(r["neutral"])
        rh, ra = R[h], R[a]
        # update ELO (identico a elo_history.reconstruir)
        dr = (rh + (0 if neutral else 100)) - ra
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        chg = k_factor(r["tournament"]) * goal_mult(abs(gh - ga)) * (w - we)
        R[h], R[a] = rh + chg, ra - chg
        # estado de forma / h2h
        gd = gh - ga
        form[h].append((gd, 3 if gh > ga else (1 if gh == ga else 0)))
        form[a].append((-gd, 3 if ga > gh else (1 if ga == gh else 0)))
        h2h[tuple(sorted((h, a)))].append(h if gh > ga else (a if ga > gh else "D"))
    return R, form, h2h


def _empirical_scores(rows):
    """Distribuciones empiricas de marcador condicionadas al resultado (desde la
    perspectiva del local): listas de (gh, ga) para muestrear el marcador una vez
    decidido H/D/A. Solo partidos recientes (>=2015) para marcadores realistas."""
    buckets = {"H": [], "D": [], "A": []}
    for r in rows:
        if r["home_score"] is None or str(r["date"]) < "2015-01-01":
            continue
        gh, ga = int(r["home_score"]), int(r["away_score"])
        key = "H" if gh > ga else ("A" if ga > gh else "D")
        buckets[key].append((gh, ga))
    return buckets


class MLMatchModel:
    """Motor de partido ML, interfaz compatible con wcsim.MatchModel."""

    def __init__(self, elo, model_path=MODEL_PATH, rng_scores=None):
        self.elo = elo                      # ELO wfr (mi-nombres) para los desempates del sim
        bundle = joblib.load(model_path)
        self.pipe = bundle["pipeline"]
        self.classes = bundle["classes"]    # orden de columnas de la tabla P
        self.numeric = bundle["numeric"]
        self.categorical = bundle["categorical"]

        rows = cargar_historico()
        self.R, self.form, self.h2h = _forward_state(rows)
        self.conf = infer_confederations(rows)
        self.scores = _empirical_scores(rows)

        self._cache = {}                    # (a_ds, b_ds, is_home) -> [P(A),P(D),P(H)]
        self._iH = self.classes.index("H")
        self._iD = self.classes.index("D")
        self._iA = self.classes.index("A")

    # --- features ---------------------------------------------------------
    def _ds(self, team):
        return MI_A_DATASET.get(team, team)   # mi-nombre -> nombre martj42

    def _avg(self, seq, idx):
        seq = [s[idx] for s in seq]
        return sum(seq) / len(seq) if seq else np.nan

    def _feature_row(self, a, b, is_home):
        ad, bd = self._ds(a), self._ds(b)
        fa, fb = self.form[ad], self.form[bd]
        recent = self.h2h[tuple(sorted((ad, bd)))]
        h2h_recent = sum(1 if w == ad else (-1 if w == bd else 0) for w in recent)
        return {
            "delta_elo": self.R[ad] - self.R[bd],
            "elo_home": self.R[ad], "elo_away": self.R[bd],
            "is_home": is_home,
            "form_gd_5_home": self._avg(fa, 0), "form_gd_5_away": self._avg(fb, 0),
            "form_pts_5_home": self._avg(fa, 1), "form_pts_5_away": self._avg(fb, 1),
            "rest_days_home": WC_REST_DAYS, "rest_days_away": WC_REST_DAYS,
            "tournament_weight": WC_TOURNAMENT_WEIGHT,
            "h2h_recent": h2h_recent,
            "confed_home": self.conf.get(ad, "OTH"),
            "confed_away": self.conf.get(bd, "OTH"),
        }

    def precompute(self, teams):
        """Precomputa P(A/D/H) para todos los cruces ordenados de `teams`, en ambos
        valores de is_home, en una sola llamada a predict_proba (batched)."""
        combos = [(a, b, h) for a in teams for b in teams if a != b for h in (0, 1)]
        X = pd.DataFrame([self._feature_row(a, b, h) for a, b, h in combos],
                         columns=self.numeric + self.categorical)
        P = self.pipe.predict_proba(X)      # columnas en orden self.pipe.classes_
        order = [list(self.pipe.classes_).index(c) for c in self.classes]
        P = P[:, order]
        for (a, b, h), p in zip(combos, P):
            self._cache[(a, b, h)] = p
        return self

    def _probs(self, a, b, is_home):
        key = (a, b, is_home)
        p = self._cache.get(key)
        if p is None:                        # cruce no precomputado: calcular al vuelo
            X = pd.DataFrame([self._feature_row(a, b, is_home)],
                             columns=self.numeric + self.categorical)
            pr = self.pipe.predict_proba(X)[0]
            order = [list(self.pipe.classes_).index(c) for c in self.classes]
            p = pr[order]
            self._cache[key] = p
        return p[self._iH], p[self._iD], p[self._iA]

    # --- compatibilidad con el simulador ---------------------------------
    def _is_home(self, team, venue):
        if venue is None:
            return 1 if team in HOSTS else 0
        return 1 if HOST_COUNTRY.get(team) == venue else 0

    def _sample_outcome(self, a, b, rng, venue=None):
        ph, pd_, pa = self._probs(a, b, self._is_home(a, venue))
        u = rng.random()
        if u < ph:
            return "H"
        if u < ph + pd_:
            return "D"
        return "A"

    def play_group(self, a, b, rng):
        out = self._sample_outcome(a, b, rng)
        gh, ga = rng.choice(self.scores[out])
        return gh, ga

    def play_knockout(self, a, b, rng, venue=None):
        out = self._sample_outcome(a, b, rng, venue)
        if out == "H":
            return a
        if out == "A":
            return b
        return self._extra_time(a, b, rng, venue)

    def _extra_time(self, a, b, rng, venue):
        """Alargue como mini-partido (sec. 12.1): prob. de definir por ELO clasico,
        con intensidad escalada; si no se rompe, penales (casi moneda)."""
        diff = self.R[self._ds(a)] - self.R[self._ds(b)]
        p_a_wins = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))   # P(a) | hay ganador
        if rng.random() < 30.0 / 90.0:                      # se define en el alargue
            return a if rng.random() < p_a_wins else b
        lo, hi = PEN_CLAMP                                   # penales: sesgo leve clamp
        p_pen = min(hi, max(lo, p_a_wins))
        return a if rng.random() < p_pen else b
