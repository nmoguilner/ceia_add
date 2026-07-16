#!/usr/bin/env python3
"""
Hito 2 del TP de Aprendizaje de Maquina: entrenamiento y comparacion de modelos.

Lee data/ml_dataset.parquet (de featurize.py), entrena >=3 familias de modelos
con validacion cruzada TEMPORAL y optimizacion de hiperparametros, selecciona el
mejor por LOG-LOSS y lo evalua UNA sola vez sobre el test reservado (Mundial
2026). Compara contra el baseline ELO (mismo conjunto) y el baseline trivial,
y corre la ablacion de features (solo-ELO -> +forma -> todo).

  uv run --extra ml python train.py

Salidas:
  models/ml_match_model.joblib   mejor pipeline (preprocesador + clasificador)
  data/ml_report.json            metricas de CV, test, baselines y ablacion

Reglas (DISENO_TP_ML.md): split y CV estrictamente temporales; escalador
ajustado solo con train; SIN class_weight (la calibracion importa porque el
modelo alimentara el Monte Carlo); SVM calibrado (Platt descalibra).
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "data", "ml_dataset.parquet")
MODEL_OUT = os.path.join(HERE, "models", "ml_match_model.joblib")
REPORT_OUT = os.path.join(HERE, "data", "ml_report.json")

CLASSES = ["A", "D", "H"]                 # orden lexicografico (= el de sklearn)
NUMERIC = ["delta_elo", "elo_home", "elo_away", "is_home",
           "form_gd_5_home", "form_gd_5_away", "form_pts_5_home", "form_pts_5_away",
           "rest_days_home", "rest_days_away", "tournament_weight", "h2h_recent"]
CATEGORICAL = ["confed_home", "confed_away"]

# Subconjuntos para la ablacion de la seccion 6 (la prueba de defensa central).
ABLATION = {
    "solo_elo":    (["delta_elo", "elo_home", "elo_away", "is_home"], []),
    "elo_forma":   (["delta_elo", "elo_home", "elo_away", "is_home",
                     "form_gd_5_home", "form_gd_5_away", "form_pts_5_home",
                     "form_pts_5_away", "rest_days_home", "rest_days_away"], []),
    "completo":    (NUMERIC, CATEGORICAL),
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def load_dataset():
    df = pd.read_parquet(DATASET)
    # Preprocesamiento (sec. 7): descartar no-FIFA (sin confederacion) -> saca el
    # ruido CONIFA y deja solo internacionales reales.
    before = len(df)
    df = df[(df.confed_home != "OTH") & (df.confed_away != "OTH")].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Dataset: {before} -> {len(df)} filas tras quitar no-FIFA (OTH)")
    return df


def make_preprocessor(numeric, categorical):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    transformers = [("num", num, numeric)]
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers)


# ---------------------------------------------------------------------------
# Modelos y grillas (modestas, para CV temporal tratable)
# ---------------------------------------------------------------------------

def model_zoo():
    return {
        "logreg": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "knn": (
            KNeighborsClassifier(weights="distance"),
            {"clf__n_neighbors": [25, 50, 100]},
        ),
        "svm_rbf": (
            CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=3),
            {"clf__estimator__C": [1.0, 10.0], "clf__estimator__gamma": ["scale"]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
            {"clf__max_depth": [None, 10, 20], "clf__min_samples_leaf": [1, 5]},
        ),
        "hist_gb": (
            HistGradientBoostingClassifier(random_state=0),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_iter": [200, 400]},
        ),
    }


# ---------------------------------------------------------------------------
# Metricas (independientes del estimador; matriz de prob. en orden CLASSES)
# ---------------------------------------------------------------------------

def metrics(y_true, proba):
    """y_true: lista de etiquetas; proba: matriz N x 3 en orden CLASSES."""
    y_idx = np.array([CLASSES.index(y) for y in y_true])
    P = np.asarray(proba)
    ll = log_loss(y_true, P, labels=CLASSES)
    onehot = np.eye(len(CLASSES))[y_idx]
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    pred = P.argmax(axis=1)
    acc = float(np.mean(pred == y_idx))
    f1 = float(f1_score(y_idx, pred, average="macro"))
    return {"log_loss": float(ll), "brier": brier, "accuracy": acc, "f1_macro": f1, "n": len(y_true)}


def proba_in_class_order(estimator, X):
    """Reordena predict_proba al orden CLASSES (sklearn ordena alfabeticamente)."""
    P = estimator.predict_proba(X)
    order = [list(estimator.classes_).index(c) for c in CLASSES]
    return P[:, order]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def elo_wdl(delta_elo, is_home, cal, maxg=16):
    """P(H/D/A) analitica del motor ELO-Poisson + Dixon-Coles, sobre el MISMO
    delta_elo del dataset. Replica backtest.wdl pero alimentado por la feature."""
    base, scale, hadv, rho = cal["mu"], cal["escala"], cal["home_adv_elo"], cal.get("rho", 0.0)
    diff = delta_elo + (hadv if is_home else 0.0)
    la = base * 10.0 ** (diff / scale)
    lb = base * 10.0 ** (-diff / scale)
    pa = [math.exp(-la) * la ** k / math.factorial(k) for k in range(maxg)]
    pb = [math.exp(-lb) * lb ** k / math.factorial(k) for k in range(maxg)]

    def tau(x, y):
        if rho == 0.0:
            return 1.0
        if x == 0 and y == 0: return 1.0 - la * lb * rho
        if x == 0 and y == 1: return 1.0 + la * rho
        if x == 1 and y == 0: return 1.0 + lb * rho
        if x == 1 and y == 1: return 1.0 - rho
        return 1.0

    M = [[pa[i] * pb[j] * tau(i, j) for j in range(maxg)] for i in range(maxg)]
    s = sum(M[i][j] for i in range(maxg) for j in range(maxg))
    p_h = sum(M[i][j] for i in range(maxg) for j in range(maxg) if i > j) / s
    p_d = sum(M[i][i] for i in range(maxg)) / s
    p_a = sum(M[i][j] for i in range(maxg) for j in range(maxg) if i < j) / s
    return [p_a, p_d, p_h]   # orden CLASSES = A, D, H


def elo_baseline_probs(df_test, cal):
    return [elo_wdl(r.delta_elo, r.is_home, cal) for r in df_test.itertuples()]


def trivial_probs(df_train, n_test):
    """Frecuencia de clases del train, repetida para cada fila de test."""
    freq = df_train["label"].value_counts(normalize=True)
    vec = [float(freq.get(c, 0.0)) for c in CLASSES]
    return [vec[:] for _ in range(n_test)]


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_and_select(df_train, df_test):
    X_tr, y_tr = df_train[NUMERIC + CATEGORICAL], df_train["label"]
    X_te, y_te = df_test[NUMERIC + CATEGORICAL], df_test["label"]
    cv = TimeSeriesSplit(n_splits=5)

    results, fitted = {}, {}
    for name, (clf, grid) in model_zoo().items():
        pipe = Pipeline([("prep", make_preprocessor(NUMERIC, CATEGORICAL)), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, scoring="neg_log_loss", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        cv_ll = -gs.best_score_
        test_m = metrics(y_te.tolist(), proba_in_class_order(gs.best_estimator_, X_te))
        results[name] = {"cv_log_loss": cv_ll, "best_params": gs.best_params_, "test": test_m}
        fitted[name] = gs.best_estimator_
        print(f"  {name:14s} CV logloss={cv_ll:.4f}  | test logloss={test_m['log_loss']:.4f} "
              f"acc={100*test_m['accuracy']:.0f}% brier={test_m['brier']:.3f}")

    best = min(results, key=lambda k: results[k]["cv_log_loss"])
    return results, fitted, best


def run_ablation(df_train, df_test, clf, grid):
    """Reentrena la familia ganadora sobre cada subconjunto de features."""
    cv = TimeSeriesSplit(n_splits=5)
    out = {}
    for tag, (num, cat) in ABLATION.items():
        cols = num + cat
        pipe = Pipeline([("prep", make_preprocessor(num, cat)), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, scoring="neg_log_loss", cv=cv, n_jobs=-1, refit=True)
        gs.fit(df_train[cols], df_train["label"])
        test_m = metrics(df_test["label"].tolist(),
                         proba_in_class_order(gs.best_estimator_, df_test[cols]))
        out[tag] = {"cv_log_loss": -gs.best_score_, "test_log_loss": test_m["log_loss"],
                    "test_brier": test_m["brier"], "test_accuracy": test_m["accuracy"]}
        print(f"  ablacion[{tag:10s}] test logloss={test_m['log_loss']:.4f} "
              f"acc={100*test_m['accuracy']:.0f}%")
    return out


def main():
    df = load_dataset()
    df_train = df[df.split == "train"].reset_index(drop=True)
    df_test = df[df.split == "test"].reset_index(drop=True)
    print(f"train={len(df_train)}  test={len(df_test)}\n")

    print("Modelos (CV temporal, seleccion por log-loss):")
    results, fitted, best = train_and_select(df_train, df_test)
    print(f"\n>> Mejor modelo por CV log-loss: {best}\n")

    # Baselines sobre el MISMO test. El delta_elo del dataset es el ELO
    # RECONSTRUIDO pre-partido -> hay que usar la calibracion ajustada sobre ESE
    # ELO (calibration_prematch.json), no la del proxy wfr (calibration.json).
    # El rho de Dixon-Coles no se reajusto para el ELO reconstruido; se toma el de
    # calibration.json (correccion de marcador bajo, ~independiente de la escala).
    with open(os.path.join(HERE, "data", "calibration_prematch.json")) as f:
        cal = json.load(f)
    with open(os.path.join(HERE, "data", "calibration.json")) as f:
        cal["rho"] = json.load(f).get("rho", 0.0)
    y_te = df_test["label"].tolist()
    elo_m = metrics(y_te, elo_baseline_probs(df_test, cal))
    triv_m = metrics(y_te, trivial_probs(df_train, len(df_test)))
    print("Baselines sobre el test (Mundial 2026):")
    print(f"  ELO (calibrado+DC) logloss={elo_m['log_loss']:.4f} acc={100*elo_m['accuracy']:.0f}% "
          f"brier={elo_m['brier']:.3f}")
    print(f"  Trivial (frec.)    logloss={triv_m['log_loss']:.4f} acc={100*triv_m['accuracy']:.0f}% "
          f"brier={triv_m['brier']:.3f}")

    print(f"\nAblacion de features (familia ganadora: {best}):")
    clf, grid = model_zoo()[best]
    ablation = run_ablation(df_train, df_test, clf, grid)

    # Persistencia
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump({"pipeline": fitted[best], "classes": CLASSES,
                 "numeric": NUMERIC, "categorical": CATEGORICAL}, MODEL_OUT)
    report = {
        "best_model": best,
        "n_train": len(df_train), "n_test": len(df_test),
        "models": results,
        "baseline_elo": elo_m, "baseline_trivial": triv_m,
        "ablation": ablation,
    }
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado -> {os.path.relpath(MODEL_OUT, HERE)} y {os.path.relpath(REPORT_OUT, HERE)}")

    # Veredicto rapido vs ELO
    best_ll = results[best]["test"]["log_loss"]
    delta = elo_m["log_loss"] - best_ll
    verdict = "MEJORA" if delta > 0 else "NO mejora"
    print(f"\n{best} vs ELO en test: log-loss {best_ll:.4f} vs {elo_m['log_loss']:.4f} "
          f"-> {verdict} ({delta:+.4f})")


if __name__ == "__main__":
    main()
