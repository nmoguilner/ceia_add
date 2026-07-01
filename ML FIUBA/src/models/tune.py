"""Tuning de hiperparámetros + importancia de variables - AdM CEIA FIUBA.

- Busca los mejores hiperparámetros con GridSearchCV para el modelo del config.
- Calcula la importancia por permutación (válida para cualquier modelo).
- Guarda el mejor modelo y una figura con el top de variables.

Uso:
    python -m src.models.tune --config configs/config.yaml
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.models.train import build_estimator, build_preprocessor, load_config, load_xy

# Grillas por modelo (prefijo model__ apunta al estimador dentro del Pipeline)
GRIDS = {
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_leaf": [1, 5],
    },
    "gradient_boosting": {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
    "knn": {"model__n_neighbors": [5, 11, 21], "model__weights": ["uniform", "distance"]},
    "svm": {"model__C": [0.1, 1, 10]},
    "linear_regression": {},
    "logistic_regression": {"model__C": [0.1, 1, 10]},
}


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    task = cfg.get("task", "classification")
    mtype = cfg["model"]["type"]
    X, y = load_xy(cfg)

    strat = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["seed"], stratify=strat,
    )

    pipe = Pipeline([("prep", build_preprocessor(X)), ("model", build_estimator(cfg))])
    scoring = "r2" if task == "regression" else "accuracy"
    grid = GRIDS.get(mtype, {})

    gs = GridSearchCV(pipe, grid, cv=5, scoring=scoring, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    print(f"Task: {task} | Modelo: {mtype}")
    print(f"Mejores params: {gs.best_params_}")
    print(f"Mejor {scoring} (CV): {gs.best_score_:.3f}")
    print(f"{scoring} en test : {gs.score(X_te, y_te):.3f}")

    # Importancia por permutación (sobre features originales)
    r = permutation_importance(
        gs.best_estimator_, X_te, y_te, n_repeats=10,
        random_state=cfg["seed"], scoring=scoring, n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 variables (importancia por permutación):")
    print(imp.head(10).round(4).to_string())

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    top = imp.head(15)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values, color="#4C72B0")
    plt.xlabel(f"Caída de {scoring} al permutar")
    plt.title(f"Importancia de variables - {mtype}")
    plt.tight_layout()
    fig_path = "reports/figures/feature_importance.png"
    plt.savefig(fig_path, dpi=120)
    print(f"\nFigura guardada: {fig_path}")

    out = Path(cfg["model"]["output_path"]).with_name("model_tuned.joblib")
    joblib.dump(gs.best_estimator_, out)
    print(f"Mejor modelo guardado: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
