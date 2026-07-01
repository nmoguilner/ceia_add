"""Compara varios modelos de ML sobre el dataset con validación cruzada.

Detecta automáticamente si el problema es de clasificación o regresión
(según `task` en el config) y reporta la métrica adecuada.

Uso:
    python -m src.models.benchmark --config configs/config.yaml
"""
import argparse

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.models.train import build_estimator, build_preprocessor, load_config, load_xy


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    task = cfg.get("task", "classification")
    X, y = load_xy(cfg)

    if task == "regression":
        modelos = ["baseline", "linear_regression", "knn", "svm", "random_forest", "gradient_boosting"]
        scoring, etiqueta, mejor_fn = "r2", "R2 (cv=5)", max
    else:
        modelos = ["baseline", "logistic_regression", "knn", "svm", "random_forest", "gradient_boosting"]
        scoring, etiqueta, mejor_fn = "accuracy", "accuracy (cv=5)", max

    print(f"Task: {task}\n{'modelo':22} {etiqueta:>18}")
    print("-" * 42)
    resultados = []
    for nombre in modelos:
        cfg["model"]["type"] = nombre
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", build_estimator(cfg))])
        scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring, n_jobs=-1)
        resultados.append((nombre, scores.mean(), scores.std()))
        print(f"{nombre:22} {scores.mean():.3f} +/- {scores.std():.3f}")

    mejor = mejor_fn(resultados, key=lambda r: r[1])
    print("-" * 42)
    print(f"Mejor: {mejor[0]} ({mejor[1]:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
