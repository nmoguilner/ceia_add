"""Entrenamiento del modelo para la cursada de AdM - CEIA FIUBA.

Pipeline reproducible que:
- soporta clasificación y regresión (campo `task` del config),
- maneja automáticamente variables numéricas y categóricas
  (imputación + escalado + one-hot encoding),
- permite elegir el modelo desde el config sin tocar el código.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_estimator(cfg: dict):
    task = cfg.get("task", "classification")
    mtype = cfg["model"]["type"]
    params = cfg["model"].get("params", {}) or {}
    seed = cfg["seed"]

    if task == "regression":
        catalogo = {
            "baseline": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(random_state=seed, **params),
            "knn": KNeighborsRegressor(),
            "svm": SVR(),
            "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        }
    else:
        catalogo = {
            "baseline": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
            "random_forest": RandomForestClassifier(random_state=seed, **params),
            "knn": KNeighborsClassifier(),
            "svm": SVC(random_state=seed, probability=True),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        }
    if mtype not in catalogo:
        raise ValueError(f"Modelo '{mtype}' no válido para task={task}. Opciones: {list(catalogo)}")
    return catalogo[mtype]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def load_xy(cfg: dict):
    df = pd.read_csv(cfg["data"]["raw_path"])
    drop_cols = cfg["data"].get("drop_cols", []) or []
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    target = cfg["data"]["target"]
    df = df.dropna(subset=[target])
    return df.drop(columns=[target]), df[target]


def report(task: str, y_test, preds) -> None:
    if task == "regression":
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"  R2   : {r2_score(y_test, preds):.3f}")
        print(f"  MAE  : {mean_absolute_error(y_test, preds):.3f}")
        print(f"  RMSE : {rmse:.3f}")
    else:
        print(classification_report(y_test, preds))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    task = cfg.get("task", "classification")
    X, y = load_xy(cfg)

    strat = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["seed"], stratify=strat,
    )

    pipe = Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", build_estimator(cfg)),
    ])
    pipe.fit(X_train, y_train)

    print(f"Task: {task} | Modelo: {cfg['model']['type']}")
    report(task, y_test, pipe.predict(X_test))

    out = Path(cfg["model"]["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"Modelo guardado en: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
