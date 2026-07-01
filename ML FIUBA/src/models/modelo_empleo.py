"""Clasificación ocupado/no-ocupado (35-64): comparación + penalización por edad.

- Baseline (clase mayoritaria) + 3 modelos, con accuracy y ROC-AUC (cv=5).
- Gráfico estrella: probabilidad de estar ocupado vs edad, por sexo,
  controlando por educación y región (partial dependence manual).

Uso:
    uv run python -m src.models.modelo_empleo
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.models.train import build_preprocessor

DATA = Path("data/processed/eph_clasificacion.csv")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=["target"]), df["target"]

    modelos = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }

    print(f"{'modelo':22} {'accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 44)
    for nombre, est in modelos.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", est)])
        sc = cross_validate(pipe, X, y, cv=5, scoring=["accuracy", "roc_auc"], n_jobs=-1)
        print(f"{nombre:22} {sc['test_accuracy'].mean():>10.3f} {sc['test_roc_auc'].mean():>10.3f}")

    # --- Mejor modelo (gradient boosting) entrenado para la curva ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    best = Pipeline([("prep", build_preprocessor(X)),
                     ("model", GradientBoostingClassifier(random_state=SEED))]).fit(X_tr, y_tr)

    # --- Partial dependence manual: prob(ocupado) vs edad, por sexo ---
    edades = np.arange(35, 65)
    plt.figure(figsize=(10, 5.5))
    for sx, col in [("Varón", "#4C72B0"), ("Mujer", "#C44E52")]:
        base = X_te[X_te["sexo"] == sx].copy()
        probs = []
        for e in edades:
            tmp = base.copy(); tmp["edad"] = e
            probs.append(best.predict_proba(tmp)[:, 1].mean())
        plt.plot(edades, probs, label=sx, color=col, lw=2)
    plt.axvline(60, ls="--", c="#C44E52", alpha=.6); plt.text(60.1, 0.5, "Jub. mujer (60)", color="#C44E52")
    plt.xlabel("Edad"); plt.ylabel("Probabilidad de estar ocupado")
    plt.title("Penalización por edad: prob. de empleo vs edad, por sexo\n(controlando educación y región — partial dependence)")
    plt.legend(); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(FIG / "pdp_edad_empleo.png", dpi=120); plt.close()

    p35 = best.predict_proba(X_te.assign(edad=35))[:, 1].mean()
    p60 = best.predict_proba(X_te.assign(edad=60))[:, 1].mean()
    print(f"\nProb. media de empleo: edad 35 = {p35:.2f}  ->  edad 60 = {p60:.2f}  "
          f"(caída de {100*(p35-p60):.0f} puntos)")
    print(f"Figura: {FIG/'pdp_edad_empleo.png'}")


if __name__ == "__main__":
    main()
