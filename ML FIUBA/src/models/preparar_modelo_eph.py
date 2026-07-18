"""Prepara el dataset de modelado a partir de la base analítica de la EPH.

Problema: predecir el INGRESO LABORAL RELATIVO de personas ocupadas (25-70),
controlando por educación, sexo, edad y región. El efecto residual de edad y
sexo (tras controlar por calificación) es la lente de discriminación.

Genera data/processed/eph_modelo.csv (muestra para entrenar varios modelos).

Uso:
    uv run python -m src.models.preparar_modelo_eph
"""
import os
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
SRC = PROC / "eph_analitico.parquet"
OUT = PROC / "eph_modelo.csv"
# Interruptor de costo compartido con el notebook: ML_FIUBA_MODO = rapido | completo
MODO = os.environ.get("ML_FIUBA_MODO", "rapido").strip().lower()
N_MUESTRA = {"rapido": 15_000, "completo": None}.get(MODO, 15_000)  # None = dataset completo
SEED = 42


def main() -> None:
    df = pd.read_parquet(SRC)

    # Ingreso relativo a la mediana de ocupados de cada trimestre (saca inflación)
    ocup = df[(df["ESTADO"] == 1) & (df["P21"] > 0)].copy()
    med = ocup.groupby(["anio", "trimestre"])["P21"].transform("median")
    ocup["ingreso_rel"] = ocup["P21"] / med

    # Cohorte de interés y features
    ocup = ocup[ocup["edad"].between(25, 70)]
    cols = {
        "ingreso_rel": "target",
        "edad": "edad", "sexo": "sexo", "nivel_ed": "nivel_ed",
        "REGION": "region", "CAT_OCUP": "cat_ocup", "PP07H": "descuento_jub",
    }
    modelo = ocup[list(cols)].rename(columns=cols).dropna(subset=["target"])

    if N_MUESTRA and len(modelo) > N_MUESTRA:   # None (modo completo) => sin muestrear
        modelo = modelo.sample(N_MUESTRA, random_state=SEED)

    modelo.to_csv(OUT, index=False)
    print(f"[modo {MODO}] Dataset de modelado: {OUT}  ({modelo.shape[0]:,} filas, {modelo.shape[1]} cols)")
    print("Columnas:", list(modelo.columns))
    print("\nTarget (ingreso relativo) describe:")
    print(modelo["target"].describe()[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
