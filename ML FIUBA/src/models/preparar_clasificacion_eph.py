"""Prepara el dataset de CLASIFICACIÓN: ocupado vs no-ocupado (35-64).

Mide la exclusión del empleo controlando por calificación, sexo y región.
- Cohorte 35-64: deja afuera 65+ para no confundir expulsión con jubilación.
- Sin variables que existen solo si la persona trabaja (cat_ocup, descuento_jub):
  serían fuga de información del target.

Genera data/processed/eph_clasificacion.csv

Uso:
    uv run python -m src.models.preparar_clasificacion_eph
"""
import os
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
SRC = PROC / "eph_analitico.parquet"
OUT = PROC / "eph_clasificacion.csv"
# Interruptor de costo compartido con el notebook: ML_FIUBA_MODO = rapido | completo
MODO = os.environ.get("ML_FIUBA_MODO", "rapido").strip().lower()
N_MUESTRA = {"rapido": 15_000, "completo": None}.get(MODO, 15_000)  # None = dataset completo
SEED = 42

REGION = {1: "GBA", 40: "Noroeste", 41: "Nordeste", 42: "Cuyo",
          43: "Pampeana", 44: "Patagonia"}


def main() -> None:
    df = pd.read_parquet(SRC)

    # Población activa/inactiva (excluye 'menor' = ESTADO 4 y NaN)
    df = df[df["ESTADO"].isin([1, 2, 3])].copy()
    df = df[df["edad"].between(35, 64)]

    df["empleo"] = (df["ESTADO"] == 1).astype(int)   # 1 = ocupado, 0 = no ocupado
    df["region"] = df["REGION"].map(REGION)

    cols = ["empleo", "edad", "sexo", "nivel_ed", "region", "anio"]
    modelo = df[cols].rename(columns={"empleo": "target"}).dropna(
        subset=["sexo", "nivel_ed"])

    if N_MUESTRA and len(modelo) > N_MUESTRA:   # None (modo completo) => sin muestrear
        modelo = modelo.sample(N_MUESTRA, random_state=SEED)

    modelo.to_csv(OUT, index=False)
    print(f"[modo {MODO}] Dataset clasificación: {OUT}  ({modelo.shape[0]:,} filas)")
    print("\nDistribución del target (1=ocupado, 0=no ocupado):")
    print(modelo["target"].value_counts(normalize=True).round(3).to_string())
    print("\nTasa de empleo por sexo:")
    print(modelo.groupby("sexo")["target"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
