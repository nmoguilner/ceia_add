"""Descarga microdatos de la EPH (INDEC) y los guarda como CSV en data/raw/.

Pensado para correr en TU máquina (con internet). El sandbox del asistente
tiene la red bloqueada, por eso este paso lo hacés vos una sola vez.

Requisitos:
    pip install pyeph pandas

Uso:
    python -m src.data.descargar_eph                 # baja 2016..2024, base individual
    python -m src.data.descargar_eph --desde 2019 --hasta 2024
    python -m src.data.descargar_eph --base hogar

Salida:
    data/raw/eph_individual_2024T1.csv, ... (un archivo por trimestre)
    data/raw/eph_individual_TODO.csv         (todos los trimestres apilados)
"""
import argparse
from pathlib import Path

import pandas as pd

try:
    import pyeph
except ImportError:
    raise SystemExit("Falta pyeph. Instalá con: pip install pyeph pandas")

OUT = Path("data/raw")


def main(desde: int, hasta: int, base: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    partes = []
    for year in range(desde, hasta + 1):
        for period in (1, 2, 3, 4):
            etiqueta = f"{year}T{period}"
            try:
                df = pyeph.get(data="eph", year=year, period=period, base_type=base)
            except Exception as e:
                print(f"  [skip] {etiqueta}: {str(e)[:80]}")
                continue
            df["anio"], df["trimestre"] = year, period
            archivo = OUT / f"eph_{base}_{etiqueta}.csv"
            df.to_csv(archivo, index=False)
            partes.append(df)
            print(f"  [ok]   {etiqueta}: {df.shape[0]} filas -> {archivo.name}")

    if partes:
        todo = pd.concat(partes, ignore_index=True)
        destino = OUT / f"eph_{base}_TODO.csv"
        todo.to_csv(destino, index=False)
        print(f"\nApilado: {todo.shape[0]} filas, {todo.shape[1]} cols -> {destino}")
    else:
        print("\nNo se descargó ningún trimestre. Revisá tu conexión.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--desde", type=int, default=2016)
    p.add_argument("--hasta", type=int, default=2024)
    p.add_argument("--base", choices=["individual", "hogar"], default="individual")
    args = p.parse_args()
    main(args.desde, args.hasta, args.base)
