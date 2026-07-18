"""Panel de la EPH: transiciones laborales y reinserción por edad.

Enlaza a la misma persona entre trimestres consecutivos (CODUSU+NRO_HOGAR+
COMPONENTE) y estudia, entre los DESOCUPADOS en t:
  - reinserción (ocupado en t+1) por edad,
  - salida a inactividad por edad,
  - descenso ocupacional al reinsertarse (calificación del nuevo empleo peor que
    la del último empleo perdido).

Uso:
    uv run python -m src.data.panel_reinsercion
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
KEY = ["CODUSU", "NRO_HOGAR", "COMPONENTE"]
COLS = KEY + ["ESTADO", "CAT_OCUP", "PP04D_COD", "PP11B_COD", "CH06", "PONDERA"]


def calif(serie):
    """Último dígito de la calificación de la tarea (1 prof .. 4 no calif)."""
    s = pd.to_numeric(serie, errors="coerce")
    d = s.dropna().astype("int64").astype(str).str.zfill(5).str[-1].astype(int)
    return d.where(d.isin([1, 2, 3, 4])).reindex(serie.index)


def main() -> None:
    archivos = sorted(p for p in RAW.glob("eph_individual_*T*.csv") if "TODO" not in p.name)
    pares = []
    for a, b in zip(archivos, archivos[1:]):
        # solo pares consecutivos del mismo año o cambio de año contiguo
        ta = pd.read_csv(a, usecols=COLS, low_memory=False)
        tb = pd.read_csv(b, usecols=COLS, low_memory=False)
        m = ta.merge(tb, on=KEY, suffixes=("_t", "_t1"))
        deso = m[m["ESTADO_t"] == 2].copy()        # desocupados en t
        if len(deso):
            pares.append(deso)
    d = pd.concat(pares, ignore_index=True)
    d["edad"] = pd.to_numeric(d["CH06_t"], errors="coerce")
    d = d[d["edad"].between(25, 70)]
    print(f"Desocupados seguidos al trimestre siguiente: {len(d):,}")

    d["reinserta"] = (d["ESTADO_t1"] == 1).astype(int)
    d["a_inactivo"] = (d["ESTADO_t1"] == 3).astype(int)

    bins = [25, 35, 45, 55, 71]; labels = ["25-34", "35-44", "45-54", "55-70"]
    d["franja"] = pd.cut(d["edad"], bins=bins, labels=labels, right=False)

    print("\n== Destino al trimestre siguiente por edad (%) ==")
    tab = d.groupby("franja", observed=True).agg(
        n=("reinserta", "size"),
        reinsercion=("reinserta", lambda x: 100 * x.mean()),
        a_inactividad=("a_inactivo", lambda x: 100 * x.mean()),
    )
    print(tab.round(1).to_string())

    # Descenso ocupacional entre los que se reinsertan
    rein = d[d["reinserta"] == 1].copy()
    rein["calif_prev"] = calif(rein["PP11B_COD_t"])      # último empleo (cuando desoc.)
    rein["calif_nuevo"] = calif(rein["PP04D_COD_t1"])    # nuevo empleo
    val = rein.dropna(subset=["calif_prev", "calif_nuevo"])
    val = val.assign(desciende=(val["calif_nuevo"] > val["calif_prev"]).astype(int))
    print("\n== Descenso ocupacional al reinsertarse por edad (%) ==")
    print(val.groupby("franja", observed=True)["desciende"].agg(
        n="size", baja=lambda x: 100 * x.mean()).round(1).to_string())


if __name__ == "__main__":
    main()
