"""Tasa de empleo e inactividad por edad y sexo (ponderadas).

Estas métricas capturan la EXPULSIÓN del mercado a edades altas, que la tasa
de desocupación esconde (efecto del trabajador desalentado).

Uso:
    uv run python -m src.data.empleo_curvas
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROC = Path("data/processed/eph_analitico.parquet")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)


def tasa(df, estado_objetivo, by="edad"):
    """% de la población (de cada grupo) que está en `estado_objetivo`, ponderado."""
    df = df.copy()
    df["w_obj"] = df["PONDERA"].where(df["ESTADO"].isin(estado_objetivo), 0)
    return df.groupby(by, observed=True).apply(
        lambda x: 100 * x["w_obj"].sum() / x["PONDERA"].sum(), include_groups=False
    )


def main() -> None:
    df = pd.read_parquet(PROC)
    edades = range(20, 76)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for sx, col in [("Varón", "#4C72B0"), ("Mujer", "#C44E52")]:
        sub = df[df["sexo"] == sx]
        emp = tasa(sub, [1]).reindex(edades).rolling(3, center=True, min_periods=1).mean()
        ina = tasa(sub, [3]).reindex(edades).rolling(3, center=True, min_periods=1).mean()
        ax[0].plot(list(edades), emp.values, label=sx, color=col, lw=2)
        ax[1].plot(list(edades), ina.values, label=sx, color=col, lw=2)

    for a, titulo, ylab in [(ax[0], "Tasa de empleo", "% de la población"),
                            (ax[1], "Tasa de inactividad", "% de la población")]:
        a.axvline(60, ls="--", c="#C44E52", alpha=.6)
        a.axvline(65, ls="--", c="#4C72B0", alpha=.6)
        a.axvspan(58, 76, color="gray", alpha=.07)
        a.set_title(titulo); a.set_xlabel("Edad"); a.set_ylabel(ylab); a.legend()
    fig.suptitle("Empleo e inactividad por edad y sexo — EPH 2016–2024", y=1.02)
    plt.tight_layout(); plt.savefig(FIG/"empleo_inactividad.png", dpi=120, bbox_inches="tight"); plt.close()

    # Números clave por franja
    bins = [25, 45, 58, 65, 76]; labels = ["25-44", "45-57", "58-64", "65-75"]
    df = df[df["edad"].between(25, 75)].copy()
    df["franja"] = pd.cut(df["edad"], bins=bins, labels=labels, right=False)
    emp = df.groupby(["franja", "sexo"], observed=True).apply(
        lambda x: 100 * x.loc[x.ESTADO == 1, "PONDERA"].sum() / x["PONDERA"].sum(),
        include_groups=False).unstack()
    print("== Tasa de empleo por franja y sexo (%) ==")
    print(emp.round(1).to_string())
    print(f"\nFigura: {FIG/'empleo_inactividad.png'}")


if __name__ == "__main__":
    main()
