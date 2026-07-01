"""Decodifica la EPH y genera el EDA de brechas por edad y sexo.

Foco (hipótesis del trabajo): a partir de ~58 años sube la desocupación y cae
el ingreso relativo, controlando luego por calificación.

- Tasas con ponderador poblacional (PONDERA).
- Ingreso comparable entre trimestres usando ingreso RELATIVO a la mediana del
  trimestre (elimina la inflación 2016-2024).

Uso:
    uv run python -m src.data.eda_eph
Salida:
    data/processed/eph_analitico.parquet   (base limpia para el modelo)
    reports/figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)

USECOLS = ["REGION", "AGLOMERADO", "PONDERA", "CH04", "CH06", "NIVEL_ED",
           "ESTADO", "CAT_OCUP", "PP07H", "P21", "PONDIIO", "P47T", "PONDII",
           "anio", "trimestre"]

SEXO = {1: "Varón", 2: "Mujer"}
NIVEL = {1: "Primario inc.", 2: "Primario comp.", 3: "Secundario inc.",
         4: "Secundario comp.", 5: "Superior inc.", 6: "Superior comp.",
         7: "Sin instrucción", 9: "Ns/Nr"}
ESTADO = {1: "Ocupado", 2: "Desocupado", 3: "Inactivo", 4: "Menor"}


def cargar() -> pd.DataFrame:
    """Carga trimestre por trimestre (poco RAM) y apila. Calcula el ingreso
    relativo DENTRO de cada trimestre antes de concatenar (elimina inflación)."""
    archivos = sorted(p for p in RAW_DIR.glob("eph_individual_*T*.csv")
                      if "TODO" not in p.name)
    partes = []
    for f in archivos:
        d = pd.read_csv(f, usecols=USECOLS, low_memory=False)
        for c in ["REGION", "AGLOMERADO", "PONDERA", "CH04", "CH06", "NIVEL_ED",
                  "ESTADO", "CAT_OCUP", "PP07H", "P21", "PONDIIO", "P47T",
                  "PONDII"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["sexo"] = d["CH04"].map(SEXO)
        d["edad"] = pd.to_numeric(d["CH06"], errors="coerce")
        d["nivel_ed"] = d["NIVEL_ED"].map(NIVEL)
        d["cond"] = d["ESTADO"].map(ESTADO)
        d = d[(d["edad"].between(14, 90)) & d["sexo"].notna()]
        # ingreso relativo a la mediana de ocupados de ESTE trimestre
        med = d.loc[(d["ESTADO"] == 1) & (d["P21"] > 0), "P21"].median()
        d["ingreso_rel"] = np.where((d["ESTADO"] == 1) & (d["P21"] > 0),
                                    d["P21"] / med, np.nan)
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    return df


def tasa_desocupacion(df, by):
    """Desocupados / PEA, ponderado por PONDERA."""
    pea = df[df["ESTADO"].isin([1, 2])].copy()
    pea["w_deso"] = pea["PONDERA"].where(pea["ESTADO"] == 2, 0)
    g = pea.groupby(by).apply(
        lambda x: 100 * x["w_deso"].sum() / x["PONDERA"].sum()
    )
    return g


def main() -> None:
    df = cargar()
    print(f"Filas analizables: {len(df):,}")

    # --- Ingreso relativo a la mediana del trimestre (elimina inflación) ---
    ocup = df[(df["ESTADO"] == 1) & (df["P21"] > 0)].copy()
    med = ocup.groupby(["anio", "trimestre"])["P21"].transform("median")
    ocup["ingreso_rel"] = ocup["P21"] / med

    # ============ FIGURA 1: desocupación por edad y sexo ============
    edades = range(20, 71)
    plt.figure(figsize=(10, 5.5))
    for sx, col in [("Varón", "#4C72B0"), ("Mujer", "#C44E52")]:
        sub = df[df["sexo"] == sx]
        serie = tasa_desocupacion(sub, "edad").reindex(edades)
        serie = serie.rolling(3, center=True, min_periods=1).mean()  # suavizado
        plt.plot(list(edades), serie.values, label=sx, color=col, lw=2)
    plt.axvline(60, ls="--", c="#C44E52", alpha=.6); plt.text(60.2, plt.ylim()[1]*.92, "Jub. mujer (60)", color="#C44E52")
    plt.axvline(65, ls="--", c="#4C72B0", alpha=.6); plt.text(65.2, plt.ylim()[1]*.85, "Jub. varón (65)", color="#4C72B0")
    plt.axvspan(58, 90, color="gray", alpha=.07)
    plt.xlabel("Edad"); plt.ylabel("Tasa de desocupación (%)")
    plt.title("Desocupación por edad y sexo — EPH 2016–2024 (ponderada)")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG/"desocupacion_edad_sexo.png", dpi=120); plt.close()

    # ============ FIGURA 2: ingreso relativo por edad y sexo ============
    plt.figure(figsize=(10, 5.5))
    for sx, col in [("Varón", "#4C72B0"), ("Mujer", "#C44E52")]:
        sub = ocup[ocup["sexo"] == sx]
        serie = sub.groupby("edad").apply(
            lambda x: np.average(x["ingreso_rel"], weights=x["PONDIIO"])
        ).reindex(edades).rolling(3, center=True, min_periods=1).mean()
        plt.plot(list(edades), serie.values, label=sx, color=col, lw=2)
    plt.axvline(60, ls="--", c="#C44E52", alpha=.6)
    plt.axvline(65, ls="--", c="#4C72B0", alpha=.6)
    plt.axvspan(58, 90, color="gray", alpha=.07)
    plt.axhline(1.0, c="k", lw=.7, alpha=.5)
    plt.xlabel("Edad"); plt.ylabel("Ingreso relativo (1 = mediana del trimestre)")
    plt.title("Ingreso laboral relativo por edad y sexo — EPH 2016–2024")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG/"ingreso_edad_sexo.png", dpi=120); plt.close()

    # ============ FIGURA 3: desocupación por franja etaria y sexo ============
    bins = [14, 30, 45, 58, 65, 90]
    labels = ["14-29", "30-44", "45-57", "58-64", "65+"]
    df["franja"] = pd.cut(df["edad"], bins=bins, labels=labels, right=False)
    tab = df.groupby(["franja", "sexo"]).apply(
        lambda x: 100 * x.loc[x.ESTADO == 2, "PONDERA"].sum() /
                  x.loc[x.ESTADO.isin([1, 2]), "PONDERA"].sum()
    ).unstack()
    ax = tab.plot(kind="bar", figsize=(9, 5.5), color={"Varón": "#4C72B0", "Mujer": "#C44E52"})
    ax.set_ylabel("Tasa de desocupación (%)"); ax.set_xlabel("Franja etaria")
    ax.set_title("Desocupación por franja etaria y sexo — EPH 2016–2024")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIG/"desocupacion_franjas.png", dpi=120); plt.close()

    # ============ Números clave (consola) ============
    print("\n== Tasa de desocupación por franja y sexo (%) ==")
    print(tab.round(1).to_string())

    print("\n== Ingreso relativo mediano por franja y sexo ==")
    ocup["franja"] = pd.cut(ocup["edad"], bins=bins, labels=labels, right=False)
    pivot = ocup.groupby(["franja", "sexo"])["ingreso_rel"].median().unstack()
    print(pivot.round(2).to_string())

    # --- Base analítica para el modelo (subconjunto útil) ---
    cols = ["anio", "trimestre", "REGION", "AGLOMERADO", "sexo", "edad",
            "nivel_ed", "cond", "CAT_OCUP", "PP07H", "P21", "P47T",
            "PONDERA", "PONDIIO", "PONDII", "ESTADO"]
    df[cols].to_parquet(PROC/"eph_analitico.parquet", index=False)
    print(f"\nBase analítica guardada: {PROC/'eph_analitico.parquet'} ({len(df):,} filas)")


if __name__ == "__main__":
    main()
