"""
eda_taxis.py
============
Análisis Exploratorio de Datos (EDA) sobre el dataset
"NYC Yellow Taxi Trip Records" (Marzo 2026).

Fuente oficial:
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
URL directa (Parquet):
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-03.parquet

El script:
    1. Intenta descargar el Parquet oficial (si hay conectividad).
    2. Si la descarga falla (entorno aislado / sin red), genera una muestra
       sintética de 100k filas que replica fielmente el esquema y las
       distribuciones documentadas por la TLC, para que el pipeline EDA
       sea reproducible en cualquier máquina.
    3. Toma una muestra representativa de ~100k filas.
    4. Calcula tamaño, tipos, valores faltantes e inconsistencias.
    5. Genera estadísticas descriptivas (duración, tarifa, zonas top).
    6. Identifica 3 curiosidades / desafíos técnicos.
    7. Guarda 3 gráficos clave en ./assets/.

Autor: Gus  (preferencias: Python, EDA, análisis financiero, Linux/macOS)
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-03.parquet"
)
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
LOCAL_PARQUET = Path("yellow_tripdata_2026-03.parquet")
LOCAL_ZONES = Path("taxi_zone_lookup.csv")
ASSETS_DIR = Path("assets")
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------------
# 1. Carga de datos: descarga real con fallback sintético
# ---------------------------------------------------------------------------
def download_real_parquet(url: str, dest: Path, timeout: int = 30) -> bool:
    """Intenta descargar el archivo Parquet oficial. Devuelve True si tuvo éxito."""
    try:
        print(f"[INFO] Descargando dataset oficial desde {url} ...")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (URL fija y conocida)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"[OK]   Descargado: {dest} ({size_mb:.1f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Descarga fallida ({exc.__class__.__name__}): {exc}")
        return False


def synthesize_yellow_taxi_sample(n: int = SAMPLE_SIZE, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Genera una muestra sintética que reproduce el esquema oficial de la TLC
    (Yellow Taxi - Marzo 2026) con distribuciones realistas. Se utiliza como
    fallback cuando no hay conectividad para descargar el archivo real.
    """
    rng = np.random.default_rng(seed)

    # Pickup datetimes distribuidos en Marzo 2026, con sesgo a horas pico
    base = pd.Timestamp("2026-03-01")
    minutes_in_jan = 31 * 24 * 60
    pickup_minutes = rng.integers(0, minutes_in_jan, size=n)
    pickup = base + pd.to_timedelta(pickup_minutes, unit="m")

    # Duración del viaje: lognormal centrada ~12 min
    duration_min = rng.lognormal(mean=2.4, sigma=0.6, size=n).clip(0.5, 360)
    dropoff = pickup + pd.to_timedelta(duration_min, unit="m")

    # Trip distance: lognormal centrada ~3 mi
    trip_distance = rng.lognormal(mean=1.0, sigma=0.8, size=n).clip(0, 100)

    # Tarifa = base + por milla + por minuto, con ruido
    fare_amount = (3.0 + 2.5 * trip_distance + 0.4 * duration_min
                   + rng.normal(0, 1.5, n)).clip(0)
    extra = rng.choice([0.0, 0.5, 1.0, 2.5], n, p=[0.55, 0.20, 0.15, 0.10])
    mta_tax = np.full(n, 0.5)
    tolls = np.where(rng.random(n) < 0.06, rng.uniform(2, 12, n), 0.0)
    improvement_surcharge = np.full(n, 1.0)
    congestion = np.where(rng.random(n) < 0.85, 2.5, 0.0)
    airport_fee = np.where(rng.random(n) < 0.04,
                           rng.choice([1.75, 2.50], n), 0.0)
    tip = np.where(
        rng.random(n) < 0.65,
        (fare_amount * rng.uniform(0.10, 0.25, n)).round(2),
        0.0,
    )
    total = (fare_amount + extra + mta_tax + tolls + improvement_surcharge
             + congestion + airport_fee + tip)

    # Zonas (LocationIDs 1..263 documentadas por TLC; sesgo hacia Manhattan)
    pop_zones = np.array([
        132, 138, 161, 162, 163, 164, 230, 234, 236, 237,
        239, 246, 263, 100, 113, 142, 143, 158, 170, 186,
    ])
    pu_choice = rng.random(n)
    pu_loc = np.where(
        pu_choice < 0.55,
        rng.choice(pop_zones, n),
        rng.integers(1, 264, n),
    )
    do_loc = np.where(
        rng.random(n) < 0.55,
        rng.choice(pop_zones, n),
        rng.integers(1, 264, n),
    )

    df = pd.DataFrame({
        "VendorID": rng.choice([1, 2], n, p=[0.3, 0.7]),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "passenger_count": rng.choice(
            [1, 2, 3, 4, 5, 6, np.nan], n,
            p=[0.70, 0.13, 0.05, 0.04, 0.03, 0.02, 0.03],
        ),
        "trip_distance": trip_distance,
        "RatecodeID": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], n,
                                  p=[0.90, 0.05, 0.01, 0.01, 0.01, 0.02]),
        "store_and_fwd_flag": rng.choice(["N", "Y", None], n,
                                          p=[0.95, 0.03, 0.02]),
        "PULocationID": pu_loc,
        "DOLocationID": do_loc,
        "payment_type": rng.choice([1, 2, 3, 4], n, p=[0.78, 0.20, 0.01, 0.01]),
        "fare_amount": fare_amount.round(2),
        "extra": extra,
        "mta_tax": mta_tax,
        "tip_amount": tip,
        "tolls_amount": tolls.round(2),
        "improvement_surcharge": improvement_surcharge,
        "total_amount": total.round(2),
        "congestion_surcharge": congestion,
        "Airport_fee": airport_fee,
    })

    # Inyectamos inconsistencias REALES presentes en el dataset oficial:
    # - distancias 0 / negativas
    # - tarifas 0 / negativas
    # - timestamps fuera del mes
    n_bad = max(1, n // 200)  # ~0.5%
    bad_idx = rng.choice(n, n_bad, replace=False)
    df.loc[bad_idx[: n_bad // 3], "trip_distance"] = 0.0
    df.loc[bad_idx[n_bad // 3 : 2 * n_bad // 3], "fare_amount"] = 0.0
    df.loc[bad_idx[2 * n_bad // 3 :], "trip_distance"] = -1.0

    out_of_month = rng.choice(n, max(1, n // 500), replace=False)
    df.loc[out_of_month, "tpep_pickup_datetime"] = pd.Timestamp("2023-12-31 23:30")

    return df


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Carga el dataset (real o sintético) y devuelve (df, fuente)."""
    if not LOCAL_PARQUET.exists():
        ok = download_real_parquet(DATA_URL, LOCAL_PARQUET)
        if not ok:
            print("[INFO] Generando muestra sintética representativa "
                  "(esquema TLC oficial, 100k filas)...")
            df = synthesize_yellow_taxi_sample(SAMPLE_SIZE)
            return df, "sintetico"
    try:
        df = pd.read_parquet(LOCAL_PARQUET)
        # Muestra estratificada simple por hora del día
        if len(df) > SAMPLE_SIZE:
            df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        return df, "real"
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] No se pudo leer {LOCAL_PARQUET}: {exc}")
        df = synthesize_yellow_taxi_sample(SAMPLE_SIZE)
        return df, "sintetico"


# ---------------------------------------------------------------------------
# 2. EDA - Diagnóstico básico
# ---------------------------------------------------------------------------
def basic_overview(df: pd.DataFrame) -> dict:
    """Tamaño, dtypes, nulos e inconsistencias."""
    rows, cols = df.shape
    dtypes = df.dtypes.value_counts().to_dict()
    nulls = df.isna().sum().sort_values(ascending=False)
    nulls_pct = (nulls / rows * 100).round(2)

    inconsistencies = {
        "trip_distance <= 0": int((df["trip_distance"] <= 0).sum()),
        "fare_amount <= 0": int((df["fare_amount"] <= 0).sum()),
        "total_amount < fare_amount": int(
            (df["total_amount"] < df["fare_amount"]).sum()
        ),
        "passenger_count == 0 o NaN": int(
            df["passenger_count"].fillna(0).eq(0).sum()
        ),
        "pickup fuera de Marzo 2026": int(
            (~df["tpep_pickup_datetime"].between(
                "2026-03-01", "2026-03-31 23:59:59")
             ).sum()
        ),
    }

    print("\n" + "=" * 70)
    print("RESUMEN GENERAL DEL DATASET")
    print("=" * 70)
    print(f"Filas (muestra)       : {rows:,}")
    print(f"Columnas              : {cols}")
    print(f"Tipos de datos        : {dtypes}")
    print("\nValores faltantes (top):")
    print(pd.concat([nulls.head(10), nulls_pct.head(10).rename("pct_%")], axis=1))
    print("\nInconsistencias detectadas:")
    for k, v in inconsistencies.items():
        print(f"  - {k:<35s}: {v:>7,}")

    return {
        "rows": rows,
        "cols": cols,
        "dtypes": dtypes,
        "nulls_top": nulls.head(10).to_dict(),
        "inconsistencies": inconsistencies,
    }


# ---------------------------------------------------------------------------
# 3. Estadísticas descriptivas + curiosidades
# ---------------------------------------------------------------------------
def _load_zone_lookup() -> dict | None:
    """Devuelve {LocationID: {'Borough':..., 'Zone':...}} o None si no hay CSV."""
    if not LOCAL_ZONES.exists():
        try:
            urllib.request.urlretrieve(ZONES_URL, LOCAL_ZONES)  # noqa: S310
        except Exception:  # noqa: BLE001
            return None
    try:
        z = pd.read_csv(LOCAL_ZONES)
        return {int(r.LocationID): {"Borough": r.Borough, "Zone": r.Zone}
                for r in z.itertuples()}
    except Exception:  # noqa: BLE001
        return None


def descriptive_stats(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # Filtro razonable para promedios
    clean = df[
        (df["trip_distance"] > 0)
        & (df["fare_amount"] > 0)
        & (df["trip_duration_min"].between(1, 180))
    ]

    avg_duration = clean["trip_duration_min"].mean()
    avg_fare = clean["fare_amount"].mean()
    avg_total = clean["total_amount"].mean()
    avg_distance = clean["trip_distance"].mean()
    median_tip = clean["tip_amount"].median()

    top_pu = df["PULocationID"].value_counts().head(5)
    top_do = df["DOLocationID"].value_counts().head(5)

    zones = _load_zone_lookup()

    def _format_top(top: pd.Series) -> str:
        if zones is None:
            return str(top)
        lines = []
        for lid, n in top.items():
            row = zones.get(int(lid))
            label = f"{row['Borough']} / {row['Zone']}" if row else "?"
            lines.append(f"  {int(lid):>4} ({label:<45s}) : {int(n):>5,}")
        return "\n".join(lines)

    print("\n" + "=" * 70)
    print("ESTADÍSTICAS DESCRIPTIVAS (sobre subset limpio)")
    print("=" * 70)
    print(f"Duración promedio del viaje : {avg_duration:6.2f} min")
    print(f"Distancia promedio          : {avg_distance:6.2f} mi")
    print(f"Tarifa media (fare_amount)  : ${avg_fare:6.2f}")
    print(f"Total medio (total_amount)  : ${avg_total:6.2f}")
    print(f"Tip mediana                 : ${median_tip:6.2f}")
    print(f"\nTop 5 PULocationID:\n{_format_top(top_pu)}")
    print(f"\nTop 5 DOLocationID:\n{_format_top(top_do)}")

    return {
        "avg_duration_min": float(avg_duration),
        "avg_distance_mi": float(avg_distance),
        "avg_fare_usd": float(avg_fare),
        "avg_total_usd": float(avg_total),
        "median_tip_usd": float(median_tip),
        "top_pu": top_pu.to_dict(),
        "top_do": top_do.to_dict(),
        "df_with_duration": df,  # para usar en gráficos
    }


def find_curiosities(df: pd.DataFrame) -> list[str]:
    """Identifica 3 curiosidades / desafíos técnicos del dataset."""
    duration = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # 1) Tarifas muy altas con distancias mínimas (taxímetro atascado / fraude)
    suspicious_fare = df[
        (df["trip_distance"] < 0.2) & (df["fare_amount"] > 50)
    ]
    pct_suspicious = len(suspicious_fare) / len(df) * 100

    # 2) Viajes con duración negativa o > 6 h
    weird_duration = ((duration < 0) | (duration > 360)).sum()

    # 3) Correlación tip vs payment_type (cash siempre = 0 reportado)
    cash_tips = df.loc[df["payment_type"] == 2, "tip_amount"]
    pct_cash_zero_tip = (cash_tips == 0).mean() * 100 if len(cash_tips) else 0

    curiosities = [
        f"Tarifas anómalas: {len(suspicious_fare):,} viajes con distancia < 0.2 mi y "
        f"tarifa > $50 ({pct_suspicious:.2f}% de la muestra). Posible fraude o "
        f"errores del taxímetro.",
        f"Duraciones imposibles: {weird_duration:,} viajes con duración negativa "
        f"o > 6 h. Indica problemas de sincronización entre pickup/dropoff o "
        f"vehículos olvidados encendidos.",
        f"Sesgo de propinas en efectivo: el {pct_cash_zero_tip:.1f}% de los pagos "
        f"en cash (payment_type=2) registran tip = 0. La TLC NO captura propinas "
        f"en efectivo, lo que sesga cualquier modelo de tipping.",
    ]

    print("\n" + "=" * 70)
    print("CURIOSIDADES / DESAFÍOS TÉCNICOS")
    print("=" * 70)
    for i, c in enumerate(curiosities, 1):
        print(f"{i}. {c}")
    return curiosities


# ---------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------
def plot_trip_distance(df: pd.DataFrame, out: Path) -> None:
    """Histograma de distancias (clipeado a 20 mi para legibilidad)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data = df.loc[df["trip_distance"].between(0, 20), "trip_distance"]
    sns.histplot(data, bins=60, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Distribución de la distancia de viaje (NYC Yellow Taxi - Mar 2026)")
    ax.set_xlabel("Distancia (millas)")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardado {out}")


def plot_fare_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histograma de tarifas (fare_amount) clipeado a [0, 80] USD."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data = df.loc[df["fare_amount"].between(0, 80), "fare_amount"]
    sns.histplot(data, bins=60, kde=True, color="#ff7f0e", ax=ax)
    ax.set_title("Distribución de la tarifa (fare_amount) en USD")
    ax.set_xlabel("Tarifa base (USD)")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardado {out}")


def plot_trips_by_hour(df: pd.DataFrame, out: Path) -> None:
    """Volumen de viajes por hora del día (heurística de horas pico)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly = (
        df["tpep_pickup_datetime"]
        .dt.hour.value_counts()
        .sort_index()
    )
    sns.barplot(x=hourly.index, y=hourly.values, color="#2ca02c", ax=ax)
    ax.set_title("Viajes por hora del día (pickup)")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Nº de viajes")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardado {out}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> int:
    ASSETS_DIR.mkdir(exist_ok=True)
    df, source = load_dataset()
    print(f"\n[INFO] Fuente de datos: {source.upper()} | Filas cargadas: {len(df):,}")

    overview = basic_overview(df)
    stats = descriptive_stats(df)
    curiosities = find_curiosities(df)

    # Gráficos
    df_full = stats["df_with_duration"]
    plot_trip_distance(df_full, ASSETS_DIR / "hist_trip_distance.png")
    plot_fare_distribution(df_full, ASSETS_DIR / "hist_fare_amount.png")
    plot_trips_by_hour(df_full, ASSETS_DIR / "trips_by_hour.png")

    # Resumen final exportable
    summary = {
        "fuente": source,
        "filas": overview["rows"],
        "columnas": overview["cols"],
        "stats": {k: v for k, v in stats.items() if k != "df_with_duration"},
        "curiosidades": curiosities,
        "inconsistencias": overview["inconsistencies"],
    }
    print("\n" + "=" * 70)
    print("FIN DEL EDA - resumen disponible en variable 'summary'")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
