# Trabajo Práctico Integrador — Aprendizaje de Máquina

### Carrera de Especialización en Inteligencia Artificial (CEIA-FIUBA) · Aprendizaje de Máquina · 3er Bimestre 2026

## Docentes

- Dr. Ing. Facundo Lucianna
- Esp. Lic. María Carina Roldán

## Autores

- Fioranelli, Rocío
- Lerner, Federico
- Makk, Azul
- Moguilner Reh, Nicolás

---

## Dataset

Viajes de **NYC Yellow Taxi** de marzo de 2026, publicados por la NYC Taxi & Limousine Commission (TLC): ~3,95 millones de registros con datos de recogida/destino, distancia, duración, tarifa y forma de pago.

- **Fuente:** [TLC Trip Record Data](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml)
- **Target:** `tip_amount` (propina en USD). El problema se plantea como **regresión** sobre el universo de viajes pagados **con tarjeta** (la propina en efectivo no se registra en el dataset, por lo que queda fuera por diseño, no por limitación técnica).

## Estructura del repositorio

```
yellow_taxi/
├── tp_grupal_yellowtaxi_target_propina.ipynb   # Notebook principal (ver secciones abajo)
├── dataset/
│   ├── yellow_tripdata_2026-03.parquet          # Dataset crudo (TLC, marzo 2026)
│   └── taxi_zone_lookup.csv                     # Tabla oficial de zonas TLC (PULocationID/DOLocationID -> barrio/borough)
├── models/                                      # Modelos entrenados exportados con joblib (se generan al correr el notebook, no versionados)
├── pyproject.toml / uv.lock                     # Definición y lockfile del entorno (gestionado con uv)
└── .gitignore
```

### `tp_grupal_yellowtaxi_target_propina.ipynb`

Notebook único que cubre el flujo completo del trabajo práctico:

1. **Exploración y comprensión de los datos** — estructura del dataset, estadística robusta vs. no robusta, tipos de faltantes (MCAR/MAR/MNAR), detección de errores y de outliers (IQR / z-score robusto).
2. **Técnicas de visualización** — distribuciones, variables categóricas, patrones temporales, correlaciones, relaciones multivariadas y dimensión espacial (zonas TLC).
3. **Planteo del problema de ML supervisado** — definición del universo de análisis y exploración del target.
4. **Preprocesamiento y limpieza** — filtrado a pagos con tarjeta, eliminación de registros físicamente imposibles, split train/test (80/20, `random_state=42`, hecho **antes** de transformar para evitar data leakage), tratamiento de outliers por winsorización.
5. **Feature engineering** — variables derivadas (`speed_mph`, `fare_per_mile`, indicadores de aeropuerto/hora pico), codificación cíclica de variables temporales, encoding de variables categóricas (dummies y target encoding fit-en-train), escalado (`StandardScaler`).
6. **Reducción de dimensionalidad (PCA)** — usado solo como herramienta de diagnóstico; se concluye que no conviene aplicarlo como preprocesamiento del modelo (mezcla señal geográfica y tarifaria, y es ciego al target).
7. **Resumen del dataset listo para modelado** — chequeos de integridad de `X_train_enc_esc` / `X_test_enc_esc` / `y_train` / `y_test`.
8. **Modelado y evaluación** — comparación de regresores (Regresión Lineal, Ridge, Random Forest, Hist Gradient Boosting), ajuste de hiperparámetros del mejor candidato con `RandomizedSearchCV`, diagnóstico del modelo final (predicho vs. real, residuos, importancia de features) y persistencia del modelo con `joblib`.
9. **Conclusiones** — síntesis del problema, la calidad de los datos, las variables candidatas, el análisis de PCA y los resultados del modelado.

## Cómo ejecutar

El entorno se gestiona con [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # crea .venv e instala las dependencias fijadas en uv.lock
uv run jupyter lab       # abre Jupyter Lab con el kernel del proyecto
```

Luego abrir `tp_grupal_yellowtaxi_target_propina.ipynb` y ejecutar las celdas en orden. El dataset crudo ya está incluido en `dataset/`, no requiere descarga adicional.
