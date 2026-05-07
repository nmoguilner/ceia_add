# Prompt para Claude Code

Copiá y pegá el bloque siguiente en Claude Code (estando en `~/Downloads/nyc_taxi_eda/`). Está pensado para que retome este trabajo desde donde quedó, con red completa y sin las limitaciones del sandbox.

---

## Contexto

Soy Gus. Estoy haciendo un EDA del dataset oficial **NYC Yellow Taxi Trip Records (Enero 2024)** de la TLC para un proyecto académico (FIUBA). El proyecto vive en `~/Downloads/nyc_taxi_eda/`. Mis preferencias: Python, EDA, análisis financiero, Linux/macOS. Respondeme en español con tono técnico-profesional.

## Estado actual del proyecto

```
nyc_taxi_eda/
├── eda_taxis.py        # Script con descarga + fallback sintético + EDA + 3 plots
├── eda_taxis.ipynb     # Notebook (28 celdas): EDA + Pearson/Kendall + entropía + MI
├── PRESENTACION.md     # Marp, 2 slides
├── requirements.txt    # Pinned deps
├── setup.sh            # Bootstrap del venv 'fiuba' + kernel Jupyter
├── README.md
├── .gitignore          # Ignora fiuba/ y *.parquet
└── assets/             # 7 PNGs ya generados
```

El `.py` y el notebook **ya corrieron** y funcionan, pero contra una **muestra sintética** (100k filas) porque el sandbox anterior no tenía acceso al CDN de la TLC. Vos sí tenés red.

## Lo que necesito que hagas

Trabajá secuencialmente. Confirmame con outputs reales (no inventes números).

### Paso 1 — Setup del venv

1. `cd ~/Downloads/nyc_taxi_eda`
2. Si no existe `fiuba/`, ejecutá `./setup.sh` (o reproducí los pasos: `python3 -m venv fiuba`, activar, `pip install -r requirements.txt`, `python -m ipykernel install --user --name nyc_taxi_eda_fiuba --display-name "Python (fiuba)"`).
3. Activá el venv y dejalo activado para todo lo siguiente.
4. Verificá: `which python` debe apuntar a `~/Downloads/nyc_taxi_eda/fiuba/bin/python`.

### Paso 2 — Descargar el dataset oficial

```bash
curl -L -o yellow_tripdata_2024-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

Verificá que el archivo pese ~48 MB y que `python -c "import pyarrow.parquet as pq; print(pq.read_metadata('yellow_tripdata_2024-01.parquet'))"` devuelva ~2.96M filas y 19 columnas.

### Paso 3 — Correr el EDA contra los datos reales

1. Ejecutá `python eda_taxis.py`. Debe imprimir `Fuente de datos: REAL` (no `SINTETICO`).
2. Re-ejecutá el notebook end-to-end:
   ```bash
   jupyter nbconvert --to notebook --execute eda_taxis.ipynb \
     --output eda_taxis.ipynb --ExecutePreprocessor.timeout=300
   jupyter nbconvert --to html eda_taxis.ipynb
   ```
3. Re-extraé los gráficos clave a `assets/` (sobrescribiendo los actuales generados con sintético).

### Paso 4 — Hallazgos sobre datos reales

Reportame en un bloque corto (no más de 200 palabras) **qué cambia respecto del análisis sintético**:

- Estadísticas descriptivas reales (duración media, tarifa media, distancia, total).
- Top 5 zonas PU/DO reales (con nombres de barrios — usá el lookup oficial: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv).
- **Pearson vs Kendall:** ¿cuán cerca están de los valores sintéticos (Pearson ~0.92 distancia↔tarifa, Kendall ~0.65)?
- **Entropía normalizada:** ¿`store_and_fwd_flag` y `RatecodeID` están tan concentrados como en sintético (H_norm ~0.20)?
- **Mutual Information:** este es el cambio MÁS importante. En sintético, `MI(PULocationID, fare_amount) ≈ 0.005` porque las zonas se generaron independientes. En datos reales debería subir significativamente (zonas predicen costo: aeropuertos JFK/LGA, Manhattan vs outer boroughs). Mostrame la matriz nueva.

### Paso 5 — Mejoras opcionales

Si todo lo anterior funciona, proponeme (sin ejecutar todavía):

1. Agregar el lookup de `taxi_zone_lookup.csv` para que las zonas top se reporten con nombre de barrio en vez de ID numérico.
2. Una sección de feature engineering para predicción de tarifa (haversine si hubiera coords, hora pico, día laboral vs fin de semana).
3. Sugerencias de modelo (lineal vs GBM) basadas en la brecha |Pearson|−|Kendall| que veas en los datos reales.

## Reglas

- Usá **siempre** el venv `fiuba` activado, nunca el Python del sistema.
- No reescribas archivos ya válidos sin necesidad — preferí `Edit` sobre `Write`.
- Si falla la descarga del parquet (ej. 403 transitorio), reintentá con `--retry 3 --retry-delay 2`.
- Si el `eda_taxis.py` necesita ajustes para procesar el archivo real eficientemente (~3M filas), hacelo (ej. leer columnas necesarias con `pd.read_parquet(..., columns=[...])`).
- No inventes valores: todos los números reportados deben venir de ejecuciones reales.

Empezá por el Paso 1 y andá confirmándome cada paso antes de seguir.
