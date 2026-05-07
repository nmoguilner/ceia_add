# NYC Yellow Taxi Trip Records — EDA (Enero 2024)

Proyecto reproducible de **Análisis Exploratorio de Datos** sobre el dataset
oficial de la TLC.

```
nyc_taxi_eda/
├── eda_taxis.py        # Script principal (EDA + gráficos)
├── eda_taxis.ipynb     # Notebook con secciones extendidas (Pearson/Kendall, entropía, MI)
├── PRESENTACION.md     # Diapositivas Marp (2 slides)
├── requirements.txt    # Dependencias pinned
├── setup.sh            # Bootstrap del venv 'fiuba' + kernel Jupyter
├── assets/             # Gráficos generados (PNG)
└── README.md
```

---

## 1 · Setup del venv `fiuba`

```bash
cd ~/Downloads/nyc_taxi_eda
chmod +x setup.sh
./setup.sh
```

`setup.sh` crea el venv `fiuba/` dentro de la carpeta, instala dependencias
y registra el kernel **"Python (fiuba)"** en Jupyter.

> **Nada se instala en el Python del sistema.** Todo queda aislado en `./fiuba`.

## 2 · Activar el venv

```bash
source fiuba/bin/activate    # macOS / Linux
python --version             # debe ser el del venv
which python                 # ~/Downloads/nyc_taxi_eda/fiuba/bin/python
```

## 3 · Correr el EDA

```bash
# Modo script: imprime resumen y guarda gráficos en assets/
python eda_taxis.py

# Modo notebook: abre Jupyter y selecciona kernel "Python (fiuba)"
jupyter lab eda_taxis.ipynb
```

Cuando hay conectividad, el script descarga el archivo oficial:
`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet`.
Si la descarga falla (red restringida), genera una muestra sintética que
reproduce el esquema TLC y las inconsistencias documentadas.

## 4 · Renderizar la presentación

`PRESENTACION.md` está en formato **Marp**:

```bash
# Opción A — VS Code: instalar la extensión "Marp for VS Code"
# Opción B — CLI:
npx @marp-team/marp-cli PRESENTACION.md --pdf
```

## 5 · Salir / desactivar

```bash
deactivate
```

---

## Contenido del análisis (resumen rápido)

- **§1–6:** carga, dtypes, nulos, inconsistencias (`trip_distance ≤ 0`,
  `fare_amount = 0`, pickups fuera de mes), estadísticas descriptivas
  (duración, distancia, tarifa, total, propinas), y top zonas PU/DO.
- **§7:** histogramas (`trip_distance`, `fare_amount`) y volumen por hora
  del día.
- **§8:** correlaciones **Pearson vs Kendall τ** + matriz `|Pearson|−|Kendall|`
  para flagear no linealidades; tests de significancia con SciPy.
- **§9:** **entropía de Shannon** por variable categórica (H, H_max,
  H_normalizada).
- **§9.1:** **Mutual Information** entre continuas (discretizadas en deciles)
  y `payment_type`, `pickup_hour`, `PULocationID`.
- **§10:** conclusiones e implicaciones para el modelo final.

---

## Troubleshooting

| Problema | Solución |
|---|---|
| `command not found: python3` | Instala Python 3.10+ desde [python.org](https://www.python.org/downloads/macos/) o `brew install python@3.11` |
| El notebook no muestra "Python (fiuba)" | Ejecuta `python -m ipykernel install --user --name nyc_taxi_eda_fiuba --display-name "Python (fiuba)"` con el venv activado |
| Falla `pip install pyarrow` en macOS Apple Silicon | `pip install --upgrade pip wheel setuptools` y reintenta |
| `ssl.SSLError` al descargar el parquet | Actualiza certificados: `/Applications/Python\ 3.x/Install\ Certificates.command` |
