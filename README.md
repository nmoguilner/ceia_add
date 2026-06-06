# CEIA — Análisis de Datos (26Co2026)

**TP Grupal — Grupo 13**
Uso de taxis Yellow Cab en USA — *NYC TLC Yellow Taxi Trip Records*

El trabajo consiste en EDA, visualización, planteo de un problema de ML supervisado y preparación
del dataset (no requiere entrenar el modelo). El análisis principal está en
[`tp_grupal_eda.ipynb`](tp_grupal_eda.ipynb).

---

## 🛠️ Cómo levantar el proyecto localmente

Usamos [`uv`](https://docs.astral.sh/uv/) como gestor de entorno y paquetes. Con el `uv.lock` del
repo todos quedamos con **exactamente las mismas versiones** (cero "en mi máquina anda").

### 1) Instalar `uv` (una sola vez)

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(en Mac con Homebrew también: `brew install uv`)*

**Windows** (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cerrá y reabrí la terminal, y verificá:
```bash
uv --version
```

### 2) Clonar el repo
```bash
git clone https://github.com/nmoguilner/ceia_add.git
cd ceia_add
```

### 3) Crear el entorno e instalar todo
```bash
uv sync
```
Lee `pyproject.toml` + `uv.lock`, crea la carpeta `.venv` y baja todas las dependencias. **No hace
falta** crear venv ni usar `pip` a mano. El dataset ya viene versionado en `dataset/`, así que no
hay que descargar nada.

### 4) Abrir el notebook
```bash
uv run jupyter lab
```
Se abre Jupyter en el navegador → abrí **`tp_grupal_eda.ipynb`** → *Run All*. ✅

---

### 💡 Alternativa con VS Code
1. Hacé los pasos 1–3 (`uv sync`).
2. Abrí la carpeta en VS Code.
3. Abrí el `.ipynb` → **Select Kernel** (arriba a la derecha) → elegí el intérprete `.venv`
   (`Python 3.12 ('.venv')`).
4. *Run All*.

### ❓ Problemas comunes
- **`uv: command not found`** → cerrá y reabrí la terminal; el instalador agrega `uv` al PATH.
- **Falta una librería** → corré `uv sync` de nuevo. Para agregar paquetes nuevos usá
  `uv add <paquete>` (no `pip install`), así queda en el `uv.lock` para todo el grupo.
- **El kernel no aparece en VS Code** → seleccioná manualmente el de `.venv`.

---

## 📂 Estructura

```
ceia_add/
├── tp_grupal_eda.ipynb     # análisis principal (EDA + viz + problema ML + preprocesamiento)
├── dataset/                # datos versionados (parquet de viajes + lookup de zonas)
├── data_dictionary_trip_records_yellow.pdf
├── pyproject.toml          # dependencias
└── uv.lock                 # versiones exactas (reproducibilidad)
```

## 📚 Dataset

NYC Taxi & Limousine Commission (TLC) — *Yellow Taxi Trip Records*.
Fuente oficial: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
