# ML FIUBA — Penalización por edad y sexo en el mercado laboral argentino

Trabajo final de la asignatura **Aprendizaje de Máquina** — Carrera de
Especialización en Inteligencia Artificial (CEIA), Facultad de Ingeniería,
Universidad de Buenos Aires.

## Objetivo e hipótesis
Medir la penalización por **edad** y **sexo** en el mercado laboral argentino,
con foco en la franja cercana a la edad jubilatoria (mujeres 60 / varones 65),
cuidando **no mezclar** ambas dimensiones ni atribuir al mercado lo que explica
la norma previsional. Hipótesis en dos partes: (**H1**) la penalización etaria
golpea a ambos sexos y, comparada a igual distancia de la jubilación propia y
entre quienes participan del mercado, es **igual o mayor en los varones**;
(**H2**) la brecha de empleo por sexo es ante todo una brecha de
**participación** (oferta: cuidados, hogar, transferencias, regímenes
especiales), más un castigo de **ingresos** entre ocupados que la composición
observable no explica.

## Dataset
Microdatos de la **EPH (INDEC)**, base individual, **2016–2024** (35 trimestres,
~1.8M registros). Se descargan con `src/data/descargar_eph.py` (paquete `pyeph`)
a `data/raw/`. Los CSV crudos **no se versionan** (ver `.gitignore`).

Variables clave (códigos EPH): `CH04` sexo · `CH06` edad · `ESTADO` condición de
actividad · `NIVEL_ED` nivel educativo · `P21` ingreso de la ocupación principal ·
`PONDERA`/`PONDIIO` ponderadores · `REGION` geografía.

Decisiones metodológicas (detalladas en la sección 3 del notebook): ponderadores siempre,
ingreso **relativo** a la mediana del trimestre (elimina la inflación 2016–2024),
y sin fuga de información en la clasificación de empleo.

## Entregable
**`notebooks/entrega.ipynb`** — notebook final, autocontenido y ejecutado de punta
a punta (intro, datos, EDA de brechas, Modelo A de ingreso, Modelo B de empleo,
Oaxaca–Blinder, fairness y la ampliación: árbol interpretable, SVM-RBF, evolución
temporal, SHAP y clustering). Se ejecuta de punta a punta con:
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/entrega.ipynb
```

## Modo rápido vs. completo (`ML_FIUBA_MODO`)
Un único interruptor escala el costo de cómputo **sin cambiar la estructura ni las
conclusiones**, sólo el tiempo. Lo leen tanto el notebook como los scripts de
preparación de `src/` (variable de entorno `ML_FIUBA_MODO`):

- **`rapido`** (por defecto): submuestrea (15k) para reejecutar en ~1–2 min e
  iterar mientras se estudia/defiende.
- **`completo`**: usa el dataset entero (~15–20 min). Es la corrida final cuyas
  cifras se citan en el texto.

```bash
# rápido (default)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/entrega.ipynb
# completo (números definitivos de la entrega)
ML_FIUBA_MODO=completo uv run jupyter nbconvert --to notebook --execute --inplace notebooks/entrega.ipynb
```

Parámetros que escala (rápido → completo): muestra de modelos supervisados
(15k → completo), subconjunto SVM-RBF (3k → 12k), muestra SHAP (1k → 5k),
`permutation_importance` n_repeats (3 → 10), grilla de GridSearch (mínima →
completa) y KMeans `n_init` (3 → 10).

## Estructura
```
ML FIUBA/
├── configs/            # Configuración del pipeline genérico (config.yaml)
├── data/
│   ├── raw/            # Microdatos EPH originales (no versionados)
│   ├── processed/      # Base analítica (eph_analitico.parquet) y datasets de modelado
│   └── external/       # Fuentes externas
├── notebooks/          # entrega.ipynb (notebook final de entrega)
├── src/
│   ├── data/           # Descarga (descargar_eph) y EDA de brechas (eda_eph, empleo_curvas)
│   ├── models/         # Preparación de datasets, pipeline genérico (train/benchmark/tune), modelo_empleo
│   └── visualization/  # Gráficos
├── models/             # Modelos entrenados (.joblib)
├── reports/figures/    # Figuras generadas
├── pyproject.toml      # Dependencias (uv)
└── uv.lock
```

## Setup (uv)
Las dependencias están declaradas en `pyproject.toml` y fijadas en `uv.lock`.
```bash
uv sync                                # crea .venv e instala todo desde el lock
uv sync --group data --group fairness  # + pyeph (descarga EPH) y fairlearn
```

## Uso
```bash
# 1. Descargar microdatos de la EPH (corré esto en tu máquina con internet)
uv run --group data python -m src.data.descargar_eph --desde 2016 --hasta 2024

# 2. EDA de brechas por edad y sexo + base analítica (eph_analitico.parquet)
uv run python -m src.data.eda_eph

# 3. Curvas de empleo e inactividad (la salida del mercado antes de la jubilación)
uv run python -m src.data.empleo_curvas

# 4a. Modelo de INGRESO (regresión): preparar dataset
uv run python -m src.models.preparar_modelo_eph

# 4b. Modelo de EXCLUSIÓN DEL EMPLEO (clasificación 35-64) + penalización por edad
uv run python -m src.models.preparar_clasificacion_eph
uv run python -m src.models.modelo_empleo

# 5. Pipeline genérico (config-driven): entrenar, comparar y tunear
uv run python -m src.models.train     --config configs/config.yaml
uv run python -m src.models.benchmark --config configs/config.yaml
uv run python -m src.models.tune      --config configs/config.yaml

# 6. Notebook final de entrega (autocontenido, lee data/processed/eph_analitico.parquet)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/entrega.ipynb
```

El pipeline genérico de `src/models` soporta **clasificación y regresión** (campo
`task` en el config); el preprocesamiento (imputación + escalado + one-hot) se
aplica automáticamente según el tipo de cada columna. El notebook de entrega es
independiente de ese pipeline: reimplementa la lógica en celdas para ser
autocontenido.
