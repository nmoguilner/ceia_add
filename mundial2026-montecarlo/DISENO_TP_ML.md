# Documento de Diseño — Trabajo Final de Aprendizaje de Máquina

**Proyecto:** ¿Puede el aprendizaje supervisado mejorar la predicción de partidos del Mundial 2026 sobre un baseline ELO?
**Autor:** Gus
**Materia:** Aprendizaje de Máquina (CEIA)
**Fecha límite de entrega:** domingo 23 de agosto de 2026
**Estado:** v2 — diseño revisado (contra el código existente) y endurecido para implementación

---

## 1. Resumen

El proyecto parte de un simulador Monte Carlo del Mundial FIFA 2026 ya construido, cuyo "motor de partido" deriva la probabilidad de cada resultado de una fórmula fija basada en ELO. La contribución de Aprendizaje de Máquina es **reemplazar ese motor por un clasificador supervisado** que aprende la probabilidad de resultado (local / empate / visitante) a partir de múltiples variables del partido, y **comparar rigurosamente** ese enfoque contra el baseline ELO. El mejor clasificador se integra de nuevo al Monte Carlo para estimar la probabilidad de campeón de cada selección, y el valor se mide tanto con métricas técnicas como con una métrica de negocio: la cercanía de esas probabilidades a las del mercado de apuestas / medios.

---

## 2. Problema y motivación

El simulador existente estima en cuántos escenarios sale campeona cada selección, simulando miles de veces los partidos que faltan. Hoy cada partido se modela con dos Poisson cuyas medias salen de la diferencia de ELO. Ese modelo es un **baseline fuerte y bien calibrado** (61 % de acierto a 3 vías, Brier 0.502 en backtest), pero usa una sola variable (Δ ELO) y una forma funcional fija.

**Pregunta:** ¿se puede mejorar la predicción probabilística de un partido aprendiendo de variables adicionales (forma reciente, descanso, importancia del torneo, localía, confederación), con modelos supervisados que capturen no linealidades e interacciones que la fórmula ELO no representa?

### 2.1 Por qué ML es apropiado acá (y no es el caso "balística")

Un criterio de la materia: si existe una fórmula cerrada y exacta que predice el resultado (p. ej. la trayectoria de un proyectil), no se hace ML; se aplica la fórmula. **El ELO no es ese caso**, por tres razones:

1. No predice el resultado, sino una **probabilidad**; el partido es genuinamente estocástico.
2. Tiene **error residual grande**: un Brier de 0.502 (≠ 0) es la prueba empírica de que sobra incertidumbre estructurada que un modelo puede intentar capturar. La balística tendría Brier ≈ 0.
3. El ELO mismo es un **modelo ajustado a datos** (sus parámetros se calibraron por MLE en este proyecto), no una ley física.

Por lo tanto, estimar la probabilidad de un resultado estocástico es un problema de predicción legítimo. El ELO no anula el proyecto: **es el baseline de dominio** que la consigna pide definir antes de los modelos complejos (el equivalente futbolístico al "precio promedio por barrio").

---

## 3. Pregunta de investigación e hipótesis

- **H1 (técnica):** un clasificador con features adicionales mejora el log-loss y el Brier respecto del baseline ELO en el conjunto de test temporal.
- **H2 (negocio):** las probabilidades de campeón generadas con el motor ML quedan más cerca del consenso de mercado/medios que las del motor ELO.
- **H0 (resultado válido y honesto):** si el ELO ya captura casi toda la señal aprendible, las mejoras serán marginales. Esa también es una conclusión aprobatoria; lo evaluado es el rigor del proceso, no la magnitud de la mejora.

---

## 4. Datos

| Fuente | Archivo | Filas | Contenido |
|---|---|---:|---|
| martj42/international_results | `data/history.parquet` | 49.477 | partidos internacionales 1872–2026: fecha, equipos, goles, torneo, `neutral` |
| ELO reconstruido (algoritmo eloratings) | `elo_history.py` | — | ELO **pre-partido** de cada equipo en cada fecha (feature estrella) |
| worldfootballrankings | `data/elo.parquet` | 48 | ELO actual de las 48 selecciones |
| StatsBomb (opcional) | `data/sb_matches.parquet` | 314 | partidos con detalle, posible enriquecimiento futuro |

**Requisitos de la materia, verificados:** (a) suficientes datos de calidad — 49 k partidos; (b) existe un patrón — 61 % > 33 % de azar; (c) el patrón es aprendible — datos tabulares, tamaño modesto, clasificación estándar. Los tres se cumplen.

> **Nota de alcance temporal (regla única de split, ver §7):** el histórico completo (1872–2026) se usa **solo** para reconstruir el ELO pre-partido y las features de forma. El **modelado** sigue la lógica del **ciclo mundialista**, alineada con el objetivo del trabajo (*"¿de qué formas puedo simular el Mundial 2026?"*): se entrena con los partidos **desde el Mundial de Qatar 2022 (2022-11-20) hasta la víspera del Mundial 2026 (2026-06-10)** — **3.705 partidos**, que incluyen el Mundial 2022 completo (64) — y se reserva como **test** el **Mundial 2026 real** (~104 partidos, ya concluido para la entrega de agosto-2026). El Mundial 2026 **no entra al entrenamiento bajo ninguna forma**: es el objeto a predecir. Arrancar en Qatar 2022 garantiza que el modelo vea al menos un Mundial completo (intensidad máxima de `tournament_weight`), evitando extrapolar sobre el único tipo de partido que compone el test.

---

## 5. Variable objetivo y unidad de análisis

- **Unidad:** un partido internacional.
- **Objetivo:** resultado en tiempo reglamentario, **3 clases** — `H` (gana local), `D` (empate), `A` (gana visitante).
- **Distribución esperada:** desbalanceada; el empate es la clase minoritaria (~25 %). Se trata explícitamente (ver §7 y §10).

---

## 6. Ingeniería de features

Esto es lo que diferencia el proyecto de "aplicar el ELO". Para cada partido, calculadas **solo con información previa al partido** (sin fuga):

| Feature | Descripción | Por qué |
|---|---|---|
| `delta_elo` | ELO pre-partido local − visitante | señal central; se incluye también como feature, no solo como baseline |
| `elo_home`, `elo_away` | niveles absolutos | nivel general del cruce |
| `is_home` / `neutral` | localía real vs cancha neutral | ventaja de campo |
| `form_gd_5` | diferencia de gol promedio últimos 5 (cada equipo) | estado de forma |
| `form_pts_5` | puntos promedio últimos 5 (cada equipo) | racha de resultados |
| `rest_days` | días desde el último partido (cada equipo) | fatiga / calendario |
| `tournament_weight` | amistoso < eliminatoria < copa | importancia/intensidad |
| `confed_home`, `confed_away` | confederación | estilo/contexto regional |
| `h2h_recent` | balance histórico reciente entre ambos | factor de enfrentamiento |

El conjunto final se decide con análisis de correlación/redundancia y, eventualmente, importancia de variables del modelo de árboles.

> **Ablación incremental (resultado central, no opcional).** La pregunta de defensa más probable es *"¿los modelos no están solo re-aprendiendo el ELO?"*. La respuesta empírica es una ablación escalonada en el test: **(a) solo `delta_elo` → (b) ELO + forma/descanso → (c) todas las features**, mostrando cómo baja el log-loss en cada paso. Acompañada de la importancia de variables, es la prueba de que las features no-ELO aportan señal propia. Si `delta_elo` domina y el resto aporta poco, esa es una conclusión válida (H0, §3) que se reporta de frente; pero la ablación debe estar sí o sí para sostener la defensa.

---

## 7. Preprocesamiento

- **Limpieza:** descartar partidos sin ELO pre-partido suficiente; eliminar duplicados; validar nulos.
- **Estandarización:** escalado de features numéricas (obligatorio para KNN y SVM; inocuo para árboles).
- **Desbalance:** el empate es minoritario, pero **no se usa `class_weight='balanced'` en el modelo que alimenta el Monte Carlo**: reponderar las clases descalibra las probabilidades a propósito (sube recall del empate a costa de la calibración), y aguas abajo necesitamos probabilidades calibradas, no etiquetas duras. Se deja que el modelo aprenda las tasas base reales; si la calibración lo pide, se ajusta con `CalibratedClassifierCV`. F1 macro se reporta como métrica del empate, **no** como criterio de selección. (Se puede mostrar una variante con `class_weight` como contraste didáctico, sin que sea el modelo productivo.)
- **Split temporal (regla única, crítica):** **train = ciclo Qatar 2022 → víspera 2026** (2022-11-20 .. 2026-06-10, 3.705 partidos); **test = Mundial 2026 real** (~104 partidos). **Nunca** split aleatorio: mezclar en el tiempo filtra el futuro y es el error clásico en datos deportivos. El corte es **estricto en 2026-06-10**: ningún partido del Mundial 2026 toca el entrenamiento. Es la **misma** partición que usa el baseline ELO en §8, para que la comparación sea sobre filas idénticas. (Las features de forma/descanso/h2h de un partido del train se calculan mirando partidos *anteriores* a él, que pueden ser previos a 2022-11-20 — eso no es fuga: es información disponible antes de ese partido.)
- **Consistencia train/test:** el escalador y cualquier transformación se ajustan **solo con train** y se aplican a test (pipeline de scikit-learn).

---

## 8. Baselines

1. **Trivial:** frecuencia de clases / "siempre gana el local". Piso de referencia.
2. **De dominio (ELO-Poisson):** el motor actual, que da probabilidades W/D/L analíticas vía `lambdas()` + Dixon-Coles. Es el baseline serio a superar.
   - **Configuración única del baseline:** el modelo **calibrado por MLE con corrección Dixon-Coles** (`base`, `escala`, `home_adv` de `calibration.json`, con `rho`), que es el mejor ELO disponible en el proyecto. No se compara contra la variante "a mano" para no elegir un rival débil a conveniencia.
   - **Comparación honesta (innegociable, y ahora natural):** como el test **ES** el Mundial 2026, el baseline ELO se evalúa sobre esos mismos partidos — exactamente lo que ya hace `backtest.py` sobre `played.parquet` (61 % acierto, Brier 0.502 sobre el snapshot parcial; se **recomputa sobre el torneo completo** en la entrega). `evaluate.py` corre ELO y ML sobre las **mismas filas del Mundial 2026** y reporta ambos lado a lado. Clave: el clasificador debe predecir esos mismos partidos sin que ninguno haya tocado el entrenamiento.

---

## 9. Modelos a comparar (≥ 3)

Se eligen tres familias con sesgos inductivos distintos, alineadas con el programa de la materia:

| Modelo | Familia | Clase del programa | Nota |
|---|---|---|---|
| Regresión Logística multinomial | lineal | IA / repaso | interpretabilidad de coeficientes |
| KNN | basado en instancias | Clase 2 | exige escalado; sensible a `k` |
| SVM (kernel RBF) | margen / no lineal | Clase 3 | probabilidades vía Platt **mal calibradas** → envolver en `CalibratedClassifierCV` |
| Random Forest | ensamble (bagging) | Clase 5 | da importancia de variables (alimenta la ablación de §6) |
| HistGradientBoosting | ensamble (boosting) | Clase 5 | candidato fuerte en tabular + calibra razonable; **no opcional** |

Son cinco familias con sesgos inductivos distintos (cubre IIA→Clase 5). Se justifica cada elección y se reporta cuál gana y por qué. Los ensambles suelen rendir mejor en datos tabulares; la logística aporta interpretabilidad. **Como el modelo final alimenta el Monte Carlo, la métrica de selección es probabilística (log-loss) y la calibración importa más que el acierto duro**: por eso SVM se calibra explícitamente y HGB pasa a candidato de primera, no a "opcional".

---

## 10. Protocolo de evaluación

- **Validación cruzada temporal:** `TimeSeriesSplit` (ventana expansiva) dentro de train; nunca CV aleatoria.
- **Optimización de hiperparámetros:** `GridSearchCV` / `RandomizedSearchCV` sobre la CV temporal. A tunear: `C`/`gamma` (SVM), `k` (KNN), profundidad y `n_estimators` (RF), `learning_rate` y `max_iter` (HGB).
- **Selección del modelo final:** por **log-loss multiclase**, porque el uso aguas abajo (alimentar el Monte Carlo) requiere probabilidades bien calibradas, no etiquetas duras.
- **Evaluación final:** una sola vez, sobre el test reservado = **Mundial 2026 real**. El tuning se hace con `TimeSeriesSplit` dentro del train (ciclo 2022–2026); el Mundial no se toca hasta esta evaluación final.

---

## 11. Métricas

**Técnicas (modelo):**

- Log-loss multiclase — métrica principal de selección.
- Brier (3 vías) y acierto — comparables con el backtest existente.
- F1 macro — robusta al desbalance del empate.
- Curva de calibración / reliability diagram (Clase 7, opcional pero ya disponible en el proyecto).

**De negocio (cercanía al mercado/medios):**

- Se integra el clasificador al Monte Carlo y se obtiene el vector `P(campeón)` por selección.
- Se compara contra un **vector de referencia** del mercado (cuotas *outright winner* normalizadas a probabilidad) o de medios (p. ej. simulaciones publicadas por La Nación).
- Métrica: error absoluto medio y correlación de Spearman entre `P(campeón)_modelo` y `P(campeón)_referencia`, para los motores ML y ELO.
- **Encuadre honesto (sanity-check, no juez):** "parecerse al mercado" es un *chequeo de plausibilidad*, no una verdad de terreno — el mercado también se equivoca. Por eso H2 es **secundaria** a log-loss (§10): si el modelo se aleja del mercado no está necesariamente mal. Se reporta como evidencia de que las probabilidades de torneo son razonables, no como criterio de selección.
- **Snapshot comparable (crítico):** el estado actual del torneo tiene la mayoría de los grupos cerrados, así que muchas de las 48 selecciones ya tienen `P(campeón)=0` y Spearman/MAE sobre un vector casi nulo mide poco. La comparación de negocio se hace sobre un estado donde las 48 tienen masa: idealmente un **snapshot pre-torneo**, contrastado con una referencia de mercado **de la misma fecha**. Mezclar fechas (modelo mid-torneo vs cuotas pre-torneo) invalida la comparación.
- **Traducción técnico→negocio:** una mejora de log-loss por partido se traduce en probabilidades de torneo más cercanas al consenso del mercado — el "valor tangible" que pide la materia.

> **Tarea de datos pendiente:** snapshotear una vez el vector de referencia (cuotas de una casa de apuestas o números publicados) **con su fecha**, para poder alinearlo con el estado del simulador. Sin esto, H2 no se puede medir.

---

## 12. Integración al Monte Carlo

El clasificador entrega `P(H), P(D), P(A)` por cruce. Integración:

> **Contrato real de la interfaz (verificado contra `wcsim.py`).** `MLMatchModel` debe replicar lo que el simulador *realmente* llama, que no es "probabilidades W/D/L":
> - `play_group(a, b, rng)` devuelve un **marcador `(gh, ga)`**, no un resultado — el simulador usa GF/GA para los desempates de grupo y los 8 mejores terceros (`_rank_key`, `simulate_group_stage`). Por eso el muestreador de marcador **es obligatorio, no un detalle**.
> - `play_knockout(a, b, rng, venue)` devuelve el **ganador**.
> - El simulador lee el atributo **`model.elo`** para los desempates y los mejores terceros, así que `MLMatchModel` debe exponer un `.elo` (dict) aunque la predicción sea ML.

- **Knockout:** se muestrea el ganador de las probabilidades `P(H/D/A)` del clasificador (la `D` se resuelve con la cascada de §12.1).
- **Fase de grupos:** se necesita el **marcador** (para desempate por diferencia de gol). Se muestrea el resultado con el clasificador y el marcador con una distribución empírica condicionada (resultado + franja de Δ ELO). **Sin marcador no hay fase de grupos**: es parte central de la integración, no opcional.
- **Performance (precómputo, crítico):** llamar `predict_proba` por partido dentro del Monte Carlo son **millones de llamadas** (N sims × ~decenas de partidos) → inviable. La solución es **precomputar una tabla de `P(H/D/A)` (y la distribución de marcador) por cruce una sola vez** y muestrear de la tabla en el loop, igual que el `_qgrid` actual del ELO. Esto exige features **estáticas pre-torneo**: por eso `MLMatchModel` se diseña **estático** (no recalcula forma/ELO dentro del torneo). Si en el futuro se quisiera ELO dinámico, no se puede precomputar y habría que repensar la performance.
- **Comparación A/B (offline):** se corre el Monte Carlo con (a) motor ELO y (b) motor ML sobre el mismo snapshot; es el espejo del ejemplo "reglas vs ML" de la materia.

### 12.1 Cascada de eliminación (90′ → alargue → penales)

Decisión de diseño basada en disponibilidad de datos por etapa:

- **90 minutos → ML.** ~50 k partidos para aprender; es la tarea del clasificador.
- **Alargue (2×15) → regla simple.** Pocos cientos de casos en la historia: insuficiente para aprender. Se modela como mini-partido con intensidad escalada (× 30/90).
- **Penales → casi moneda.** Débilmente predecibles y con pocos datos; se mantiene un sesgo leve (~55/45) en lugar de un modelo. Saber *cuándo no usar ML* es parte de la madurez que evalúa la materia.

Agregar la etapa de alargue (hoy ausente: el motor salta de empate a penales) suma realismo y reduce cuántas veces se llega a penales.

---

## 13. Visualizaciones planeadas

- Comparación de métricas entre modelos (barras: log-loss, Brier, F1).
- Curva de calibración por modelo.
- Importancia de variables (Random Forest).
- `P(campeón)` ML vs ELO vs mercado (barras agrupadas, top-15).
- Convergencia del estimador Monte Carlo (ya existe en el proyecto).

---

## 14. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Los modelos solo "re-aprenden" el ELO → valor agregado flaco | ingeniar features con señal propia (forma, descanso, torneo); medir su aporte |
| Desbalance del empate | `class_weight`, F1 macro, calibración |
| Pocos datos en alargue/penales | no usar ML ahí; reglas principiadas (§12.1) |
| Fuga temporal | split y CV estrictamente temporales; transformaciones ajustadas solo en train |
| Entrenamiento lento | train acotado al ciclo 2022–2026 (3.705 partidos); modelos clásicos, sin redes |
| Pocos Mundiales en train (solo Qatar 2022, 64 partidos) | el modelo igual ve un torneo completo (no extrapola); reportar el efecto `tournament_weight` con cautela y apoyar la conclusión en la ablación, no en ese único coeficiente |

---

## 15. Cumplimiento de la consigna (checklist)

- [x] Introducción al problema con motivación.
- [x] Descripción y preparación de datos (mini-preprocesamiento; base ya existente).
- [x] Baseline definido (trivial + ELO de dominio).
- [x] ≥ 3 modelos comparados con justificación.
- [x] Métrica de comparación definida (log-loss principal).
- [x] Validación cruzada (temporal) + optimización de hiperparámetros.
- [x] Evaluación en test reservado.
- [x] Visualizaciones comparativas de métricas.
- [x] Storytelling de punta a punta + métrica de negocio.
- [x] Sin redes neuronales (corresponde a Aprendizaje Profundo).
- [x] Calibración (Clase 7) incluida como opcional/bonus.

---

## 16. Plan de trabajo hacia el 23 de agosto

| Hito | Entregable | Momento sugerido |
|---|---|---|
| 0. Serie ELO pre-partido | refactor de `elo_history.reconstruir()` → `data/elo_prematch.parquet` | antes de featurizar |
| 1. Featurización | `featurize.py` → tabla de entrenamiento con ELO pre-partido + features §6 | tras Clase 2 (KNN) |
| 2. Baseline + EDA | métricas del ELO replicadas + perfil del dataset | misma semana |
| 3. Modelos + tuning | 3 modelos, CV temporal, GridSearch | tras Clase 3–5 |
| 4. Evaluación | test final + visualizaciones comparativas | post Clase 5 |
| 5. Integración MC | motor ML en el simulador + `P(campeón)` ML vs ELO | post Clase 5 |
| 6. Métrica de negocio | snapshot de mercado + comparación | en paralelo |
| 7. Informe/notebook | storytelling completo, conclusiones | última semana |

Avance incremental: tras la Clase 5 (ensambles) el trabajo ya puede cerrarse; calibración (Clase 7) es bonus.

---

## 17. Estructura del entregable (notebook)

1. Introducción al problema y motivación.
2. Datos: descripción, EDA express, preparación.
3. Baseline (trivial + ELO).
4. **Entrenamiento de modelos** (núcleo): qué algoritmos, por qué, comparación.
5. Métricas y evaluación (técnicas + negocio) con visualizaciones.
6. Integración al Monte Carlo y resultado de torneo.
7. Conclusiones, limitaciones y trabajo futuro.

---

## 18. Anexo — Especificación para Claude Code (ejecución)

Módulos a construir (el motor `wcsim.py` ya expone `MatchModel` como punto de inyección):

- **Prerrequisito — serie de ELO pre-partido.** Hoy `elo_history.reconstruir()` calcula las `muestras` (ELO pre-partido de *cada* partido histórico) pero **las descarta**: solo persiste el ELO final de las 48 selecciones. `featurize.py` necesita esa serie completa, así que el primer paso es **refactorizar `reconstruir()` para volcar las `muestras` a un parquet** (p. ej. `data/elo_prematch.parquet` con `date, home, away, elo_home_pre, elo_away_pre, neutral, gh, ga`), o importarla y llamarla directamente desde `featurize.py`.
- **`featurize.py`** — entrada: `data/history.parquet` + serie de ELO pre-partido (del prerrequisito). Salida: `data/ml_dataset.parquet` con una fila por partido, las features de §6 y la etiqueta `{H,D,A}`. Marca el split temporal como columna reproducible: `train` = 2022-11-20 .. 2026-06-10, `test` = Mundial 2026 (≥ 2026-06-11). Las features de cada partido miran solo partidos anteriores a él (sin fuga), aunque para eso lea el histórico previo a la ventana de train.
- **`train.py`** — pipeline scikit-learn (escalado + modelo), CV temporal (`TimeSeriesSplit`), `GridSearchCV`, selección por log-loss. **Sin `class_weight`** en el modelo productivo (§7); SVM envuelto en `CalibratedClassifierCV`. Persiste el mejor modelo y un reporte de métricas, incluida la **ablación de §6**.
- **`MLMatchModel`** (en módulo nuevo) — implementa el **contrato real** de §12: `play_group(a,b,rng) → (gh,ga)` (con muestreador de marcador), `play_knockout(a,b,rng,venue) → ganador`, y atributo `.elo` para los desempates del simulador; incluye la cascada de §12.1. **Estático con tabla precomputada** de `P(H/D/A)` y de marcador por cruce (§12, performance).
- **`evaluate.py` / notebook** — corre ELO (baseline calibrado, §8) y ML **sobre las mismas filas del test = Mundial 2026**; métricas técnicas, calibración, ablación, y comparación de `P(campeón)` ML vs ELO vs referencia de mercado (con snapshot de fecha alineada, §11).

**Reglas operativas:** las features se calculan sin fuga temporal; escalador ajustado solo con train; baseline ELO re-evaluado sobre el test (no se reusa el 0.502 del backtest del torneo); nada de redes neuronales; el sandbox no escribe sobre `.git` — commits/push se delegan al entorno del usuario.
