# HANDOFF — TP1 AMIA (LDA/QDA y optimización matemática)

**Alumno:** Gustavo Varela · **Materia:** Análisis Matemático para IA (AMIA), CEIA-FIUBA · TP optativo
**Estado general:** implementación y benchmarks completos y verificados. Quedan decisiones de entrega y 2 preguntas pendientes (ver §Pendientes).
**Última actualización:** 2026-06-12

---

## 1. Entorno y cómo retomar

- **Repo:** `~/projects/ceia-amia-tp` (clon de github.com/martinerrazquin/ceia-amia-tp).
- **Gestor:** `uv` → Python 3.12, NumPy 2.3.1, SciPy 1.16.0, scikit-learn 1.7.0.
- **Setup:** `uv sync`
- **Abrir el notebook:** `uv run --with jupyter jupyter lab tp_resuelto.ipynb`
- **Re-ejecutar todo:** `uv run --with jupyter --with nbconvert jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 tp_resuelto.ipynb`

## 2. Entregable

- **`tp_resuelto.ipynb`** — 32 celdas, ejecutado de punta a punta, 0 errores. Es la **fuente de verdad** (el `build_nb.py` que lo generó fue eliminado para no tener doble fuente). Editar el notebook directo.
- Estructura: §0 versiones · §1 Q3–Q7 · §2 Tensorización (P1–P7) · §3 Cholesky (P8–P13) · §4 Cross-Validation · §5 Conclusiones.

## 3. Código implementado (en los módulos `base/` y `utils/`)

| Clase / función | Archivo | Punto |
|---|---|---|
| `FasterQDA` | `base/qda.py` | P3 (elimina el `for`; materializa la matriz n×n) |
| `EfficientQDA` | `base/qda.py` | P6 (esquiva la n×n vía identidad de P5) |
| `TensorizedChol` | `base/cholesky.py` | P12 (hereda de `QDA_Chol3`) |
| `EfficientChol` | `base/cholesky.py` | P14 (sin `for` + sin n×n + Cholesky) |
| `cv_accuracy` | `utils/bench.py` | StratifiedKFold (pedido: usar CV para accuracy) |

## 4. Verificación y resultados

- **Correctitud:** las **9 variantes predicen idéntico a `QDA`** (chequeo byte a byte, `np.array_equal`). Son optimizaciones de cómputo, no cambian el modelo.
- **Accuracy CV (5-fold estratificado, Wine):** `0.9887 ± 0.0138` (igual para las 9).
- **Benchmark letters (n_test≈1000, k=26, p=16)** — lo que más muestra:

| modelo | test speedup | mem test | nota |
|---|---|---|---|
| QDA | 1× | 0.08 MB | baseline |
| TensorizedQDA | ~4.9× | 0.13 MB | paraleliza solo clases |
| **FasterQDA** | ~37× | **~205 MB** | ⚠️ matriz n×n (P4) |
| **EfficientQDA** | ~160× | ~9.8 MB | mismo tiempo, memoria mínima |
| QDA_Chol2 | ~0.78× | 0.07 MB | la **peor** en predict (`solve_triangular` por obs) |
| **EfficientChol** | ~210× | ~9.8 MB | 🏆 la mejor |

(En Wine las diferencias de memoria casi no se ven; por eso se agregó el subsample de letters.)

## 5. PENDIENTES / decisiones abiertas

1. **Q1 y Q2 → RESUELTO: no existen en el TP oficial.** El notebook de entrega real
   (`~/Downloads/AMIA_2025_TP1 (2).ipynb`) usa el `LabelEncoder` de **sklearn** y solo trae
   **Q3–Q7** + la Consigna QDA (P1–P13). Las Q1/Q2 que aparecieron antes eran de otra
   versión/cohorte (con encoder propio) y NO aplican. `tp_resuelto.ipynb` ya cubre el 100%.

2. **Formato de entrega → RESUELTO.** Se eligió el mono-notebook self-contained.
   **`AMIA_2025_TP1_resuelto.ipynb`** (en `~/projects/ceia-amia-tp/`) = material provisto intacto
   (teoría + código base + consigna) + sección "RESOLUCIÓN" con las 4 clases **inline**, Q3–Q7,
   P1–P13, benchmarks, CV y conclusiones. Ejecutado, 0 errores. **Es el entregable final.**
   (`tp_resuelto.ipynb`, versión repo que importa de `base/`, queda como respaldo.)

3. **Mail al profesor (avance/estado):** texto redactado (ver abajo) pero **no enviado**. Faltan: dirección del profesor y su nombre. ⚠️ El conector de Gmail disponible **solo crea borradores, no envía** → quedaría en Drafts para enviar a mano.

4. **Reportar versiones** Python/NumPy/SciPy en la entrega: ya resuelto (celda §0).

5. **Chiste de los determinantes:** preparado, no requiere acción. Resumen: en **LDA** el término `log|Σ|` se cancela en el `argmax` (Σ compartida) → ahí el determinante "no sirve"; en **QDA** es esencial (cada clase su Σ_j); y con **Cholesky** ni se llama a `det()` porque el determinante de una triangular es el producto de su diagonal.

## 6. Notas para quien retome

- **Determinantes en el código:** `LA.det` en `base/qda.py` (líneas 24/41/76/104); en `base/cholesky.py` se calcula como `L_inv.diagonal().prod()` (líneas 25/63) — propiedad det(triangular)=∏diagonal.
- **Preferencias de Gus (importante):** respuestas conceptuales = **"La idea" (intuición) primero, "El detalle" (código/fórmulas) después**; pregunta visualmente separada de la respuesta. **No transcribir sus borradores verbatim** — integrar el concepto redactado. Conclusiones conceptuales, no volcado de números.
- **Libro de referencia:** `referencias/mml-book.pdf` (gitignored, descargable de mml-book.github.io). Cholesky §4.3 · determinante §4.1 · gaussiana §6.5.
- **Git:** trabajo en working tree, **sin commitear**.

---

### Borrador del mail (pendiente de destinatario y envío)

> **Asunto:** TP1 AMIA — estado de avance
>
> Profesor [nombre]:
>
> Le escribo para comentarle cómo voy con el TP1 (LDA/QDA y optimización matemática).
>
> - Implementadas las cuatro clases pedidas: `FasterQDA`, `EfficientQDA`, `TensorizedChol`, `EfficientChol`.
> - Verifiqué que las nueve variantes predicen idéntico a `QDA` (byte a byte).
> - Respondidas las preguntas conceptuales y corridos los benchmarks en Wine y un subsample de *letters*; se ve el costo en memoria de la matriz n×n de `FasterQDA` frente a las variantes *Efficient*.
> - Sumé una evaluación de accuracy por cross-validation (5-fold estratificado).
>
> Entorno: Python 3.12, NumPy 2.3.1, SciPy 1.16.0 (gestionado con uv).
>
> Saludos,
> Gustavo Varela
