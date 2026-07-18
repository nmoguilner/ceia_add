# Experimentos (fuera de la entrega)

Pruebas exploratorias que **no** forman parte del notebook de entrega.

## `svm_rbf_metal.py` — SVM-RBF acelerado por GPU (Metal)

Responde al desafío "¿se puede correr el SVM-RBF en la GPU del Mac?".

**Enfoque honesto (híbrido):** scikit-learn es CPU-only y no hay un SVM-kernel-en-Metal
llave en mano. Lo que de verdad conviene a la GPU es la **matriz de kernel RBF** (el
cuello O(n²), álgebra densa); el solver SMO es secuencial y se deja en CPU. Entonces:

1. La **matriz de Gram RBF** se calcula en la GPU con **MLX** (Apple/Metal), usando
   `||x-y||² = ||x||² + ||y||² - 2·x·yᵀ` (el `x·yᵀ` es un matmul que la GPU acelera).
2. Se resuelve con `sklearn.svm.SVC(kernel="precomputed")` (SMO en CPU sobre el Gram).

**Resultados** (Apple M5 Max, GPU 40 núcleos, Metal 4):

- Correctitud: el kernel GPU es **idéntico a sklearn a precisión float32**
  (máx |Δ| ≈ 9e-4) y el AUC del SVM coincide (Δ ≈ 6e-5).
- Matriz de Gram: GPU **~2-3x** más rápida que `sklearn.rbf_kernel` (que ya usa BLAS
  multinúcleo) para n = 2k–20k.
- End-to-end (kernel + fit): **~1.5x**. El tope lo pone el solver SMO (CPU), que pasa
  a dominar el tiempo una vez que la matriz se calcula rápido.

**Conclusión:** sí se puede, con un speedup real pero **modesto** a esta escala. La
GPU paga más a mayor n / más dimensiones. Para el trabajo principal no se adopta
(no aporta accuracy y agrega una dependencia + complejidad), pero queda como prueba
de que la parte n² del SVM-RBF es perfectamente "metalizable".

```bash
uv run --group gpu python experiments/svm_rbf_metal.py
```
