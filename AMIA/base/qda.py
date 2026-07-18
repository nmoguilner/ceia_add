import numpy as np
import numpy.linalg as LA

from base.bayesian import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

  def _fit_params(self, X, y):
    # estimate each covariance matrix
    self.inv_covs = [LA.inv(np.cov(X[:,y.flatten()==idx], bias=True))
                      for idx in range(len(self.log_a_priori))]
    # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
    # Q6: por que se usa bias=True en vez del default bias=False?
    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]
    # Q7: que hace axis=1? por que no axis=0?

  def _predict_log_conditional(self, x, class_idx):
    # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
    # this should depend on the model used
    inv_cov = self.inv_covs[class_idx]
    unbiased_x =  x - self.means[class_idx]
    return 0.5*np.log(LA.det(inv_cov)) -0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X,y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self,x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class FasterQDA(TensorizedQDA):
    """P3) Elimina el ciclo `for` de `predict`: paraleliza sobre clases Y
    observaciones simultáneamente.

    A diferencia de `TensorizedQDA` (que paraleliza sobre las k clases pero
    sigue iterando observación por observación), acá pasamos toda la matriz
    X (p, n) de una sola vez.

    P4) El precio es que aparece explícitamente una matriz de n x n: al hacer
    `unbiased.transpose(0,2,1) @ inv_cov @ unbiased` con unbiased de shape
    (k, p, n), el resultado es (k, n, n). Solo nos interesa su diagonal
    (las n formas cuadráticas (x_i - mu)^T Sigma^-1 (x_i - mu)); el resto de
    la matriz son "interacciones cruzadas" entre observaciones distintas que
    se descartan. Es O(n^2) en cómputo y memoria, innecesariamente.
    """

    def predict(self, X):
        # X: (p, n)
        # unbiased[k] = X - mu_k  ->  (k, p, n) por broadcasting
        unbiased = X[np.newaxis, :, :] - self.tensor_means

        # (k, n, p) @ (k, p, p) @ (k, p, n) -> (k, n, n)  <-- la matriz n x n
        prod = unbiased.transpose(0, 2, 1) @ self.tensor_inv_cov @ unbiased

        # nos quedamos solo con la diagonal de cada (n, n): (k, n)
        quad = np.diagonal(prod, axis1=1, axis2=2)

        # 0.5 * log|inv_cov| por clase: (k, 1)
        log_dets = 0.5 * np.log(LA.det(self.tensor_inv_cov))[:, np.newaxis]
        log_conditionals = log_dets - 0.5 * quad                  # (k, n)
        log_posteriori = self.log_a_priori[:, np.newaxis] + log_conditionals

        # clase con MAP por observación: (n,) -> (1, n)
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)


class EfficientQDA(TensorizedQDA):
    """P6) Reimplementa `FasterQDA` esquivando la matriz de n x n.

    Usa la identidad demostrada en P5:
        diag(A @ B) = np.sum(A * B.T, axis=1)
    con A = unbiased^T (n, p) y B = inv_cov @ unbiased (p, n). Eso es
    equivalente a, directamente, sumar sobre las p features el producto
    elemento a elemento de `unbiased` con `inv_cov @ unbiased`:
        (x-mu)^T Sigma^-1 (x-mu) = sum_p  unbiased * (Sigma^-1 unbiased)
    Nunca se materializa la (n, n): se trabaja con matrices (k, p, n).
    """

    def predict(self, X):
        # X: (p, n)
        unbiased = X[np.newaxis, :, :] - self.tensor_means        # (k, p, n)
        M = self.tensor_inv_cov @ unbiased                        # (k, p, n)

        # diag de la forma cuadrática vía Hadamard + suma sobre p: (k, n)
        quad = np.sum(unbiased * M, axis=1)

        log_dets = 0.5 * np.log(LA.det(self.tensor_inv_cov))[:, np.newaxis]
        log_conditionals = log_dets - 0.5 * quad
        log_posteriori = self.log_a_priori[:, np.newaxis] + log_conditionals

        return np.argmax(log_posteriori, axis=0).reshape(1, -1)
