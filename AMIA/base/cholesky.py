import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri

from base.bayesian import BaseBayesianClassifier


class QDA_Chol1(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.Ls = [
        cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True)
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L = self.Ls[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = solve_triangular(L, unbiased_x, lower=True)

    return -np.log(L.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        dtrtri(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True), lower=1)[0]
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()


class TensorizedChol(QDA_Chol3):
    """P12) Versión tensorizada de la variante Cholesky.

    Hereda de `QDA_Chol3` (que invierte la triangular L con `dtrtri`, la más
    barata en el fit). Apila las inversas triangulares L^-1 en un tensor
    (k, p, p) y las medias en (k, p, 1), igual que `TensorizedQDA`, para
    calcular las k log-condicionales de una observación en una sola pasada.

    Idea (P8): si Sigma = L L^T, entonces
        (x-mu)^T Sigma^-1 (x-mu) = || L^-1 (x-mu) ||^2 = (y**2).sum()
    con y = L^-1 (x-mu), y  0.5*log|Sigma^-1| = log( prod diag(L^-1) ).
    Paraleliza sobre clases (no sobre observaciones: sigue el for de predict).
    """

    def _fit_params(self, X, y):
        super()._fit_params(X, y)
        self.tensor_L_inv = np.stack(self.L_invs)          # (k, p, p)
        self.tensor_means = np.stack(self.means)           # (k, p, 1)
        # 0.5*log|Sigma^-1| = sum(log(diag(L^-1))) por clase, precomputado: (k,)
        self.log_dets = np.log(
            np.diagonal(self.tensor_L_inv, axis1=1, axis2=2)
        ).sum(axis=1)

    def _predict_log_conditionals(self, x):
        unbiased = x - self.tensor_means                   # (k, p, 1)
        y = self.tensor_L_inv @ unbiased                   # (k, p, 1)
        quad = (y ** 2).sum(axis=(1, 2))                   # (k,)
        return self.log_dets - 0.5 * quad

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class EfficientChol(QDA_Chol3):
    """P14) Combina los insights de `EfficientQDA` y `TensorizedChol`:
    sin ciclo for y sin matriz de n x n.

    Para todas las observaciones a la vez (X de (p, n)):
        unbiased = X - mu_k            -> (k, p, n)
        y = L^-1 @ unbiased            -> (k, p, n)
        || y ||^2 por observación      = sum sobre p de y*y -> (k, n)
    La matriz n x n nunca aparece: en vez de y^T @ y (que sería (k, n, n))
    sumamos y*y a lo largo del eje de las p features.
    """

    def _fit_params(self, X, y):
        super()._fit_params(X, y)
        self.tensor_L_inv = np.stack(self.L_invs)          # (k, p, p)
        self.tensor_means = np.stack(self.means)           # (k, p, 1)
        self.log_dets = np.log(
            np.diagonal(self.tensor_L_inv, axis1=1, axis2=2)
        ).sum(axis=1)                                      # (k,)

    def predict(self, X):
        # X: (p, n)
        unbiased = X[np.newaxis, :, :] - self.tensor_means  # (k, p, n)
        y = self.tensor_L_inv @ unbiased                    # (k, p, n)
        quad = np.sum(y * y, axis=1)                        # (k, n)
        log_conditionals = self.log_dets[:, np.newaxis] - 0.5 * quad
        log_posteriori = self.log_a_priori[:, np.newaxis] + log_conditionals
        return np.argmax(log_posteriori, axis=0).reshape(1, -1)
