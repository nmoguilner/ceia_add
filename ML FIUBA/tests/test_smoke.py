"""Test mínimo: el estimador se construye correctamente desde el config."""
from src.models.train import build_estimator


def test_build_estimator():
    cfg = {"seed": 42, "task": "classification",
           "model": {"type": "random_forest", "params": {"n_estimators": 10}}}
    model = build_estimator(cfg)
    assert model.n_estimators == 10
