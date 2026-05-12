import numpy as np

from src.models.metrics import regression_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([0.1, 0.2, 0.3])

    metrics = regression_metrics(y, y)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
