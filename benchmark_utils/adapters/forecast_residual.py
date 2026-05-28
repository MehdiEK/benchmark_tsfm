"""Forecast-residual anomaly detection adapter.

Uses a forecasting model to predict the next step at every position in the
test series.  The anomaly score at each timestep is the absolute prediction
error (or norm across channels).

This is a zero-shot AD strategy: no labels are required.

Usage
-----
    adapter = ForecastResidualAdapter(forecaster, prediction_length=1)
    # no fit() needed for zero-shot
    scores = adapter.predict(x_test)   # (T,) anomaly scores
"""

import numpy as np
from .base import BaseTSFMAdapter


class ForecastResidualAdapter(BaseTSFMAdapter):
    """Anomaly scoring via one-step-ahead forecast residuals.

    Parameters
    ----------
    forecaster : object with
        ``predict(context: np.ndarray (T, C)) -> np.ndarray (H, C)``
    prediction_length : int
        Number of steps predicted at each position (default 1).
    min_context : int
        Minimum number of past timesteps required before the first prediction.
    """

    def __init__(self, forecaster, prediction_length=1, min_context=10):
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.min_context = min_context

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Score every timestep of a test series.

        Parameters
        ----------
        x : (T, C)

        Returns
        -------
        scores : (T,) float — higher means more anomalous.
            Timesteps before ``min_context`` receive score 0.
        """
        T, C = x.shape
        scores = np.zeros(T, dtype=np.float32)

        for t in range(self.min_context, T):
            context = x[:t]  # (t, C)
            try:
                pred = self.forecaster.predict(context)  # (H, C) or (H,)
                pred = np.asarray(pred).reshape(self.prediction_length, -1)
                actual = x[t: t + self.prediction_length]  # (H, C)
                if actual.shape[0] < self.prediction_length:
                    continue
                error = float(np.mean(np.abs(pred - actual)))
            except Exception:
                error = 0.0
            scores[t] = error

        return scores
