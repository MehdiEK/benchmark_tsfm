"""
Rolling-window utilities for forecasting evaluation.

Given a list of time series (each a numpy array of shape (T_i, C)),
`make_forecasting_splits` returns:
    - X_test : List[np.ndarray]  each (T_context_i, C) — full context up to
                                  the prediction point (variable length)
    - y_test : List[np.ndarray]  each (prediction_length, C) — the target

The context for window k is everything from the start of the series up to
(T_train + k * stride), so models with long context windows get to use it.
"""

from typing import List, Optional, Tuple
import numpy as np


def make_forecasting_splits(
    series: List[np.ndarray],
    prediction_length: int,
    n_windows: int = 1,
    stride: Optional[int] = None,
    min_context: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create rolling-window evaluation splits from a list of time series.

    Parameters
    ----------
    series : list of (T_i, C) arrays — full time series (train + test combined)
    prediction_length : int
    n_windows : int
        Number of rolling evaluation windows per series.
    stride : int or None
        Step between consecutive prediction points.
        Defaults to ``prediction_length`` (non-overlapping).
    min_context : int
        Minimum context length required before the first prediction point.

    Returns
    -------
    X_test : list of (T_context, C) arrays — grows with each window
    y_test : list of (prediction_length, C) arrays
    """
    if stride is None:
        stride = prediction_length

    X_test, y_test = [], []

    for ts in series:
        ts = np.asarray(ts)  # (T, C)
        T = ts.shape[0]
        # The last prediction point must end at or before T
        # First prediction point: min_context + prediction_length - 1 <= T - 1
        last_end = T
        for w in range(n_windows):
            pred_end = last_end - (n_windows - 1 - w) * stride
            pred_start = pred_end - prediction_length
            if pred_start < min_context:
                continue
            # Full history as context (variable length)
            X_test.append(ts[:pred_start])
            y_test.append(ts[pred_start:pred_end])

    return X_test, y_test
