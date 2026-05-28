"""Linear probe adaptation strategy.

A foundation model encoder extracts embeddings; a linear classifier (or
regressor) is trained on top.  Suitable for classification and — with a
threshold on reconstruction error — anomaly detection.

Usage
-----
    adapter = LinearProbeAdapter(encoder, task="classification", n_classes=5)
    adapter.fit(X_train, y_train)        # called inside Solver.run()
    label = adapter.predict(x)           # called by objective
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .base import BaseTSFMAdapter


class LinearProbeAdapter(BaseTSFMAdapter):
    """Frozen encoder + linear head.

    Parameters
    ----------
    encoder : object with ``encode(x: np.ndarray (T, C)) -> np.ndarray (D,)``
    task : {"classification", "anomaly_detection"}
    n_classes : int, required when task == "classification"
    max_iter : int
        Maximum iterations for the logistic regression solver.
    """

    def __init__(self, encoder, task="classification", n_classes=None,
                 max_iter=1000):
        self.encoder = encoder
        self.task = task
        self.n_classes = n_classes
        self.max_iter = max_iter
        self._label_enc = LabelEncoder()

    def fit(self, X_train, y_train, **kwargs):
        embeddings = np.stack([self.encoder.encode(x) for x in X_train])

        if self.task == "classification":
            y_enc = self._label_enc.fit_transform(y_train)
            self._head = LogisticRegression(
                max_iter=self.max_iter, multi_class="auto"
            )
            self._head.fit(embeddings, y_enc)

        elif self.task == "anomaly_detection":
            # Train a reconstruction baseline: predict embedding from itself
            # (identity ridge) then use residual norm as anomaly score.
            # Participants can replace with a more principled approach.
            self._train_embeddings = embeddings
            self._train_mean = embeddings.mean(axis=0)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        emb = self.encoder.encode(x)

        if self.task == "classification":
            label_enc = self._head.predict([emb])[0]
            return int(self._label_enc.inverse_transform([label_enc])[0])

        elif self.task == "anomaly_detection":
            # Score: L2 distance from the training mean embedding,
            # broadcast to every timestep (uniform window score).
            score = float(np.linalg.norm(emb - self._train_mean))
            return np.full(x.shape[0], score, dtype=np.float32)

        raise ValueError(f"Unknown task: {self.task}")
