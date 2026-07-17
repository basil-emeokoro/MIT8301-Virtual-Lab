from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class LogisticRegressionScratch(ClassifierMixin, BaseEstimator):
    """Vectorised binary logistic regression trained with gradient descent."""

    def __init__(self, learning_rate: float = 0.08, max_iter: int = 5000,
                 tolerance: float = 1e-7, l2: float = 0.01, threshold: float = 0.5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.l2 = l2
        self.threshold = threshold

    @staticmethod
    def _sigmoid(z):
        z = np.clip(np.asarray(z, dtype=float), -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, X, y):
        probability = np.clip(self._sigmoid(X @ self.coef_ + self.intercept_), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(probability) + (1-y) * np.log(1-probability)) + self.l2 * np.sum(self.coef_**2) / (2*len(y)))

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=float)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must contain only 0 and 1")
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.intercept_ = 0.0
        self.loss_history_ = []
        self.converged_ = False
        previous = np.inf
        for iteration in range(1, self.max_iter + 1):
            probability = self._sigmoid(X @ self.coef_ + self.intercept_)
            error = probability - y
            self.coef_ -= self.learning_rate * ((X.T @ error) / len(y) + self.l2 * self.coef_ / len(y))
            self.intercept_ -= self.learning_rate * float(np.mean(error))
            loss = self._compute_loss(X, y)
            self.loss_history_.append(loss)
            if abs(previous - loss) < self.tolerance:
                self.converged_ = True
                break
            previous = loss
        self.n_iter_ = iteration
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X, accept_sparse=False, dtype=float)
        positive = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
