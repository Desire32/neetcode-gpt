import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = y_true.shape[0]
        # y_true: true labels (0 or 1)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        return np.round(-1/n * np.sum(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred)), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = y_true.shape[0]
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        return np.round(-1/n * (np.sum(y_true * np.log(y_pred + 1e-7))), 4)
