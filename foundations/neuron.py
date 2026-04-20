import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = x @ w.T + b
        relu = max(0, z)
        sigmoid = 1 / (1 + np.exp(-z))

        if activation == "sigmoid": return np.round(sigmoid, 5)
        else: return float(np.round(relu, 5))
