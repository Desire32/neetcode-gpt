import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        weights = np.zeros(X.shape[1],)
        b = 0
        n = X.shape[0]
        
        for epoch in range(epochs):
            y_hat = X @ weights + b
            
            grad_w = (2/n) * X.T @ (y_hat - y)
            grad_b = (2/n) * np.sum(y_hat - y)
            
            weights = weights - lr * grad_w
            b = b - lr * grad_b
            
        return (np.round(weights, 5), round(b, 5))
