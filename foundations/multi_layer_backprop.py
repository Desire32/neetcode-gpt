import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
            x: List[float],
            W1: List[List[float]], b1: List[float],
            W2: List[List[float]], b2: List[float],
            y_true: List[float]) -> dict:

        # Convience convertation
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)


        # 1. Forward
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2

        L = np.mean((z2 - y_true) ** 2)

        # 2. Backward
        dZ2 = 2 * (z2 - y_true) / len(y_true)
        dA1 = W2.T @ dZ2

        dZ1 = dA1 * (z1 > 0)
        
        dW2 = np.outer(dZ2, a1)
        db2 = dZ2

        dW1 = np.outer(dZ1, x) + 0.0 # what a trick
        db1 = dZ1

        return {
            'loss': round(float(L), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
