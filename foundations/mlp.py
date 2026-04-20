import numpy as np
from numpy.typing import NDArray
from typing import List

class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        num_layers = len(weights)
        curr = x
        for i in range(num_layers - 1):
            curr = np.dot(curr, weights[i]) + biases[i]
            print(f"curr after dot product: {curr}")
            curr = np.maximum(0, curr)
            print(f"curr after relu: {curr}")

        return np.round(np.dot(curr, weights[-1]) + biases[-1], 5)
        