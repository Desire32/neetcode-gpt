import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        max_v = z - np.max(z)
        exp = np.exp(max_v)
        return np.round(exp / np.sum(exp), 4)
