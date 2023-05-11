from collections import deque

import numpy as np


class RollingAverage:
    def __init__(self, window_size: int = 10):
        self.values = deque([], window_size)

    def __call__(self, value) -> np.ndarray:
        self.values.append(value)
        return np.mean(self.values, axis=0)

    def reset(self):
        self.values.clear()
