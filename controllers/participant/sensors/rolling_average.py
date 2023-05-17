from collections import deque

import numpy as np


class RollingAverage:
    def __init__(self, window_size: int = 10):
        self.values = deque([], window_size)
        self.__mean = np.zeros(3, dtype=np.float32)

    def __call__(self, value) -> np.ndarray:
        self.append(value)
        self.update()
        return self.__mean

    @property
    def mean(self) -> np.ndarray:
        return self.__mean

    def append(self, value):
        self.values.append(value)

    def update(self):
        self.__mean = np.mean(self.values, axis=0)

    def update_and_get_mean(self) -> np.ndarray:
        self.update()
        return self.__mean

    def reset(self):
        self.values.clear()
        self.__mean = np.zeros_like(self.__mean)
