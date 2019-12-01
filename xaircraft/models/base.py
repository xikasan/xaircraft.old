# coding: utf-8

import numpy as np


class BaseModel:

    def __init__(self, dt, dtype=np.float32, name="BaseModel"):
        self.dt = dt
        self.dtype = dtype
        self.name = name

    def __call__(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def generate_inf_range(self, size):
        low = np.array([-np.inf] * size, dtype=self.dtype)
        high = np.array([np.inf] * size, dtype=self.dtype)
        return low, high
