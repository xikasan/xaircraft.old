# coding: utf-8


import numpy as np


class DelayBuffer:

    def __init__(self, delay_step, init_value=0, dtype=np.float32):
        # parameters
        self._capacity = delay_step
        self._init_value = init_value
        self.dtype = dtype
        # state
        self._buf = np.array([init_value] * delay_step, dtype=dtype)
        self._index = 0

    def __call__(self, source):
        self._buf[self._index] = source
        self._index = (self._index + 1) % self._capacity
        return self._buf[self._index]

    def __len__(self):
        return self._capacity

    def reset(self):
        self._buf = np.array([self._init_value] * self._capacity, dtype=self.dtype)
        self._index = 0

    def get_newest(self):
        return self._buf[self._index]

    def buffer(self):
        buf = [self._buf[self._index:], self._buf[:self._index]]
        return np.concatenate(buf, axis=0)

