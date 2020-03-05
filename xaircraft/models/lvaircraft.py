# coding: utf-8

import numpy as np
import xtools as xt
import xtools.simulation as xs
from xaircraft.models.base import BaseModel


class LVAircraftEx(xs.BaseModel):

    H0 = 12192  # m
    U0 = 625    # m/s

    def __init__(
            self,
            dt,
            range_elevator=xt.d2r([-10, 10]),
            range_throttle=[-1, 1],
            dtype=np.float32,
            name="LVAircraftEx"
    ):
        super().__init__(dt, dtype=dtype, name=name)

        # parameters
        self.A, self.B = self.construct_matrices()

    def construct_matrices(self):
        A = np.array([
            #H  U   u        w         Theta     q
            [0, 0,  0,       1,        0,        0],       # H
            [0, 0,  1,       0,        0,        0],       # U
            [0, 0, -0.0225,  0.0022, -32.3819,   0],       # u
            [0, 0, -0.2282, -0.4038,   0,      869],       # w
            [0, 0,  0,       0,        0,        1],       # Theta
            [0, 0, -0.0001, -0.0018,   0,       -0.5518]   # q
        ])
        B = np.array([
            [0,      0],
            [0,      0],
            [0.500,  0],
            [0,     -0.0219],
            [0,      0],
            [0,     -1.2394]
        ], dtype=self.dtype)
        return A.T, B.T


class LVAircraft(xs.BaseModel):

    def __init__(
            self,
            dt,
            range_elevator=xt.d2r([-10, 10]),
            range_throttle=[-100, 100],
            dtype=np.float32,
            name="LVAircraft",
    ):
        super().__init__(dt, dtype=dtype, name=name)

        # parameters
        self._A, self._B = self.construct_matrices()

        # state
        self._x = np.zeros(4, dtype=self.dtype)

        # env params
        # action space
        # U = [throttle, elevator]
        self._act_low  = np.array([np.min(range_throttle), np.min(range_elevator)])
        self._act_high = np.array([np.max(range_throttle), np.max(range_elevator)])
        # observation space
        self._obs_low, self._obs_high = self.generate_inf_range(4)

    def __call__(self, action):
        action = np.clip(action, self._act_low, self._act_high)
        fn = lambda x: x.dot(self._A) + action.dot(self._B)
        dx = xs.no_time_rungekutta(fn, self.dt, self._x)
        self._x += dx * self.dt
        return self._x

    def reset(self):
        self._x = np.zeros(4, dtype=self.dtype)

    def get_state(self):
        return self._x

    def get_u(self):
        return self._x[0]

    def get_v(self):
        return self._x[1]

    def get_theta(self):
        return self._x[2]

    def get_q(self):
        return self._x[3]

    def construct_matrices(self):
        A = np.array([
            [-0.0225,  0.0022, -32.3819,   0],
            [-0.2282, -0.4038,   0,      869],
            [ 0,       0,        0,        1],
            [-0.0001, -0.0018,   0,       -0.5518]
        ], dtype=self.dtype)
        B = np.array([
            [0.500,  0],
            [0,     -0.0219],
            [0,      0],
            [0,     -1.2394]
        ], dtype=self.dtype)
        return A.T, B.T
