# coding: utf-8

import numpy as np
import xtools as xt
import xtools.simulation as xs
from xaircraft.models.base import BaseModel
from xaircraft.utils.buffer import DelayBuffer


class CSurface(BaseModel):

    def __init__(
            self,
            dt,
            gain=1.0,
            tau=0.2,
            delay=0.2,
            deflection_range=[-10, 10],
            speed_range=[-20, 20],
            initial_deflection=0.0,
            name="CSurface"
    ):
        super().__init__(dt, name)
        self.gain = gain
        self.tau = tau
        self.delay = delay
        self._delay_step = int(delay / dt)

        # state
        self._state = 0
        self._init_state = initial_deflection
        self._buf = DelayBuffer(self._delay_step)

        # env params
        # action space
        self._act_low  = np.min(deflection_range) / gain
        self._act_high = np.max(deflection_range) / gain
        # observation space
        self._obs_low  = np.min(deflection_range)
        self._obs_high = np.max(deflection_range)
        # state speed
        self._speed_range = [np.min(speed_range), np.max(speed_range)]

    def __call__(self, action):
        # delay
        action = np.clip(action, self._act_low, self._act_high)
        action = self._buf(action)
        # dynamics
        fn = lambda x: (action - x) / self.tau
        dx = xs.no_time_rungekutta(fn, self.dt, self._state)
        dx = np.clip(dx, *self._speed_range)
        next_state = self._state + dx * self.dt
        self._state = np.clip(next_state, self._obs_low, self._obs_high)
        return self.get_obs()

    def reset(self):
        self._state = self._init_state
        self._buf.reset()

    def get_obs(self):
        return self._state
