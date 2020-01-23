# coding: utf-8

import numpy as np
import xtools as xt
import xtools.simulation as xs
from enum import Enum, auto
from xaircraft.models.base import BaseModel
from xaircraft.utils.buffer import DelayBuffer


class FailMode(Enum):

    NORMAL = auto()
    DELAY = auto()
    SATURATION = auto()
    GAIN_REDUCTION = auto()
    RATE_LIMITATION = auto()

    def __init__(self, value):
        self._val = value - 1
        self._mag = 1

    def set_val(self, val):
        self._mag = val
        return self

    def get_val(self):
        return self._mag

    @classmethod
    def get_mode(cls, mode):
        mode = mode.upper()
        for key, fail in cls.__members__.items():
            if key == mode:
                return fail
        raise ValueError("Not defined mode name {} is given.".format(mode))


class CSurface(xs.BaseModel):

    def __init__(
            self,
            dt,
            gain=1.0,
            tau=0.2,
            delay=0.2,
            state_range=[-10, 10],
            speed_range=[-20, 20],
            initial_state=0.0,
            initial_fail_mode=FailMode.NORMAL,
            name="CSurface"
    ):
        super().__init__(dt, name)
        self._K = gain
        self._t = tau
        self.delay = delay
        self._delay_step = int(delay / dt)

        # state
        self._state = None
        self._dx = None
        self._init_state = initial_state
        self._buf = DelayBuffer(self._delay_step)
        # fail mode
        self._fail_mode = None
        self._init_fail_mode = initial_fail_mode

        # env params
        # action space
        self._act_low  = np.min(state_range) / gain
        self._act_high = np.max(state_range) / gain
        # observation space
        self._obs_low  = np.min(state_range)
        self._obs_high = np.max(state_range)
        # state speed
        self._speed_range = xt.as_ndarray([np.min(speed_range), np.max(speed_range)])

    def __call__(self, action):
        fmode = self._fail_mode
        # delay
        action = np.clip(action, self._act_low, self._act_high)
        action = self._buf(action)
        # dynamics
        # fail.gain_reduction
        K = self._K if not fmode == FailMode.GAIN_REDUCTION \
            else self._K * fmode.get_val()
        fn = lambda x: (K * action - x) / self._t
        dx = xs.no_time_rungekutta(fn, self.dt, self._state)
        dx_range = self._speed_range if not fmode == FailMode.RATE_LIMITATION \
            else self._speed_range * fmode.get_val()
        self._dx = np.clip(dx, *dx_range)
        next_state = self._state + self._dx * self.dt
        obs_factor = 1.0 if not fmode == FailMode.SATURATION \
            else fmode.get_val()
        self._state = np.clip(next_state, self._obs_low * obs_factor, self._obs_high * obs_factor)
        return self.get_obs()

    def reset(self):
        self._state = self._init_state
        self._dx = self._init_state * 0.0
        self._buf.reset()

    def get_obs(self):
        return self._state

    def get_full_state(self):
        return xt.as_ndarray([self._state, self._dx])

    def set_fail_mode(self, mode, val=None):
        if val is not None:
            mode.set_val(val)
        self._fail_mode = mode
        if mode == FailMode.DELAY:
            old_buf = self._buf.buffer()
            self._delay_step = int(self._delay_step * mode.get_val())
            self._buf = DelayBuffer(self._delay_step, init_value=self._buf.get_newest())
            for bf in old_buf:
                self._buf(bf)


if __name__ == '__main__':
    FailMode.get_mode("SATURATION")
