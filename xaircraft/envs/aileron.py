# coding: utf-8

import numpy as np
import xtools as xt
from xaircraft.envs.base import BaseEnv
from xaircraft.models.csurface import CSurface


class MuPAL_Aileron(BaseEnv):

    def __init__(
            self,
            dt,
            use_degree=False,
            name="Aileron"
    ):
        super().__init__(dt, name=name)

        state_range = xt.as_ndarray([-10, 10])
        speed_range = xt.as_ndarray([-20, 20])
        state_range = state_range if use_degree else xt.d2r(state_range)

        self._dynamics = CSurface(
            dt,
            gain=0.67,
            tau=0.12,
            delay=0.06,
            state_range=state_range if use_degree else xt.d2r(state_range),
            speed_range=speed_range if use_degree else xt.d2r(speed_range),
            name="AileronBody"
        )

        # Env spaces
        # action
        self._act_low  = self._dynamics._act_low
        self._act_high = self._dynamics._act_high
        self.action_space = self.generate_space(self._act_low, self._act_high)

        # observation
        self._obs_low  = self._dynamics._obs_low
        self._obs_high = self._dynamics._obs_high
        self.observation_space = self.generate_space(self._obs_low, self._obs_high)

    def reset(self):
        self._dynamics.reset()
        return self._dynamics.get_obs()

    def step(self, action):
        action = self.clip_action(action)
        self._dynamics(action)
        return self.get_obs()

    def get_obs(self):
        return self._dynamics.get_obs()

    def get_full_state(self):
        return self._dynamics.get_full_state()
