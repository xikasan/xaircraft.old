# coding: utf-8

import gym
import numpy as np
import xtools as xt
import xtools.simulation as xs
from xaircraft.envs.base import BaseEnv
from xaircraft.models.csurface import CSurface
from xaircraft.models.lvaircraft import LVAircraft


class LVAircraftPitch(BaseEnv):

    def __init__(
            self,
            dt,
            name="LVAircraftPitch"
    ):
        super().__init__(dt, name=name)

        self._elevator = CSurface(
            dt,
            gain=0.67,
            tau=0.02,
            delay=0.06,
            state_range=xt.d2r([-20, 20]),
            speed_range=xt.r2d([-40, 40]),
            name="Elevator"
        )
        self._throttle = CSurface(
            dt,
            gain=1.0,
            tau=1.0,
            delay=0.5,
            deflection_range=[-100, 100],
            speed_range=[-100, 100],
            name="Throttle"
        )
        self._body = LVAircraft(
            dt,
            range_elevator=xt.d2r([-20, 20]),
            range_throttle=[-100, 100],
        )

        # env params
        # action space
        self._act_low  = np.array([self._throttle._act_low,  self._elevator._act_low])
        self._act_high = np.array([self._throttle._act_high, self._elevator._act_high])
        self.action_space = self.generate_space(self._act_low, self._act_high)
        # observation space
        self._obs_low   = np.array(xt.d2r([-20, -20]))
        self._obs_high  = np.array(xt.d2r([ 20,  20]))
        self.observation_space = self.generate_space(self._obs_low, self._obs_high)

    def reset(self):
        self._elevator.reset()
        self._throttle.reset()
        return self.get_obs()

    def step(self, action):
        action = self.clip_action(action)
        action = np.array([self._throttle(action[0]), self._elevator(action[1])])
        next_state = self._body(action)
        return self.get_obs()

    def get_obs(self):
        return self._body.get_obs()[2:]
