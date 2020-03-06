# coding: utf-8

import gym
import numpy as np
import xtools as xt
from .base import BaseEnv
from ..models.lvaircraft import LVAircraftEx


class LVAircraftAltitudeV0(BaseEnv):

    def __init__(
            self,
            dt,
            range_target=[-500, 500],
            name="LVAircraftAltitudeV0"
    ):
        super().__init__(dt, name=name)

        self._model = LVAircraftEx(
            dt,
            range_elevator=xt.d2r([-5, 5]),
            range_throttle=[-1.0, 1.0],
            name=name
        )
        self.IX_U = self._model.IX_U
        self.IX_H = self._model.IX_H
        self.IX_u = self._model.IX_u
        self.IX_w = self._model.IX_w
        self.IX_T = self._model.IX_T
        self.IX_q = self._model.IX_q
        self.IX_C = len(self._model.get_observation())

        self.range_target = range_target
        self.target_altitude = 0

        # env params
        self.action_space = self.generate_space(self._model.act_low, self._model.act_high)
        self._obs_low  = np.concatenate([self._model.obs_low,  [self._model.H0 + np.min(self.range_target)]])
        self._obs_high = np.concatenate([self._model.obs_high, [self._model.H0 + np.max(self.range_target)]])
        self.observation_space = self.generate_space(self._obs_low, self._obs_high)

    def reset(self):
        self._model.reset()
        self.target_altitude = float(self._model.H0 + np.random.randint(
            np.min(self.range_target),
            np.max(self.range_target)
        ))
        return self.get_observation()

    def step(self, action):
        action = np.asanyarray(action).astype(self._model.dtype)
        assert action.shape == (2,), "action size must be 2 and only single action is accepted"
        next_obs = self._model(action)

        reward = self.calc_reward(next_obs)
        done = False
        info = {}

        return self.get_observation(), reward, done, info

    def calc_reward(self, obs):
        reward_position = obs[self.IX_H] - self.target_altitude
        reward_position = reward_position / 500
        reward_attitude = obs[self.IX_T]
        reward = - reward_position ** 2 - reward_attitude ** 2
        return reward

    def get_observation(self):
        obs = self._model.get_observation()
        return np.concatenate([obs, [self.target_altitude]])
