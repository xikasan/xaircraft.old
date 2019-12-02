# coding: utf-8

import gym
import numpy as np
from cached_property import cached_property


class BaseEnv(gym.Env):

    def __init__(
            self,
            dt,
            name="BaseEnv"
    ):
        super().__init__()
        self.name = name
        self._act_low = 0
        self._act_high = 1

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    @cached_property
    def action_size(self):
        return self.action_space.high.size

    @cached_property
    def observation_size(self):
        return self.observation_space.high.size

    @staticmethod
    def generate_space(low, high):
        high = np.array(high) if type(high) is list else high
        low  = np.array(low)  if type(low)  is list else low
        return gym.spaces.Box(high=high, low=low)

    def clip_action(self, action):
        return np.clip(action, self._act_low, self._act_high)
