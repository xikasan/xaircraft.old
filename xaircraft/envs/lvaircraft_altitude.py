# coding: utf-8

import gym
import numpy as np
import xtools as xt
import xtools.simulation as xs
from .base import BaseEnv
from ..models.lvaircraft import LVAircraftEx


class LVAircraftAltitudeV0(BaseEnv):

    def __init__(
            self,
            dt,
            range_target=[-2000, 2000],
            sampling_interval=1,
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
        self.action_space = xs.BaseModel.generate_space(self._model.act_low, self._model.act_high)
        self._obs_low  = np.concatenate([self._model.obs_low,  [self._model.H0 + np.min(self.range_target)]])
        self._obs_high = np.concatenate([self._model.obs_high, [self._model.H0 + np.max(self.range_target)]])
        self.observation_space = xs.BaseModel.generate_space(self._obs_low, self._obs_high)

        self.viewer = None

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
        reward_position = reward_position / np.max(self.range_target)
        reward_attitude = obs[self.IX_T]
        reward = - reward_position ** 2 - reward_attitude ** 2
        return reward

    def get_observation(self):
        obs = self._model.get_observation()
        return np.concatenate([obs, [self.target_altitude]])

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        aircraft_length = 30
        aircraft_width  = 10

        target_width = np.abs(self.range_target[0] - self.range_target[1]) * 2
        target_ratio = screen_height / target_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # target line
            l, r, t, b = 0, screen_width, 1, -1
            target_line = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            target_line.set_color(0.4, 0.6, 0.8)
            self.target_trans = rendering.Transform()
            target_line.add_attr(self.target_trans)
            self.viewer.add_geom(target_line)
            # aircraft
            aircraft = rendering.make_capsule(aircraft_length, aircraft_width)
            aircraft.set_color(0.8, 0.4, 0.4)
            self.aircraft_trans = rendering.Transform()
            aircraft.add_attr(self.aircraft_trans)
            self.viewer.add_geom(aircraft)

        xs = self.get_observation()

        # target line
        ht = self.target_altitude - self._model.H0
        ht = ht * target_ratio + screen_height / 2
        self.target_trans.set_translation(0, ht)

        # aircraft
        h = xs[self.IX_H] - self._model.H0
        h = h * target_ratio + screen_height / 2
        x = (screen_width - aircraft_length) / 2
        self.aircraft_trans.set_translation(x, h)
        T = xs[self.IX_T]
        self.aircraft_trans.set_rotation(T)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
