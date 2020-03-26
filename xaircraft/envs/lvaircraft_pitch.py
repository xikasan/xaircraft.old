# coding: utf-8

import gym
import numpy as np
import xtools as xt
import xtools.simulation as xs
from xaircraft.envs.base import BaseEnv
from xaircraft.models.csurface import CSurface
from xaircraft.models.lvaircraft import LVAircraft, LVAircraftEx


class LVAircraftPitchV0(BaseEnv):

    def __init__(
            self,
            dt=0.1,
            range_target=xt.d2r([-10, 10]),
            dtype=np.float32,
            name="LVAircraftPitchV0"
    ):
        super().__init__(dt, dtype=dtype, name=name)

        self._model = LVAircraftEx(
            dt,
            range_elevator=xt.d2r([-5, 5]),
            range_throttle=[-1.0, 1.0],
            name=name
        )
        self.IX_T = 0
        self.IX_q = 1
        self.IX_C = 2

        self.range_target = np.asanyarray(range_target).astype(self.dtype)
        self._target_width = np.max(range_target) - np.min(range_target)
        self.target_pitch = 0

        # env parameters
        self.action_space = xs.BaseModel.generate_space(self._model.act_low, self._model.act_high)
        self._obs_low  = np.concatenate([self._model.obs_low[ [self._model.IX_T, self._model.IX_q]], [np.min(self.range_target)]])
        self._obs_high = np.concatenate([self._model.obs_high[[self._model.IX_T, self._model.IX_q]], [np.max(self.range_target)]])
        self.observation_space = xs.BaseModel.generate_space(self._obs_low, self._obs_high)

        self.viewer = None

    def reset(self):
        self._model.reset()
        target = np.random.rand(1).astype(self.dtype)
        target = target * self._target_width - np.min(self.range_target)
        self.target_pitch = np.squeeze(target)
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
        IX_T = self.IX_T if len(obs) == 3 else self._model.IX_T
        IX_q = self.IX_q if len(obs) == 3 else self._model.IX_q
        reward_pitch = obs[IX_T] - self.target_pitch
        reward_pitch = reward_pitch / np.max(self.range_target)
        reward_q     = obs[IX_q]
        reward = - reward_pitch ** 2 - reward_q ** 2
        return reward

    def get_observation(self):
        obs = self._model.get_observation()
        obs = obs[[self._model.IX_T, self._model.IX_q]]
        print("get_observation de Env", obs)
        return np.concatenate([obs, [self.target_pitch]])


class LVAircraftPitchV1(LVAircraftPitchV0):

    def __init__(
            self,
            dt=0.1,
            range_target=xt.d2r([-10, 10]),
            target_change_inerval = 10.0,
            name="LVAircraftPitchV1"
    ):
        super().__init__(dt, range_target, name)
        self._target_generator = xs.RandomRectangularCommand(
            np.max(range_target),
            target_change_inerval,
        )
        self._time = 0.0

    def reset(self):
        self._model.reset()
        self._target_generator.reset()
        self.target_pitch = self._target_generator(0.0)
        return self.get_observation()

    def step(self, action):
        # target change
        self._time = xt.round(self._time + self.dt, 2)
        self.target_pitch = self._target_generator(self._time)

        # simulation step
        return super().step(action)



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
