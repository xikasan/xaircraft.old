# coding: utf-8

import gym
import xaircraft
import numpy as np
import pandas as pd
import xtools as xt
import xtools.simulation as xsim
from xaircraft.models.lvaircraft import LVAircraftEx
from matplotlib import pyplot as plt


def test_LVAircraft_Model():
    xt.info("test for LVAircraft")

    dt = 0.05
    env = LVAircraftEx(dt, range_elevator=xt.d2r([-3, 3]))
    obs = env.reset()

    due = 300

    Ht = 12500
    Kp = -0.001

    rb = xsim.ReplayBuffer({
        "time": 1,
        "tgt": 1,
        "obs": 6
    }, capacity=int(due / dt + 1))

    rb.add(time=0, tgt=Ht, obs=obs)
    for time in xsim.generate_step_time(due, dt):
        H = obs[env.IX_H]
        error = Ht - H
        act = error * Kp
        act = np.array([0, act]).astype(env.dtype)
        obs = env(act)
        rb.add(time=time, tgt=Ht, obs=obs)

    result = xsim.Retriever(rb.buffer())
    result = pd.DataFrame({
        "time": result("time"),
        "target": result("tgt"),
        "altitude": result("obs", idx=env.IX_H)
    })
    result.plot(x="time")
    plt.show()


def test_LVAircraft_Altitude_Env():
    xt.info("test for LVAircraft Altitude Env")
    from xaircraft.envs.lvaircraft_altitude import LVAircraftAltitudeV0

    dt = 0.1
    due = 60

    env = LVAircraftAltitudeV0(dt)
    xt.info("env", env)

    Kp = -0.001

    rb = xsim.ReplayBuffer({
        "time": 1,
        "obs": env.observation_size,
        "reward": 1
    }, capacity=int(due / dt + 1))

    obs = env.reset()
    rb.add(time=0, obs=obs, reward=env.calc_reward(obs))

    for time in xsim.generate_step_time(due, dt):
        H  = obs[env.IX_H]
        Hc = obs[env.IX_C]
        er = Hc - H
        de = er * Kp
        ds = np.array([0, de])
        obs, reward, _, _ = env.step(ds)

        rb.add(time=time, obs=obs, reward=reward)

    result = xsim.Retriever(rb.buffer())
    result = pd.DataFrame({
        "time": result("time"),
        "command":  result("obs", idx=env.IX_C),
        "altitude": result("obs", idx=env.IX_H),
        "reward":   result("reward")
    })

    fig, axes = plt.subplots(nrows=2, sharex=True)
    result.plot(x="time", y=["command", "altitude"], ax=axes[0])
    result.plot(x="time", y="reward", ax=axes[1])
    plt.show()


def test_LVAircraft_Altitude_GymEnv():
    xt.info("test for LVAir Alt gym registration")

    dt = 0.1
    due = 100

    env = gym.make("LVAircraftAltitude-v0", dt=dt)
    xt.info("env", env)
    exit()

    Kp = -0.001

    rb = xsim.ReplayBuffer({
        "time": 1,
        "obs": env.observation_size,
        "reward": 1
    }, capacity=int(due / dt + 1))

    obs = env.reset()
    rb.add(time=0, obs=obs, reward=env.calc_reward(obs))

    for time in xsim.generate_step_time(due, dt):
        H  = obs[env.IX_H]
        Hc = obs[env.IX_C]
        er = Hc - H
        de = er * Kp
        ds = np.array([0, de])
        obs, reward, _, _ = env.step(ds)

        env.render()
        rb.add(time=time, obs=obs, reward=reward)

    env.close()

    result = xsim.Retriever(rb.buffer())
    result = pd.DataFrame({
        "time": result("time"),
        "command":  result("obs", idx=env.IX_C),
        "altitude": result("obs", idx=env.IX_H),
        "reward":   result("reward")
    })

    fig, axes = plt.subplots(nrows=2, sharex=True)
    result.plot(x="time", y=["command", "altitude"], ax=axes[0])
    result.plot(x="time", y="reward", ax=axes[1])
    # plt.show()


if __name__ == '__main__':
    # test_LVAircraft_Altitude_Env()
    test_LVAircraft_Altitude_GymEnv()
