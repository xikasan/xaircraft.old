# coding: utf-8

import numpy as np
import pandas as pd
import xtools as xt
import xtools.simulation as xs
from matplotlib import pyplot as plt
import gym
import xaircraft
from xaircraft.envs.lvaircraft_pitch import LVAircraftPitch, LVAircraftPitchV0


def test_LVAircraft_Pitch_GymEnv():
    xt.info("test for LVAir Pitch gym")

    dt = 0.1
    due = 60

    env = gym.make("LVAircraftPitch-v1", dt=dt)
    xt.info("env", env)

    Kp = -2.5

    buf = xs.ReplayBuffer({
        "time": 1,
        "act": 2,
        "obs": env.observation_size,
        "reward": 1
    }, capacity=int(due / dt + 1))

    # reset env
    obs = env.reset()
    act = Kp * (obs[env.IX_C] - obs[env.IX_T])
    act = np.array([0, act]).astype(np.float32)
    buf.add(time=0, obs=obs, act=act, reward=env.calc_reward(obs))

    for time in xs.generate_step_time(due, dt):
        error = obs[env.IX_C] - obs[env.IX_T]
        act = Kp * error
        act = np.array([0, act]).astype(np.float32)
        next_obs, reward, done, _ = env.step(act)

        buf.add(time=time, act=act, obs=next_obs, reward=reward)
        obs = next_obs

    result = xs.Retriever(buf.buffer())
    result = pd.DataFrame({
        "time": result("time"),
        "pitch": result("obs", idx=env.IX_T),
        "target": result("obs", idx=env.IX_C),
        "elevator": result("act", idx=1),
        "reward": result("reward")
    })
    fig, axes = plt.subplots(nrows=3, sharex=True)
    result.plot(x="time", y=["target", "pitch"], ax=axes[0])
    result.plot(x="time", y="elevator", ax=axes[1])
    result.plot(x="time", y="reward", ax=axes[2])
    plt.show()


def test_LVAircraftPitch():
    xt.info("test for LVAircraftPitch")

    dt = 0.02
    due = 300.0
    air = LVAircraftPitch(dt)
    xt.debug("LVAircraftPitch", air)
    action = np.array([1, xt.d2r(1)])

    # logger
    log = xs.ReplayBuffer({"time": 1, "command": 2, "state": 2}, capacity=int(due / dt + 1))
    log.add(time=0, command=action, state=air.get_obs())
    xt.debug("log", log)

    for t in xs.generate_step_time(due, dt):
        print("[time]", t)
        next_Tq = air.step(action)
        log.add(time=t, command=action, state=next_Tq)
        xt.debug("[T, q]", next_Tq)
        print("-"*30)

    results = log.buffer()
    results = pd.DataFrame({
        "time": np.squeeze(results["time"]),
        "theta": np.squeeze(results["state"][:, 0]),
        "q": np.squeeze(results["state"][:, 1])
    })

    fig, axes = plt.subplots(nrows=2, sharex=True)
    plt.xlim([0, due])
    axes[0].hlines(0, 0, due)
    axes[1].hlines(0, 0, due)
    results.plot(x="time", y="theta", ax=axes[0])
    results.plot(x="time", y="q",     ax=axes[1])
    plt.show()


if __name__ == '__main__':
    test_LVAircraft_Pitch_GymEnv()
