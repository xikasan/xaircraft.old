# coding: utf-8

import numpy as np
import pandas as pd
import xtools as xt
import xtools.simulation as xsim
from xaircraft.models.lvaircraft import LVAircraftEx
from matplotlib import pyplot as plt


def test_LVAircraft():
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


if __name__ == '__main__':
    test_LVAircraft()
