# coding: utf-8

import numpy as np
import pandas as pd
import xtools as xt
import xtools.simulation as xs
from matplotlib import pyplot as plt
from xaircraft.envs.lvaircraft_pitch import LVAircraftPitch


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
    test_LVAircraftPitch()
