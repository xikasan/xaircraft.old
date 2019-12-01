# coding: utf-8

import numpy as np
import pandas as pd
import xtools as xt
import xtools.simulation as xs
from matplotlib import pyplot as plt
from xaircraft.models.csurface import CSurface


def test_CSurface():
    xt.info("test for CSurface")

    # parameters
    dt = 0.02
    due = 10.0

    # model
    surface = CSurface(dt)
    xt.debug("CSurface", surface)
    action = 2.0

    # buffer
    log = xs.ReplayBuffer({"time": 1, "command": 1, "deflection": 1}, capacity=int(due/dt+1))
    log.add(time=0, command=action, deflection=surface.get_obs())
    xt.debug("log", log)

    for t in xs.generate_step_time(due, dt):
        if (int(t*100) % 200) == 0:
            action *= -1
        obs = surface(action)
        log.add(time=t, command=action, deflection=obs)

    result = pd.DataFrame({key: np.squeeze(val) for key, val in log.buffer().items()})
    result.plot(x="time", y=["command", "deflection"])
    plt.show()


if __name__ == '__main__':
    test_CSurface()
