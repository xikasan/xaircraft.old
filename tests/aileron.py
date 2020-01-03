# coding: utf-8

import numpy as np
import pandas as pd
import xtools as xt
import xtools.simulation as xs
from matplotlib import pyplot as plt
from xaircraft.envs.aileron import MuPAL_Aileron


def test_MuLAP_Aileron():
    xt.info("test for MuPAL_Aileron")

    dt = 0.02
    due = 10.0

    aileron = MuPAL_Aileron(dt, use_degree=True)
    aileron.reset()
    xt.info("MuPAL_Aileron", aileron)
    action = 2.0

    # logger
    logger = xs.ReplayBuffer({
        "time": 1,
        "command": 1,
        "aileron": 1
    }, capacity=int(due / dt + 1))
    logger.add(time=0.0, command=action, aileron=aileron.get_obs())

    for time in xs.generate_step_time(due, dt):
        if (int(time*100) % 100) == 0:
            action *= -1.0
        obs_ = aileron.step(action)
        print("time:", time, "\t command:", action, "\t aileron:", obs_)
        logger.add(time=time, command=action, aileron=obs_)

    result = pd.DataFrame({key: np.squeeze(val) for key, val in logger.buffer().items()})
    result.plot(x="time", y=["command", "aileron"])
    plt.show()


if __name__ == '__main__':
    test_MuLAP_Aileron()
