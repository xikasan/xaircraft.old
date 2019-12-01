# coding: utf-8

import numpy as np
import xtools as xt
from xaircraft.models.lvaircraft import LVAircraft


def test_LVAircraft():
    xt.info("test for LVAircraft")

    dt = 0.05
    lvair = LVAircraft(dt)
    lvair(np.ones(2))


if __name__ == '__main__':
    test_LVAircraft()
