# coding: utf-8

from xaircraft.utils.buffer import DelayBuffer
import xtools as xt
import xtools.simulation as xs


def test_DelayBuffer():
    bf = DelayBuffer(5)
    xt.debug("DelayBuffer", bf)

    for t in xs.generate_step_time(10, 1):
        xt.info("time", t)
        r = bf(1)
        print("input", t, "->", "output", r)


if __name__ == '__main__':
    test_DelayBuffer()
