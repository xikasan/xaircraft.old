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


def test_DelayBuffer_get_buf():
    bf = DelayBuffer(5)
    xt.debug("DelayBuffer", bf)

    for step in range(15):
        xt.info("step", step)
        if step == 4:
            print("-"*30)
            break
        r = bf(step)
        print("input", step, "output", r)

    xt.debug("buffer", bf._buf)
    xt.debug("index", bf._index)
    xt.info("buffer", bf.buffer())


if __name__ == '__main__':
    # test_DelayBuffer()
    test_DelayBuffer_get_buf()
