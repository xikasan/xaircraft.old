# coding: utf-8

import gym
import numpy as np
import tensorflow as tf
from enum import Enum, auto
import xtools as xt
tk = tf.keras


class AngleType(Enum):
    RADIAN = auto()
    DEGREE = auto()


class DefaultModel:

    def __init__(
            self,
            dt,
            angle_type=None,
            dtype=np.float32,
            name="DefaultModel"
    ):
        self.dt = dt
        self.dtype = dtype
        self.name = name

        self.angle_type = angle_type
        self._internal_angle = AngleType.RADIAN

        self._act_high = np.inf
        self._act_low = -np.inf

    def __call__(self, action):
        raise NotImplementedError()

    def use_radian_internal(self):
        self._internal_angle = AngleType.RADIAN

    def use_degree_internal(self):
        self._internal_angle = AngleType.DEGREE

    def import_angle(self, angle):
        if self.angle_type is not None:
            if self.angle_type != self._internal_angle:
                if self._internal_angle == AngleType.RADIAN:
                    return xt.d2r(angle)
                if self._internal_angle == AngleType.DEGREE:
                    return xt.r2d(angle)
                raise ValueError("internal_angle has unexpected value {}".format(self._internal_angle))
        return angle

    def export_angle(self, angle):
        if self.angle_type is not None:
            if self.angle_type != self._internal_angle:
                if self.angle_type == AngleType.RADIAN:
                    return xt.d2r(angle)
                if self.angle_type == AngleType.DEGREE:
                    return xt.r2d(angle)
                raise ValueError("angle_type has unexpected value {}".format(self.angle_type))
        return angle

    def clip_action(self, action):
        return np.clip(action, self._act_low, self._act_high)
        # return tf.clip_by_value(action, self._act_low, self._act_high)

    def get_variable(self, values, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return tf.Variable(values, dtype=dtype) * 1


if __name__ == '__main__':
    deg = AngleType.DEGREE
    rad = AngleType.RADIAN

    print(dir(xt))
