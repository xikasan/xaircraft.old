from xaircraft.envs.lvaircraft_pitch import LVAircraftPitch

from gym.envs.registration import register

register(
    id='LVAircraftAltitude-v0',
    entry_point='xaircraft.envs.lvaircraft_altitude:LVAircraftAltitudeV0'
)


register(
    id='LVAircraftPitch-v0',
    entry_point='xaircraft.envs.lvaircraft_pitch:LVAircraftPitchV0'
)

register(
    id='LVAircraftPitch-v1',
    entry_point='xaircraft.envs.lvaircraft_pitch:LVAircraftPitchV1'
)

