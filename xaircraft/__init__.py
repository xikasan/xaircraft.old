from xaircraft.envs.lvaircraft_pitch import LVAircraftPitch

from gym.envs.registration import register


register(
    id='LVAircraft-v0',
    entry_point='xaircraft.envs.statespace.acl_flyby_v0.acl_flyby_v0:ACLFlybyV0Env'
)
