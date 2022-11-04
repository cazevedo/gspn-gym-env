from gym.envs.registration import register

register(
    id='gspn-env-v0',
    entry_point='gspn_gym_env.envs:GSPNenv',
)

register(
    id='multi-gspn-env-v0',
    entry_point='gspn_gym_env.envs:MultiGSPNenv',
)

register(
    id='multi-gspn-mmdp-env-v0',
    entry_point='gspn_gym_env.envs:MultiGSPN_MMDPenv',
)