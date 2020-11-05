from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)

# Mujoco Oracle Counterparts
# ----------------------------------------

# - randomised reward functions

register(
    'AntDirOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2DOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoalOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDirOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVelOracle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelOracleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)
