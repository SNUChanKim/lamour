from gymnasium.envs.registration import registry, register, make, spec
try:
    from envs.mujoco.ant_ood import AntOODEnv
    from envs.mujoco.ant_normal import AntNormalEnv
    from envs.mujoco.walker2d_normal import Walker2dNormalEnv
    from envs.mujoco.walker2d_ood import Walker2dOODEnv
    from envs.mujoco.half_cheetah_normal import HalfCheetahNormalEnv
    from envs.mujoco.half_cheetah_ood import HalfCheetahOODEnv
    from envs.mujoco.hopper_normal import HopperNormalEnv
    from envs.mujoco.hopper_ood import HopperOODEnv
    from envs.mujoco.humanoid_normal import HumanoidNormalEnv
    from envs.mujoco.humanoid_ood import HumanoidOODEnv
    from envs.mani_skill2.envs.push_chair_normal import PushChairNormalEnv
    from envs.mani_skill2.envs.push_chair_ood import PushChairOODEnv
    
except ImportError:
    Box2D = None

register(
    id='AntOOD-v3',
    entry_point='envs:AntOODEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='AntNormal-v3',
    entry_point='envs:AntNormalEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Walker2dNormal-v3",
    entry_point="envs:Walker2dNormalEnv",
    max_episode_steps=1000,
)

register(
    id="Walker2dOOD-v3",
    entry_point="envs:Walker2dOODEnv",
    max_episode_steps=1000,
)

register(
    id="HalfCheetahNormal-v3",
    entry_point="envs:HalfCheetahNormalEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetahOOD-v3",
    entry_point="envs:HalfCheetahOODEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HopperNormal-v3',
    entry_point='envs:HopperNormalEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperOOD-v3',
    entry_point='envs:HopperOODEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HumanoidNormal-v3',
    entry_point='envs:HumanoidNormalEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidOOD-v3',
    entry_point='envs:HumanoidOODEnv',
    max_episode_steps=1000,
)

register(
    id='PushChairNormal-v1',
    entry_point='envs:PushChairNormalEnv',
    max_episode_steps=100,
)

register(
    id='PushChairOOD-v1',
    entry_point='envs:PushChairOODEnv',
    max_episode_steps=400,
)