import contextlib
import os
import warnings
import numpy as np
from gymnasium import utils
from envs.mujoco.mujoco_env import MuJocoPyEnv
from gymnasium.spaces import Box
import math
from utils.utils import euler_to_quaternion, euler_from_quaternion

_DEFAULT_VALUE_AT_MARGIN = 0.1
_STAND_HEIGHT = 1.4
_WALK_SPEED = 1
_RUN_SPEED = 10

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 1,
    'distance': 4.0,
    'lookat': np.array((0.0, 0.0, 2.0)),
    'elevation': -20.0,
}


def _sigmoids(x, value_at_1, sigmoid):
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, ' 'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, ' 'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='invalid value encountered in cos')
            cos_pi_scaled_x = np.cos(np.pi * scaled_x)
        return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if np.isscalar(x) else value


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidOODEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 40,
    }

    def __init__(
        self,
        xml_file="humanoid.xml",
        task='run',
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            task,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._move_speed = {
            'walk': _WALK_SPEED,
            'run': _RUN_SPEED,
        }.get(task, _RUN_SPEED)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        if exclude_current_positions_from_observation:
            observation_space = Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float64)
        else:
            observation_space = Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float64)

        xml_file_path = os.path.join(os.path.dirname(__file__), 'assets', xml_file) if not os.path.exists(xml_file) else xml_file
        
        self.return_to_in_distribution = False

        MuJocoPyEnv.__init__(
            self,
            xml_file_path,
            5,
            observation_space=observation_space,
            **kwargs,
        )
        
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if (self._terminate_when_unhealthy and self.return_to_in_distribution) else False
        return terminated
    
    def _get_obs(self):
        joint_angles = self.sim.data.qpos[7:]  # Skip the 7 DoFs of the free root joint.
        head_height = self.sim.data.body_xpos[2, 2]  # ['head', 'z']
        torso_frame = self.sim.data.body_xmat[1].reshape(3, 3)  # ['torso']
        torso_pos = self.sim.data.body_xpos[1]  # ['torso']
        positions = []
        for idx in [16, 10, 13, 7]:  # ['left_hand', 'left_foot', 'right_hand', 'right_foot']
            torso_to_limb = self.sim.data.body_xpos[idx] - torso_pos
            positions.append(torso_to_limb.dot(torso_frame))
        extremities = np.hstack(positions)
        torso_vertical_orientation = self.sim.data.body_xmat[1, [6, 7, 8]]  # ['torso', ['zx', 'zy', 'zz']]
        center_of_mass_velocity = self.sim.data.sensordata[0:3]  # ['torso_subtreelinvel']
        velocity = self.sim.data.qvel

        return np.concatenate(
            [joint_angles, [head_height], extremities, torso_vertical_orientation, center_of_mass_velocity, velocity]
        )

    def _get_reward(self):
        head_height = self.sim.data.body_xpos[2, 2]  # ['head', 'z']
        torso_upright = self.sim.data.body_xmat[1, 8]  # ['torso', 'zz']
        center_of_mass_velocity = self.sim.data.sensordata[0:3]  # ['torso_subtreelinvel']
        control = self.sim.data.ctrl.copy()

        standing = tolerance(head_height, bounds=(_STAND_HEIGHT, float('inf')), margin=_STAND_HEIGHT / 4)
        upright = tolerance(torso_upright, bounds=(0.9, float('inf')), margin=1.9, sigmoid='linear', value_at_margin=0)
        stand_reward = standing * upright
        small_control = tolerance(control, margin=1, value_at_margin=0, sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move
        else:
            com_velocity = np.linalg.norm(center_of_mass_velocity[[0, 1]])
            move = tolerance(
                com_velocity,
                bounds=(self._move_speed, float('inf')),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid='linear',
            )
            move = (5 * move + 1) / 6
            return small_control * stand_reward * move

    def step(self, action):
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self.terminated
        
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "original_reward": reward,
            "xy": xy_position_after,
        }
        if not self.is_healthy:
            reward = 0
        else:
            self.return_to_in_distribution = True

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        self.return_to_in_distribution = False
        
        qx, qy, qz, qw = euler_to_quaternion(0, -math.pi/2, 0)
        qpos = self.init_qpos
        qpos[3] = qw
        qpos[4] = qx
        qpos[5] = qy
        qpos[6] = qz
        
        qpos[2] = 0.12

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)