import math
import torch
import numpy as np
import random

def convert_push_chair_info_to_state_for_reward(info: dict) -> np.ndarray:
    state = np.empty(5)
    state[0] = info['dist_ee_to_chair']
    state[1] = info['dist_chair_to_target']
    state[2] = info['chair_tilt']
    state[3] = info['chair_vel_norm']
    state[4] = info['chair_ang_vel_norm']
    return state

def get_original_task(env_name: str) -> str:
    """
    Retrieve the original task description based on the environment name.

    Args:
        env_name (str): The name of the environment.

    Returns:
        str: A description of the original task.

    Raises:
        Exception: If the environment is not intended for the experiments.
    """
    if any(env in env_name for env in ['Ant', 'HalfCheetah', 'Hopper', 'Walker2d', 'Humanoid']):
        return 'Run forward'
    elif 'PushChair' in env_name:
        return 'Push a chair to a target location on the ground (indicated by a red hemisphere) and prevent it from falling over.'
    else:
        raise Exception('The environment is not intended for our experiments.')

def get_env_name_for_llm(env_name):
    for env in ['Ant', 'HalfCheetah', 'Hopper', 'Walker2d']:
        if env in env_name:
            return f'gym MuJoCo {env} environment'
    
    if 'Humanoid' in env_name:
        return 'DM Control Suite Humanoid environment'
    
    if 'PushChair' in env_name:
        return 'ManiSkill2 PushChair environment'
    else:
        raise Exception('The environment is not intended for our experiments.')
        
def extract_env_name_symbol(env_name):
    if 'Normal' in env_name:
        env_name_sym = env_name.split('Normal')[0].lower()
    elif 'OOD' in env_name:
        env_name_sym = env_name.split('OOD')[0].lower()
    else:
        raise Exception('The environment is not intended for our experiments.')
    
    return env_name_sym

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2*math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5*z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def set_seed(env, seed):
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_obs_dim(observation_space):
    if len(observation_space.shape) == 0:
        obs_shape = 1
    elif len(observation_space.shape) >= 2:
        obs_shape = 1
        for i in range(len(observation_space.shape)):
            obs_shape *= observation_space.shape[i]
    else:
        obs_shape = observation_space.shape[0]
    return obs_shape

def get_action_dim(action_space):
    if len(action_space.shape) == 0:
        action_shape = 1
    elif len(action_space.shape) > 1:
        action_shape = action_space.shape[1]
    else:
        action_shape = action_space.shape[0]
    
    return action_shape    
 
def euler_from_quaternion(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # Half angles
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    # Quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w