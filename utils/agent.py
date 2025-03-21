
import pickle
import os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithms.algorithms import SAC, SeRO, LaMoUR

def get_agent(env, args):
    obs_idx = 1 if len(env.observation_space.shape) > 1 else 0
    if args.observation_type == 'vector':
        num_inputs = env.observation_space.shape[obs_idx]
    elif args.observation_type == 'box':
        num_inputs = env.observation_space.shape[obs_idx:]
    else:
        raise NotImplementedError
    
    if 'sac' in args.policy:
        agent = SAC(num_inputs, env.action_space, args)
    elif 'sero' in args.policy:
        agent = SeRO(num_inputs, env.action_space, args)
    elif 'lamour' in args.policy:
        agent = LaMoUR(num_inputs, env.action_space, args)
    else:
        raise NotImplemented("{} is not implemented.".format(args.policy))

    return agent

def load_agent_trained(agent, args):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      
    suffix="{}".format("_" + str(args.seed))
    env_name = args.env_name
    if 'OOD' in env_name:
        env_name = env_name.replace('OOD', 'Normal')
    
    args.critic_path = "{}/trained_models/{}/{}/critic{}.pt".format(parent_dir, env_name, args.policy, suffix)
    args.actor_path = "{}/trained_models/{}/{}/actor{}.pt".format(parent_dir, env_name, args.policy, suffix)
    mapping_path = "{}/trained_models/{}/{}/mapping{}.pt".format(parent_dir, env_name, args.policy, suffix)
    agent.load_model(args, args.actor_path, args.critic_path, mapping_path, args.load_model)

def load_agent_retrained(agent, args):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    suffix="{}".format("_" + str(args.seed))
    original_suffix = suffix
    
    if 'sero' in args.policy:
        policy_original = 'sero'
    elif 'lamour' in args.policy:
        policy_original = 'lamour'    
    else:
        policy_original = args.policy
        
    if 'Normal' in args.env_name:
        normal_env_name = args.env_name
        ood_env_name = args.env_name.replace('Normal', 'OOD')
    elif 'OOD' in args.env_name:
        ood_env_name = args.env_name
        normal_env_name = args.env_name.replace('OOD', 'Normal')
    else:
        raise NotImplementedError("Environment is not implemented for evaluation.")
    
    args.critic_path = "{}/trained_models/{}/{}/critic{}.pt".format(parent_dir, ood_env_name, args.policy, suffix)
    args.actor_path = "{}/trained_models/{}/{}/actor{}.pt".format(parent_dir, ood_env_name, args.policy, suffix)
    mapping_path = "{}/trained_models/{}/{}/mapping{}.pt".format(parent_dir, ood_env_name, args.policy, suffix)
    original_actor_path = "{}/trained_models/{}/{}/actor{}.pt".format(parent_dir, normal_env_name, policy_original, original_suffix)
    original_mapping_path = "{}/trained_models/{}/{}/mapping{}.pt".format(parent_dir, normal_env_name, policy_original, original_suffix)
    
    agent.load_model(args, args.actor_path, args.critic_path, mapping_path, args.load_model, original_actor_path=original_actor_path, original_mapping_path=original_mapping_path)
    
    
def load_buffer(args):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      
    
    suffix="{}".format("_" + str(args.seed))
    buffer_path = "{}/trained_models/{}/{}/buffer_{}.obj".format(parent_dir, args.env_name, args.policy, suffix)
    filehandler = open(buffer_path, 'rb')
    buffer = pickle.load(filehandler)
    print("Load Buffer from {}".format(buffer_path))
    return buffer

def save_buffer(args, buffer):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('{}/trained_models/{}/{}'.format(parent_dir, args.env_name, args.policy), exist_ok=True)
    
    suffix="{}".format("_" + str(args.seed))
    buffer_path = "{}/trained_models/{}/{}/buffer_{}.obj".format(parent_dir, args.env_name, args.policy, suffix)
    filehandler = open(buffer_path, 'wb')
    pickle.dump(buffer, filehandler)
    