import os, sys
from pathlib import Path
import datetime
import gymnasium as gym
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import envs
from run.config import get_args
from run.training import set_agent, set_virtual_display, set_environment, set_logging, evaluate, save_checkpoint
from utils.utils import convert_push_chair_info_to_state_for_reward
from utils.replay_memory import ReplayMemory

import pyvirtualdisplay

def set_training(args, set_log=True):
    torch.set_num_threads(args.num_threads)
    set_virtual_display(args)
    env = set_environment(args)
    agent = set_agent(args, env)            
    if args.use_env_reward:
        args.policy = args.policy + '_env'
    memory = ReplayMemory(args.buffer_size, args.seed)
    if set_log:
        writer = set_logging(args)
        return env, agent, memory, writer
    else:
        return env, agent, memory

def step_vectorized(args, env, agent, memory, writer, states, total_steps, infos=None):
    batch_size = states.shape[0]
    
    # Calculate uncertainty for all states
    deg_uncertainties = []
    for state in states:
        deg_uncertainties.append(agent.cal_uncertainty(state, original=True))
    deg_uncertainties = np.array(deg_uncertainties)
    
    # Select actions for all environments
    if total_steps < args.start_steps:
        # Random actions for all environments
        actions = np.array([env.single_action_space.sample() for _ in range(batch_size)])
        stds = np.zeros_like(actions)
    else:
        # Get actions from the agent for all environments
        actions_list = []
        stds_list = []
        for state in states:
            action, std = agent.select_action(state)
            actions_list.append(action)
            stds_list.append(std)
        
        actions = np.array(actions_list)
        stds = np.array(stds_list)
        if total_steps % 5000 == 0:
            writer.add_scalar('deg_uncertainty', np.mean(deg_uncertainties), total_steps)
            writer.add_scalar('std', np.mean(stds), total_steps)
    
    # Execute actions in all environments
    next_states, rewards, terminated, truncated, next_infos = env.step(actions)
    dones = np.logical_or(terminated, truncated)
    masks = np.where(truncated, 1.0, np.where(terminated, 0.0, 1.0))
    
    # Calculate next state uncertainties
    next_deg_uncertainties = []
    for next_state in next_states:
        next_deg_uncertainties.append(agent.cal_uncertainty(next_state, original=True))
    next_deg_uncertainties = np.array(next_deg_uncertainties)
    
    # Calculate auxiliary rewards and handle policy-specific logic
    aux_rewards = np.zeros(batch_size)
    is_recovereds = np.zeros(batch_size, dtype=bool)
    modified_rewards = rewards.copy()
    
    for i in range(batch_size):
        # Policy-specific auxiliary reward calculation
        if 'sero' in args.policy:
            aux_rewards[i] = -next_deg_uncertainties[i] if rewards[i] == 0 else 0
        elif 'lamour' in args.policy:
            _state = states[i]
            _next_state = next_states[i]
            if 'PushChair' in args.env_name and infos is not None:
                _state = convert_push_chair_info_to_state_for_reward(infos)
                _next_state = convert_push_chair_info_to_state_for_reward(next_infos)
            aux_rewards[i] = agent.lamour_codes['calculate_reward'](_next_state, actions[i]) if rewards[i] == 0 else 0
            is_recovereds[i] = agent.lamour_codes['is_recovered'](_state)
        
        # print(next_infos)
        if args.use_env_reward and next_infos is not None:
            modified_rewards[i] = next_infos['original_reward'][i]
        
        # Push experiences to memory
        if 'lamour' in args.policy:
            memory.push(states[i], is_recovereds[i], actions[i], modified_rewards[i], 
                        next_states[i], aux_rewards[i], masks[i])
        else:
            memory.push(states[i], deg_uncertainties[i], actions[i], modified_rewards[i], 
                        next_states[i], next_deg_uncertainties[i], masks[i])
    
    return next_states, rewards, aux_rewards, dones, next_infos, memory

def step(args, env, agent, memory, writer, state, total_steps, info=None):
    deg_uncertainty = agent.cal_uncertainty(state, original=True)
    if total_steps < args.start_steps:
        action = env.action_space.sample()
    else:
        action, std = agent.select_action(state)
        writer.add_scalar('deg_uncertainty', deg_uncertainty, total_steps)
        writer.add_scalar('std', std.mean(), total_steps)

    next_state, reward, terminated, truncated, next_info = env.step(action)
    next_deg_uncertainty = agent.cal_uncertainty(next_state, original=True)
    done = terminated or truncated
    mask = 1.0 if truncated else float(not terminated)
    # Calculate auxiliary reward
    aux_reward = 0
    is_recovered = False
    if 'sero' in args.policy:
        aux_reward = -next_deg_uncertainty if reward == 0 else 0
    elif 'lamour' in args.policy:
        _state = state
        _next_state = next_state
        if 'PushChair' in args.env_name:
            _state = convert_push_chair_info_to_state_for_reward(info)
            _next_state = convert_push_chair_info_to_state_for_reward(next_info)
        aux_reward = agent.lamour_codes['calculate_reward'](_next_state, action) if reward == 0 else 0
        is_recovered = agent.lamour_codes['is_recovered'](_state)

    _reward = reward

    if args.use_env_reward:
        _reward = info.get('original_reward', _reward)

    # Push experience into memory
    if 'lamour' in args.policy:
        memory.push(state, is_recovered, action, _reward, next_state, aux_reward, mask)
    else:
        memory.push(state, deg_uncertainty, action, _reward, next_state, next_deg_uncertainty, mask)

    return next_state, reward, aux_reward, done, next_info, memory

def update_agent(args, agent, memory, writer, updates, total_steps):
    """Update agent's parameters if enough samples are collected."""
    if len(memory) > args.batch_size and total_steps >= args.start_steps:
        for _ in range(args.updates_per_step):
            train_start = time.time()
            
            # Update parameters
            critic_loss, policy_loss, entropy_loss, regularization_loss = agent.reupdate_parameters(
                memory, args.batch_size, updates
            )
            
            train_end = time.time()
            
            # Log losses
            if updates % 1000 == 0:
                writer.add_scalar('loss/critic', critic_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy', entropy_loss, updates)
                writer.add_scalar('loss/regularization', regularization_loss, updates)
                writer.add_scalar('one_step_update_time', train_end - train_start, updates)
                
                if args.automatic_entropy_tuning:
                    writer.add_scalar('alpha', agent.alpha.item(), updates)
                
            # Run evaluation if needed
            if updates % args.eval_interval == 0:
                evaluate(args, agent, writer, updates)
                
            updates += 1
            
    return updates

def retrain():
    args = get_args()
        
    # Set up training environment and components
    env, agent, memory, writer = set_training(args)
    
    total_num_steps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_rewards = np.zeros(args.n_envs)
        episode_aux_rewards = np.zeros(args.n_envs)
        episode_steps = np.zeros(args.n_envs, dtype=int)
        dones = np.zeros(args.n_envs, dtype=bool)
        
        # Reset all environments
        states, infos = env.reset(seed=args.seed)
        
        while not np.all(dones):
            if args.render and args.n_envs == 1:
                env.render()
            
            # Step in all environments
            next_states, rewards, aux_rewards, step_dones, infos, memory = step_vectorized(
                args, env, agent, memory, writer, states, total_num_steps, infos
            )
            
            # Update episode information for non-done environments
            active_envs = ~dones
            episode_steps[active_envs] += 1
            episode_rewards[active_envs] += rewards[active_envs]
            episode_aux_rewards[active_envs] += args.aux_coef * aux_rewards[active_envs]
            
            # Mark newly finished environments
            dones = np.logical_or(dones, step_dones)
            
            # Update total steps (count steps from all environments)
            total_num_steps += 1
            
            # Update agent multiple times
            updates = update_agent(args, agent, memory, writer, updates, total_num_steps)
            
            # Store next state
            states = next_states
            
            # Break if total steps exceeded
            if total_num_steps > args.num_steps:
                break
                
            # Log completed episodes if any environment is done but not all
            if np.any(dones) and not np.all(dones):
                for i, done in enumerate(dones):
                    if done:
                        # Log completed episode
                        writer.add_scalar(f'train/episode_return_env_{i}', episode_rewards[i], i_episode)
                        writer.add_scalar(f'train/episode_aux_return_env_{i}', episode_aux_rewards[i], i_episode)
                        writer.add_scalar(f'train/episode_steps_env_{i}', episode_steps[i], i_episode)
        
        # Log average statistics across all environments
        avg_reward = np.mean(episode_rewards)
        avg_aux_reward = np.mean(episode_aux_rewards)
        avg_steps = np.mean(episode_steps)
        
        writer.add_scalar('train/episode_return', avg_reward, i_episode)
        writer.add_scalar('train/episode_aux_return', avg_aux_reward, i_episode)
        writer.add_scalar('train/episode_steps', avg_steps, i_episode)
        print(f"Episode: {i_episode}, total steps: {total_num_steps}, "
              f"avg episode steps: {avg_steps:.2f}, avg reward: {avg_reward:.2f}")
        
        if i_episode % args.save_interval == 0:
            save_checkpoint(args, agent, memory)
        
        if total_num_steps > args.num_steps:
            break
               
    save_checkpoint(args, agent, memory)
    env.close()

if __name__ == "__main__":
    retrain()