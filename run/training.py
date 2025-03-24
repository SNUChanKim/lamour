import os, sys
import datetime
from pathlib import Path
import gymnasium as gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

sys.path.append(str(Path(__file__).resolve().parent.parent))
import envs
from run.config import get_args
from utils.replay_memory import ReplayMemory
from utils.utils import set_seed
from utils.agent import get_agent, load_agent_trained, save_buffer

import pyvirtualdisplay

def set_virtual_display(args):
    if args.server:
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

def make_env(env_name, seed, idx, render_mode=None):
    """Factory function to create environments with different seeds."""
    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        env.reset(seed=seed + idx)
        set_seed(env, seed+idx)
        return env
    return _init

def set_environment(args):
    render_mode = "human" if args.render else None
    env_fns = [make_env(args.env_name, args.seed, i, render_mode if i == 0 and args.render else None) 
                for i in range(args.n_envs)]
    
    # Use AsyncVectorEnv for better performance when n_envs is large
    if args.async_env and args.n_envs > 4:
        return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)

def set_agent(args, env):
    agent = get_agent(env, args)
    if args.load_model:
        load_agent_trained(agent, args)
    return agent

def set_logging(args):
    parent_dir = Path(__file__).parent.parent
    log_dir = parent_dir / 'log' / args.env_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.policy}_seed_{args.seed}_envs_{args.n_envs}"
    writer = SummaryWriter(str(log_dir / run_name))
    return writer

def set_training(args, set_log=True):
    torch.set_num_threads(args.num_threads)
    set_virtual_display(args)
    env = set_environment(args)
    agent = set_agent(args, env)
    memory = ReplayMemory(args.buffer_size, args.seed)
    if set_log:
        writer = set_logging(args)
        return env, agent, memory, writer
    else:
        return env, agent, memory

def evaluate(args, agent, writer, updates):
    print("RUN EVALUATION")
    avg_reward = 0.
    # Always use a single environment for evaluation
    render_mode = "human" if args.render else None
    eval_env = gym.make(args.env_name, render_mode)
    set_seed(eval_env, args.seed + 12345)
    
    for _ in range(args.eval_episodes):
        state, _ = eval_env.reset(seed=args.seed + 12345)
        episode_reward = 0
        done = False
        while not done:
            if args.render:
                eval_env.render()
            action, _ = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = eval_env.step(action) 
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= args.eval_episodes

    writer.add_scalar('avg_reward/evaluation', avg_reward, updates)
    eval_env.close()
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(args.eval_episodes, round(avg_reward, 2)))
    print("----------------------------------------")

def step_vectorized(args, env, agent, memory, writer, states, total_steps):
    batch_size = states.shape[0]
    
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
            writer.add_scalar('deg_uncertainty', 0, total_steps)
            writer.add_scalar('std', np.mean(stds), total_steps)
    
    # Execute actions in all environments
    next_states, rewards, terminated, truncated, infos = env.step(actions)
    dones = np.logical_or(terminated, truncated)
    masks = np.where(truncated, 1.0, np.where(terminated, 0.0, 1.0))
    
    # Add experiences to replay buffer
    aux_rewards = np.zeros_like(rewards)  # Placeholder for auxiliary rewards
    
    for i in range(batch_size):
        memory.push(states[i], 0, actions[i], rewards[i], next_states[i], 0, masks[i])
    
    return next_states, rewards, aux_rewards, dones, memory

def update_agent(args, agent, memory, writer, updates, total_steps):
    """Update agent's parameters if enough samples are collected."""
    if len(memory) > args.batch_size and total_steps >= args.start_steps:
        for _ in range(args.updates_per_step):
            train_start = time.time()
            
            # Update parameters
            critic_loss, policy_loss, entropy_loss, _ = agent.update_parameters(
                memory, args.batch_size, updates
            )
            
            train_end = time.time()
            
            # Log losses
            if updates % 1000 == 0:
                writer.add_scalar('loss/critic', critic_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy', entropy_loss, updates)
                writer.add_scalar('one_step_update_time', train_end - train_start, updates)
                
                if args.automatic_entropy_tuning:
                    writer.add_scalar('alpha', agent.alpha.item(), updates)
                
            # Run evaluation if needed
            if updates % args.eval_interval == 0:
                evaluate(args, agent, writer, updates)
                
            updates += 1
            
    return updates

def save_checkpoint(args, agent, memory):
    """Save agent model and replay buffer."""
    agent.save_model(
        env_name=args.env_name,
        policy=args.policy,
        suffix=f"_{args.seed}"
    )
    if args.save_buffer:
        save_buffer(args, memory)

def train():
    args = get_args()
    env, agent, memory, writer = set_training(args)
    
    total_num_steps = 0
    updates = 0
    
    for i_episode in itertools.count(1):
        episode_rewards = np.zeros(args.n_envs)
        episode_aux_rewards = np.zeros(args.n_envs)
        episode_steps = np.zeros(args.n_envs, dtype=int)
        dones = np.zeros(args.n_envs, dtype=bool)
        
        # Reset all environments
        states, _ = env.reset()
        
        while not np.all(dones):
            if args.render and args.n_envs == 1:
                env.render()
                
            # Step in all environments
            next_states, rewards, aux_rewards, step_dones, memory = step_vectorized(
                args, env, agent, memory, writer, states, total_num_steps
            )
            
            # Update episode information for non-done environments
            active_envs = ~dones
            episode_steps[active_envs] += 1
            episode_rewards[active_envs] += rewards[active_envs]
            episode_aux_rewards[active_envs] += args.aux_coef * aux_rewards[active_envs]
            
            # Mark newly finished environments
            dones = np.logical_or(dones, step_dones)
            
            total_num_steps += 1
            
            # Update agent multiple times
            updates = update_agent(args, agent, memory, writer, updates, total_num_steps)
            
            # Store next state
            states = next_states
            
            # Break if total steps exceeded
            if total_num_steps > args.num_steps:
                break
                
            # Reset done environments if not all environments are done
            if np.any(dones) and not np.all(dones):
                for i, done in enumerate(dones):
                    if done:
                        # Log completed episode
                        writer.add_scalar('train/episode_return', episode_rewards[i], i_episode)
                        writer.add_scalar('train/episode_aux_return', episode_aux_rewards[i], i_episode)
                        writer.add_scalar('train/episode_steps', episode_steps[i], i_episode)
                        # print(f"Env {i}, Episode: {i_episode}, steps: {episode_steps[i]}, "
                        #       f"reward: {round(episode_rewards[i], 2)}")
                        
                        # Reset this specific environment
                        # Note: This requires manual reset for individual environments in a vectorized setup
                        # For simplicity, we'll just track which ones are done and reset all at once when all are done
        
        # Log statistics for all environments
        avg_reward = np.mean(episode_rewards)
        avg_aux_reward = np.mean(episode_aux_rewards)
        avg_steps = np.mean(episode_steps)
        
        writer.add_scalar('train/avg_episode_return', avg_reward, i_episode)
        writer.add_scalar('train/avg_episode_aux_return', avg_aux_reward, i_episode)
        writer.add_scalar('train/avg_episode_steps', avg_steps, i_episode)
        
        print(f"Episode: {i_episode}, total steps: {total_num_steps}, "
              f"avg episode steps: {avg_steps:.2f}, avg reward: {avg_reward:.2f}")
        
        if i_episode % args.save_interval == 0:
            save_checkpoint(args, agent, memory)
        
        if total_num_steps > args.num_steps:
            break
                       
    save_checkpoint(args, agent, memory)
    env.close()

if __name__ == "__main__":
    train()