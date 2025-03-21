import os, sys
from pathlib import Path
import gymnasium as gym
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import envs
from run.config import get_args
from utils.utils import set_seed
from utils.agent import get_agent, load_agent_trained, load_agent_retrained
from run.training import set_environment, set_virtual_display

import pyvirtualdisplay

def set_eval_agent(args, env):
    agent = get_agent(env, args)
    if args.eval_retrained:
        load_agent_retrained(agent, args)
    else:
        load_agent_trained(agent, args)
    return agent

def set_eval(args):
    set_virtual_display(args)
    render_mode = "human" if args.render else None
    env = gym.make(args.env_name, render_mode=render_mode)
    set_seed(env, args.seed + 12345)
    agent = set_eval_agent(args, env)
    return env, agent

def step(env, agent, state):
    action, std = agent.select_action(state, evaluate=True)

    next_state, reward, terminated, truncated, info = env.step(action) 
    
    next_deg_uncertainty = agent.cal_uncertainty(next_state, original=True)
    aux_reward = -next_deg_uncertainty if reward == 0 else 0
    return next_state, reward, aux_reward, next_deg_uncertainty, terminated, truncated, info

def evaluation():
    args = get_args()    
    env, agent = set_eval(args)
    
    total_episode = 0
    total_reward = 0.0
    total_reward_sq = 0.0
    success = 0
    for i_episode in range(args.num_evaluation):
        
        total_episode += 1
        episode_reward = 0
        episode_uncertainty = 0
        done = False
        state, info = env.reset(seed=args.seed + 12345)

        episode_step = 0
        epsiode_aux_reward = 0
        succ = False
        while not done:
            if args.render:
                env.render()
            
            next_state, reward, aux_reward, next_deg_uncertainty, terminated, truncated, info = step(env, agent, state)
            done = terminated or truncated
            if 'PushChair' in args.env_name:
                success += info['success']
                if info['success']:
                    succ = True
            else:
                success += truncated
                if truncated:
                    succ = True
            episode_step += 1
            epsiode_aux_reward += args.aux_coef*aux_reward
            episode_uncertainty += next_deg_uncertainty
            episode_reward += reward

            state = next_state
        total_reward += episode_reward
        total_reward_sq += episode_reward**2
        print("---------------------------------------------------------------")
        print("Episode: {}, Step: {}, Return: {}, Success: {}".format(i_episode, episode_step, episode_reward, succ))
        print("---------------------------------------------------------------")
    
    env.close()
    
    success_rate = success/args.num_evaluation
    avg_reward = total_reward/args.num_evaluation
    avg_reward_sq = total_reward_sq/args.num_evaluation
    std_reward = np.sqrt(avg_reward_sq - avg_reward**2)
    
    print("==================================================================")
    print(f"Algo: {args.policy}")
    print(f"Avg. Return: {round(avg_reward, 2)}, Standard Deviation: {round(std_reward, 2)}, Success Rate: {round(success_rate, 2)}")
    print("==================================================================")


if __name__ == "__main__":
    evaluation()