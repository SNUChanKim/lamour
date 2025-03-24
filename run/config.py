import argparse

def get_args():
    parser = argparse.ArgumentParser(description='sero')
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Enable cuda device')
    parser.add_argument(
        '--cuda-device',
        type=int,
        default=0,
        help='PCI_BUS_ID of CUDA device')
    parser.add_argument(
        '--num-threads',
        type=int,
        default=2000,
        help='number of cpu threads used')
    parser.add_argument(
        '--server',
        action='store_true',
        help='Running on server which has no GUI')
    parser.add_argument(
        '--env-name',
        default='AntNormal-v3',
        help='name of gym environment  (default: AntNormal-v3)')
    parser.add_argument(
        '--observation-type',
        default='vector',
        help='type of obeservation | vector | box | (default: vector)')
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--gamma', 
        type=float, 
        default=0.99, 
        help='discount factor  (default: 0.99)')
    parser.add_argument(
        '--drop-p', 
        type=float, 
        default=0.1, 
        help='dropout probability of MCD  (default: 0.1)')
    parser.add_argument(
        '--tau', 
        type=float, 
        default=0.005, 
        help='soft update coefficient  (default: 0.005)')
    parser.add_argument(
        '--alpha', 
        type=float, 
        default=0.2, 
        help='entropy coefficient  (default: 0.2)')
    parser.add_argument(
        '--policy',
        default='sero', 
        help='type of policy | sac | sero | lamour |')
    parser.add_argument(
        '--use-aux-reward',
        action='store_true', 
        help='determine whether to use auxiliary reward')
    parser.add_argument(
        '--use-env-reward',
        action='store_true', 
        help='determine whether to use env reward in OOD states')
    parser.add_argument(
        '--target-update-interval', 
        type=int, 
        default=1, 
        help='interval to update target network of critic (default: 1)')
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=256,
        help='batch size (default: 256)')
    parser.add_argument(
        '--num-steps', 
        type=int, 
        default=1000000,            
        help='maximum number of steps (default: 1000000)')
    parser.add_argument(
        '--updates-per-step', 
        type=int, 
        default=1,
        help='model updates per simulator step (default: 1)')
    parser.add_argument(
        '--start-steps', 
        type=int, 
        default=10000,
        help='Steps sampling random actions (default: 10000)')
    parser.add_argument(
        '--automatic-entropy-tuning', 
        type=bool,
        default=True,
        help='Enable automatic entropy tuning (default: True)')
    parser.add_argument(
        '--hidden-size', 
        type=int, 
        default=256, 
        help='size of hidden unit of neural networks  (default: 256)')
    parser.add_argument(
        '--aux-coef',
        type=float,
        default=0.1,
        help='coefficient for auxiliary reward (default: 0.1)')
    parser.add_argument(
        '--consol-coef',
        type=float,
        default=0.0,
        help='coefficient for consolidating policy (default: 0.0)')
    parser.add_argument(
        '--lr-critic', 
        type=float, 
        default=0.0003, 
        help='learning rate of critic network (default: 0.0003)')
    parser.add_argument(
        '--lr-policy', 
        type=float, 
        default=0.0003, 
        help='learning rate of policy network (default: 0.0003)')
    parser.add_argument(
        '--lr-alpha', 
        type=float, 
        default=0.00003, 
        help='learning rate of entropy coefficient "alpha" (default: 0.00003)')
    parser.add_argument(
        '--save-buffer',
        action='store_true',
        help='define whether save buffer')
    parser.add_argument(
        '--buffer-size', 
        type=int, 
        default=1000000,
        help='size of replay buffer (default: 10000000)')
    parser.add_argument(
        '--eval-interval', 
        type=int, 
        default=5000,
        help='interval between updates for evaluation (default: 5000)')
    parser.add_argument(
        '--save-interval', 
        type=int, 
        default=50,
        help='interval between episodes for save model (default: 50)')
    parser.add_argument(
        '--eval-episodes', 
        type=int, 
        default=5,
        help='number of episodes for evaluation (default: 5)')
    parser.add_argument(
        '--eval', 
        type=bool, 
        default=True,
        help='Evaluates a policy a policy every "args.eval_interval" episode (default: True)')
    parser.add_argument(
        '--load-model',
        action='store_true',
        help='determine if load model')
    parser.add_argument(
        '--actor-path', 
        default='../trained_models/AntNormal-v3/sac/actor_1.pt',
        help='actor model path')
    parser.add_argument(
        '--critic-path', 
        default='../trained_models/AntNormal-v3/sac/critic_1.pt',
        help='critic model path')
    parser.add_argument(
        '--render', 
        action='store_true',
        help='rendering GUI')
    parser.add_argument(
        '--num-evaluation', 
        type=int, 
        default=100,
        help='number of total evaluation (default: 100)')
    parser.add_argument(
        '--eval-retrained', 
        action='store_true', 
        help='determine whether to evaluate retrained agent or trained agent')
    parser.add_argument(
        '--retrain-lamour', 
        action='store_true', 
        help='determine whether to generate lamour code for retraining')
    parser.add_argument(
        '--n-envs', 
        type=int, 
        default=1, 
        help='Number of environments to run in parallel')
    parser.add_argument(
        '--async-env', 
        action='store_true', 
        default=False,
        help='Use AsyncVectorEnv instead of SyncVectorEnv for better performance with many environments')
    
    args = parser.parse_args()
    return args
