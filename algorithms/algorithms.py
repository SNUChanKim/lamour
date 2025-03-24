import os, sys
from pathlib import Path
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import soft_update, hard_update, get_action_dim, extract_env_name_symbol, get_original_task, get_env_name_for_llm
from algorithms.model import QNetwork, StochasticPolicy, SeROPolicy
from lamour.lamour import LaMoUR as lamour

class BaseAgent:
    """
    A base class for reinforcement learning agents, providing common functionalities such as
    action selection, model saving, and model loading.

    Methods:
        __init__(num_inputs, action_space, args):
            Initializes the BaseAgent with hyperparameters and model configurations.

        select_action(state, evaluate=False):
            Selects an action based on the given state using the policy. Supports both
            exploration and evaluation modes.

        save_model(env_name, policy, suffix="", actor_path=None, critic_path=None):
            Saves the actor and critic models to the specified or default paths.

        load_model(args, actor_path, critic_path):
            Loads the actor and critic models from the specified paths.
    """
    
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize the BaseAgent with hyperparameters and model configurations.

        Args:
            num_inputs (int): The number of input features in the state space.
            action_space (gym.Space): The action space of the environment.
            args (Namespace): A namespace containing various hyperparameters and configuration options.
                - gamma (float): Discount factor for future rewards.
                - tau (float): Target smoothing coefficient.
                - alpha (float): Entropy regularization coefficient.
                - target_update_interval (int): Interval for updating target networks.
                - automatic_entropy_tuning (bool): Whether to automatically tune entropy.
                - cuda_device (int): The CUDA device ID.
                - cuda (bool): Whether to use CUDA for computation.
                - hidden_size (int): The size of hidden layers in the neural network.
        """
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device(f"cuda:{args.cuda_device}" if args.cuda else "cpu")
        self.num_inputs = num_inputs
        self.num_actions = get_action_dim(action_space)
        self.hidden_size = args.hidden_size
        self.args = args

    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.

        Args:
            state (np.ndarray): The current state of the environment.
            evaluate (bool, optional): If True, selects the action deterministically for evaluation;
                otherwise, samples an action stochastically for exploration. Defaults to False.

        Returns:
            tuple: 
                - action (np.ndarray): The selected action.
                - std (np.ndarray): The standard deviation of the action distribution.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            action, _, _, std, _ = self.policy(state)
        else:
            _, _, action, std, _ = self.policy(state)
        return action.clone().detach().cpu().numpy()[0], std.clone().detach().cpu().numpy()[0]

    def save_model(self, env_name, policy, suffix="", actor_path=None, critic_path=None):
        """
        Save the actor and critic models to files.

        Args:
            env_name (str): The name of the environment.
            policy (str): The name of the policy type.
            suffix (str, optional): A suffix to append to the filenames. Defaults to an empty string.
            actor_path (str, optional): The file path to save the actor model. If None, uses a default path.
            critic_path (str, optional): The file path to save the critic model. If None, uses a default path.
        """
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs('{}/trained_models/{}/{}'.format(parent_dir, env_name, policy), exist_ok=True)
        
        if actor_path is None:
            actor_path = "{}/trained_models/{}/{}/actor{}.pt".format(parent_dir, env_name, policy, suffix)
        if critic_path is None:
            critic_path = "{}/trained_models/{}/{}/critic{}.pt".format(parent_dir, env_name, policy, suffix)
        
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def load_model(self, args, actor_path, critic_path):
        """
        Load the actor and critic models from files.

        Args:
            args (Namespace): A namespace containing configuration options, including CUDA device ID.
            actor_path (str): The file path of the actor model to load.
            critic_path (str): The file path of the critic model to load.
        """
        print('Loading models from {} and {}'.format(actor_path, critic_path))
         
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
            self.policy.eval()
            
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location='cuda:{}'.format(args.cuda_device)))
            self.critic.eval()

class SAC(BaseAgent):
    """
    An implementation of the Soft Actor-Critic (SAC) algorithm, extending the BaseAgent class. 
    Includes functionalities for training, uncertainty calculation, and model loading.

    Methods:
        __init__(num_inputs, action_space, args):
            Initializes the SAC agent, including networks, optimizers, and entropy tuning.
        
        cal_uncertainty(state, original=False):
            Calculates the uncertainty of the agent's predictions. Placeholder implementation.
        
        update_parameters(memory, batch_size, updates):
            Updates the agent's critic, policy, and entropy parameters using a batch of experience.

        reupdate_parameters(memory, batch_size, updates):
            Calls the `update_parameters` method and returns the update results. 
            Designed for modular extensions.

        load_model(args, actor_path, critic_path, mapping_path=None, retrain=False, original_actor_path=None, original_mapping_path=None):
            Loads the trained models (actor and critic) from specified paths, with support for retraining.
    """
    
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize the SAC agent.

        Args:
            num_inputs (int): Number of input features in the state space.
            action_space (gym.Space): Action space of the environment.
            args (Namespace): Configuration parameters for the SAC algorithm.
                - gamma (float): Discount factor for future rewards.
                - tau (float): Target network update coefficient.
                - alpha (float): Initial entropy coefficient.
                - lr_alpha (float): Learning rate for entropy coefficient optimization.
                - lr_critic (float): Learning rate for critic network.
                - lr_policy (float): Learning rate for policy network.
                - automatic_entropy_tuning (bool): Whether to tune entropy automatically.
                - observation_type (str): Type of observation used by the agent.
                - drop_p (float): Dropout probability for policy network.
        """
        super().__init__(num_inputs, action_space, args)
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.tensor([np.log(args.alpha)], requires_grad=True, device=self.device, dtype=torch.float32)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr_alpha)

        self.critic = QNetwork(num_inputs, self.num_actions, self.hidden_size, args.observation_type).to(self.device)
        self.critic_target = QNetwork(num_inputs, self.num_actions, self.hidden_size, args.observation_type).to(self.device)
        self.policy = StochasticPolicy(num_inputs, self.num_actions, self.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)
        hard_update(self.critic_target, self.critic)

    def cal_uncertainty(self, state, original=False):
        """
        Calculate the uncertainty of the agent's predictions.

        Args:
            state (np.ndarray): The current state of the environment.
            original (bool, optional): Flag to determine the mode of uncertainty calculation.
                Defaults to False.

        Returns:
            int: A placeholder value of 0.
        """
        return 0
    
    def update_parameters(self, memory, batch_size, updates):
        """
        Update the critic, policy, and entropy parameters using a batch of experience.

        Args:
            memory (ReplayBuffer): The replay buffer containing experience tuples.
            batch_size (int): Number of samples to draw from the replay buffer.
            updates (int): The current update step count.

        Returns:
            tuple: 
                - critic_loss (float): Loss value for the critic.
                - policy_loss (float): Loss value for the policy.
                - alpha_loss (float): Loss value for the entropy coefficient.
                - extra (int): Placeholder value of 0.
        """
        state_batch, _, action_batch, reward_batch, next_state_batch, _, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Update critic
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                next_action, next_log_pi, _, _, _ = self.policy(next_state_batch)
                target_q_value = reward_batch + mask_batch * self.gamma * (torch.min(*self.critic_target(next_state_batch, next_action)) - self.alpha * next_log_pi)
            current_q1, current_q2 = self.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Update policy
            pi, log_pi, _, _, _ = self.policy(state_batch)
            policy_loss = ((self.alpha * log_pi) - torch.min(*self.critic(state_batch, pi))).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Update alpha
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.).to(self.device)

            # Soft update targets
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

            return critic_loss.item(), policy_loss.item(), alpha_loss.item(), 0
    
    def reupdate_parameters(self, memory, batch_size, updates):
        """
        Re-update the agent's parameters by calling the `update_parameters` method.

        Args:
            memory (ReplayBuffer): The replay buffer containing experience tuples.
            batch_size (int): Number of samples to draw from the replay buffer.
            updates (int): The current update step count.

        Returns:
            tuple: 
                - critic_loss (float): Loss value for the critic.
                - policy_loss (float): Loss value for the policy.
                - alpha_loss (float): Loss value for the entropy coefficient.
                - extra (int): Placeholder value of 0.
        """
        critic_loss, policy_loss, alpha_loss, _ = self.update_parameters(memory, batch_size, updates)
        return critic_loss, policy_loss, alpha_loss, 0
    
    def load_model(self, args, actor_path, critic_path, mapping_path=None, retrain=False, original_actor_path=None, original_mapping_path=None):
        """
        Load the trained actor and critic models from specified paths.

        Args:
            args (Namespace): Configuration parameters, including the CUDA device.
            actor_path (str): Path to the saved actor model.
            critic_path (str): Path to the saved critic model.
            mapping_path (str, optional): Path to a mapping file. Defaults to None.
            retrain (bool, optional): If True, reloads the actor model for further training. Defaults to False.
            original_actor_path (str, optional): Path to the original actor model for reference. Defaults to None.
            original_mapping_path (str, optional): Path to the original mapping file for reference. Defaults to None.
        """
        if retrain:
            print('Loading models from {}'.format(actor_path))
            if actor_path is not None:
                self.policy.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy.train()
                
        else:
            super().load_model(args, actor_path, critic_path)
    
class SeRO(SAC):
    """
    An implementation of the SeRO algorithm, extending the SAC class with added functionality
    for handling uncertainty and auxiliary rewards.

    Methods:
        __init__(num_inputs, action_space, args):
            Initializes the SeRO agent with policy networks, optimizers, and configuration parameters.

        cal_uncertainty(state, original=False):
            Calculates the uncertainty of the policy for the given state.

        reupdate_parameters(memory, batch_size, updates):
            Updates the agent's parameters using a batch of experience, incorporating auxiliary rewards.

        save_model(env_name, policy, suffix="", actor_path=None, critic_path=None):
            Saves the actor, critic, and variance mapping models to specified or default paths.

        load_model(args, actor_path, critic_path, mapping_path=None, retrain=False, original_actor_path=None, original_mapping_path=None):
            Loads the actor, critic, and variance mapping models from specified paths, with support for retraining and original policy handling.
    """
    
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize the SeRO agent.

        Args:
            num_inputs (int): Number of input features in the state space.
            action_space (gym.Space): Action space of the environment.
            args (Namespace): Configuration parameters for the SeRO algorithm.
                - aux_coef (float): Coefficient for the auxiliary reward.
                - consol_coef (float): Coefficient for the consistency loss.
                - lr_policy (float): Learning rate for the policy network.
                - use_aux_reward (bool): Whether to use auxiliary rewards in training.
        """
        super().__init__(num_inputs, action_space, args)
        self.policy = SeROPolicy(num_inputs, self.num_actions, self.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
        self.policy_original = SeROPolicy(num_inputs, self.num_actions, args.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
        self.aux_coeff = args.aux_coef
        self.consol_coef = args.consol_coef
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)

        if not args.use_aux_reward:
            self.aux_coeff = 0.0
        
    def cal_uncertainty(self, state, original=False):
        """
        Calculate the uncertainty of the policy for a given state.

        Args:
            state (np.ndarray): The current state of the environment.
            original (bool, optional): If True, use the original policy for uncertainty calculation. Defaults to False.

        Returns:
            np.ndarray: The uncertainty value for the given state.
        """
        state = torch.FloatTensor(state.copy()).to(self.device).unsqueeze(0)
        if original:
            deg_uncertainty = self.policy_original.uncertainty(state)
        else:
            deg_uncertainty = self.policy.uncertainty(state)
        return deg_uncertainty.clone().detach().cpu().numpy()[0]
    
    def reupdate_parameters(self, memory, batch_size, updates):
        """
        Re-update the agent's parameters using a batch of experience, incorporating auxiliary rewards.

        Args:
            memory (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples to draw from the replay buffer.
            updates (int): The current update step count.

        Returns:
            tuple:
                - critic_loss (float): Loss value for the critic network.
                - policy_loss (float): Loss value for the policy network.
                - alpha_loss (float): Loss value for the entropy coefficient.
                - upc_loss (float): Average uncertainty-based consistency loss.
        """
        state_batch, uncertainty_batch, action_batch, reward_batch, next_state_batch, next_uncertainty_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        uncertainty_batch = torch.FloatTensor(uncertainty_batch).to(self.device)
        next_uncertainty_batch = torch.FloatTensor(next_uncertainty_batch).to(self.device)

        # Auxiliary reward calculation
        aux_reward_batch = -torch.where(reward_batch == 0, next_uncertainty_batch, torch.zeros_like(next_uncertainty_batch))
        total_reward = reward_batch + self.aux_coeff * aux_reward_batch

        # Update critic
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                next_action, next_log_pi, _, _, _ = self.policy(next_state_batch)
                q1_next, q2_next = self.critic_target(next_state_batch, next_action)
                soft_value = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
                target_q = total_reward + mask_batch * self.gamma * soft_value

            q1, q2 = self.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Update policy
            pi, log_pi, _, _, _ = self.policy(state_batch)
            _, _, _, _, original_action = self.policy_original(state_batch)
            dist_entropy = -self.consol_coef * self.policy.evaluate(state_batch, original_action)
            upc_loss = (1 - uncertainty_batch) * dist_entropy

            q1_pi, q2_pi = self.critic(state_batch, pi)
            min_q = torch.min(q1_pi, q2_pi)
            policy_loss = ((self.alpha * log_pi) - min_q + upc_loss).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Update alpha
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.0).to(self.device)

            # Soft update targets
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

            return critic_loss.item(), policy_loss.item(), alpha_loss.item(), upc_loss.mean().item()
    
    def save_model(self, env_name, policy, suffix="", actor_path=None, critic_path=None):
        """
        Save the actor, critic, and variance mapping models to specified or default paths.

        Args:
            env_name (str): The name of the environment.
            policy (str): The name of the policy type.
            suffix (str, optional): A suffix to append to the filenames. Defaults to an empty string.
            actor_path (str, optional): Path to save the actor model. Defaults to None.
            critic_path (str, optional): Path to save the critic model. Defaults to None.
        """
        super().save_model(env_name, policy, suffix, actor_path, critic_path)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mapping_path = '{}/trained_models/{}/{}/mapping{}.pt'.format(parent_dir, env_name, policy, suffix)
        torch.save(self.policy.do_variance, mapping_path)
    
    def load_model(self, args, actor_path, critic_path, mapping_path=None, retrain=False, original_actor_path=None, original_mapping_path=None):
        """
        Load the actor, critic, and variance mapping models from specified paths, with support for retraining.

        Args:
            args (Namespace): Configuration parameters for the SeRO algorithm.
            actor_path (str): Path to the saved actor model.
            critic_path (str): Path to the saved critic model.
            mapping_path (str, optional): Path to the variance mapping file. Defaults to None.
            retrain (bool, optional): If True, reloads the actor model for further training. Defaults to False.
            original_actor_path (str, optional): Path to the original actor model for reference. Defaults to None.
            original_mapping_path (str, optional): Path to the original variance mapping file for reference. Defaults to None.
        """
        if retrain:
            print('Loading models from {}'.format(actor_path))
            if actor_path is not None:
                self.policy.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy_original.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy.train()
                self.policy_original.eval()

                if hasattr(self.policy, 'do_variance'):
                    self.policy.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                    self.policy_original.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                    self.policy.fix_uncertainty = True
                    self.policy_original.fix_uncertainty = True
                    self.policy.dropout.train()
                    self.policy_original.dropout.train()
                
        else:
            super().load_model(args, actor_path, critic_path)
                
            if original_actor_path is not None and os.path.exists(original_actor_path):
                self.policy_original.load_state_dict(torch.load(original_actor_path, map_location='cuda:{}'.format(args.cuda_device)))
            else:
                self.policy_original.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
            self.policy_original.eval()
            
            if hasattr(self.policy, 'do_variance'):
                self.policy.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                if original_mapping_path is not None:
                    self.policy_original.do_variance = torch.load(original_mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                else:
                    self.policy_original.do_variance = torch.load(mapping_path, map_location='cuda:{}'.format(args.cuda_device))
                self.policy.fix_uncertainty = True
                self.policy_original.fix_uncertainty = True
                self.policy.dropout.train()
                self.policy_original.dropout.train()
                
class LaMoUR(SAC):
    """
    An implementation of the Lamour algorithm that extends the Soft Actor-Critic (SAC) 
    with additional capabilities for generating environment-specific Python code and 
    auxiliary reward mechanisms.

    Methods:
        __init__(num_inputs, action_space, args):
            Initialize the LaMoUR class with environment configurations, coefficients, 
            and generated code handling.

        import_codes():
            Dynamically load or generate Python code for the specified environment.

        reupdate_parameters(memory, batch_size, updates):
            Update the critic, policy, and entropy parameters using a batch of experience, 
            incorporating auxiliary rewards.
    """
    
    def __init__(self, num_inputs, action_space, args):
        """
        Initializes the LaMoUR class, extending the SAC implementation with additional functionality.

        Args:
            num_inputs (int): Number of input features in the state space.
            action_space (gym.Space): The action space of the environment.
            args (Namespace): Additional configuration parameters.
                - aux_coef (float): Coefficient for auxiliary rewards.
                - consol_coef (float): Coefficient for uncertainty regularization.
                - env_name (str): Name of the environment.
                - use_aux_reward (bool): Whether to enable auxiliary rewards.
        """
        super().__init__(num_inputs, action_space, args)
        self.policy_original = StochasticPolicy(num_inputs, self.num_actions, self.hidden_size, args.observation_type, action_space, args.drop_p).to(self.device)
        self.aux_coeff = args.aux_coef if args.use_aux_reward else 0.0
        self.consol_coef = args.consol_coef
        self.env_name = args.env_name
        self.env_name_sym = extract_env_name_symbol(args.env_name)
        self.lamour = lamour()
        self.lamour_codes={}
        if args.retrain_lamour:
            self.import_codes()
        
    def import_codes(self):
        """
        Import dynamically generated Python code for the environment. 

        If the required code does not exist, it generates the necessary files and 
        loads the code for use in training.
        """
        base_path = Path(__file__).resolve().parent.parent / 'lamour'
        code_path = base_path / f'outputs/{self.env_name_sym}/generated_code.py'

        # Generate code if it doesn't exist
        if not code_path.is_file():
            ood_image_path = base_path / f'ood_images/{self.env_name_sym}_ood.png'
            environment_description_file = base_path / f'env_description/{self.env_name_sym}_env.txt'
            ood_description_output_file = base_path / f'outputs/{self.env_name_sym}/ood_description_output.txt'
            behavior_reasoning_output_file = base_path / f'outputs/{self.env_name_sym}/behavior_reasoning_output.txt'
            original_task = get_original_task(self.env_name)
            env_name_for_llm = get_env_name_for_llm(self.env_name)
            use_few_shot = False
            if 'PushChair' in self.env_name:
                use_few_shot = True

            self.lamour.generate_codes(
                ood_image_path=ood_image_path,
                original_task=original_task,
                environment_description_file=environment_description_file,
                ood_description_output_file=ood_description_output_file,
                behavior_reasoning_output_file=behavior_reasoning_output_file,
                code_generation_output_file=code_path,
                env_name=env_name_for_llm,
                use_few_shot=use_few_shot
            )

        # Load and execute the generated code
        with code_path.open("r") as file:
            codes = file.read()
            
        exec(codes, self.lamour_codes)
    
    def reupdate_parameters(self, memory, batch_size, updates):
        """
        Updates the critic, policy, and entropy parameters using a batch of experience.

        Incorporates auxiliary rewards and regularization to improve stability and performance.

        Args:
            memory (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples to draw from the replay buffer.
            updates (int): Current update step count.

        Returns:
            tuple:
                - critic_loss (float): Loss value for the critic network.
                - policy_loss (float): Loss value for the policy network.
                - alpha_loss (float): Loss value for the entropy coefficient.
                - lpc_loss (float): Average language model-based consistency loss.
        """
        state_batch, is_recovered_batch, action_batch, reward_batch, next_state_batch, aux_reward_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        is_recovered_batch = torch.FloatTensor(is_recovered_batch).to(self.device).unsqueeze(1)
        aux_reward_batch = torch.FloatTensor(aux_reward_batch).to(self.device).unsqueeze(1)

        # Auxiliary reward calculation
        aux_reward_batch = torch.where(reward_batch == 0, aux_reward_batch, torch.zeros_like(aux_reward_batch))
        total_reward = reward_batch + self.aux_coeff * aux_reward_batch

        # Update critic
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                next_action, next_log_pi, _, _, _ = self.policy(next_state_batch)
                q1_next, q2_next = self.critic_target(next_state_batch, next_action)
                soft_value = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
                target_q = total_reward + mask_batch * self.gamma * soft_value

            q1, q2 = self.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Update policy
            pi, log_pi, _, _, _ = self.policy(state_batch)
            _, _, _, _, original_action = self.policy_original(state_batch)
            dist_entropy = -self.consol_coef * self.policy.evaluate(state_batch, original_action)
            lpc_loss = (is_recovered_batch + 1e-20) * dist_entropy # to resolve NaN issue during back propagation

            q1_pi, q2_pi = self.critic(state_batch, pi)
            min_q = torch.min(q1_pi, q2_pi)
            policy_loss = ((self.alpha * log_pi) - min_q + lpc_loss).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Update alpha
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.0).to(self.device)

            # Soft update targets
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

            return critic_loss.item(), policy_loss.item(), alpha_loss.item(), lpc_loss.mean().item() 
        
    def save_model(self, env_name, policy, suffix="", actor_path=None, critic_path=None):
        """
        Saves the actor, critic, and optional variance mapping models to specified paths.

        Args:
            env_name (str): The name of the environment.
            policy (str): The policy type (e.g., "Stochastic").
            suffix (str, optional): Additional suffix for saved filenames. Defaults to an empty string.
            actor_path (str, optional): Custom path to save the actor model. Defaults to None.
            critic_path (str, optional): Custom path to save the critic model. Defaults to None.
        """
        super().save_model(env_name, policy, suffix, actor_path, critic_path)
        
    def load_model(self, args, actor_path, critic_path, mapping_path=None, retrain=False, original_actor_path=None, original_mapping_path=None):
        """
        Loads actor, critic, and optional variance mapping models from specified paths.

        Supports retraining and initialization of original models for policy comparison.

        Args:
            args (Namespace): Configuration parameters for model loading.
            actor_path (str): Path to the saved actor model.
            critic_path (str): Path to the saved critic model.
            mapping_path (str, optional): Path to the variance mapping file. Defaults to None.
            retrain (bool, optional): If True, loads the actor model for further training. Defaults to False.
            original_actor_path (str, optional): Path to the original actor model for comparison. Defaults to None.
            original_mapping_path (str, optional): Path to the original variance mapping file. Defaults to None.
        """
        if retrain:
            print('Loading models from {}'.format(actor_path))
            if actor_path is not None:
                self.policy.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy_original.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
                self.policy.train()
                self.policy_original.eval()

        else:
            super().load_model(args, actor_path, critic_path)
                
            if original_actor_path is not None and os.path.exists(original_actor_path):
                self.policy_original.load_state_dict(torch.load(original_actor_path, map_location='cuda:{}'.format(args.cuda_device)))
            else:
                self.policy_original.load_state_dict(torch.load(actor_path, map_location='cuda:{}'.format(args.cuda_device)))
            self.policy_original.eval()
