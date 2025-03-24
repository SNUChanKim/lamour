import os, sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.random_process import OUProcess

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, doubleQ=True):
        super(QNetwork, self).__init__()
        self.doubleQ = doubleQ
        self.obs_type = observation_type

        if observation_type == 'vector':
            self.Q1 = self._build_vector_network(num_inputs, num_actions, hidden_dim)
            if self.doubleQ:
                self.Q2 = self._build_vector_network(num_inputs, num_actions, hidden_dim)
        elif observation_type == 'box':
            self.encoder, encoded_dim = self._build_box_encoder(num_inputs, hidden_dim)
            self.Q1 = self._build_box_network(encoded_dim, num_actions, hidden_dim)
            if self.doubleQ:
                self.Q2 = self._build_box_network(encoded_dim, num_actions, hidden_dim)
        else:
            raise NotImplementedError

    def _build_vector_network(self, num_inputs, num_actions, hidden_dim):
        return nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _build_box_encoder(self, num_inputs, hidden_dim):
        fake_in = torch.zeros(1, num_inputs[2], num_inputs[0], num_inputs[1])
        encoder = nn.Sequential(
            nn.Conv2d(num_inputs[2], 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )
        fake_out = encoder(fake_in)
        encoded_dim = fake_out.numel()
        return encoder, encoded_dim

    def _build_box_network(self, encoded_dim, num_actions, hidden_dim):
        return nn.Sequential(
            nn.Linear(encoded_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        if self.obs_type == 'vector':
            state_action_pair = torch.cat([state, action], dim=-1)
            if self.doubleQ:
                return self.Q1(state_action_pair), self.Q2(state_action_pair)
            return self.Q1(state_action_pair)

        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = h.view(h.size(0), -1)
            h_action_pair = torch.cat([h, action], dim=-1)
            if self.doubleQ:
                return self.Q1(h_action_pair), self.Q2(h_action_pair)
            return self.Q1(h_action_pair)

        else:
            raise NotImplementedError
        
class BasePolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, action_space=None, drop_p=0.1):
        super(BasePolicy, self).__init__()
        self.obs_type = observation_type
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=drop_p)

        if observation_type == 'vector':
            self.shared = self._build_vector_network(num_inputs, hidden_dim, drop_p)
        elif observation_type == 'box':
            self.encoder, encoded_dim = self._build_box_encoder(num_inputs, hidden_dim, drop_p)
            self.shared = self._build_shared_network(encoded_dim, hidden_dim, drop_p)
        else:
            raise NotImplementedError

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)
        
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            if len(action_space.high.shape) > 1:
                high = action_space.high[0,:]
                low = action_space.low[0,:]
            else:
                high = action_space.high
                low = action_space.low
            self.action_scale = torch.FloatTensor((high - low) / 2.)
            self.action_bias = torch.FloatTensor((high + low) / 2.)

    def _build_vector_network(self, num_inputs, hidden_dim, drop_p):
        return nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout
        )

    def _build_box_encoder(self, num_inputs, hidden_dim, drop_p):
        fake_in = torch.zeros(1, num_inputs[2], num_inputs[0], num_inputs[1])
        encoder = nn.Sequential(
            nn.Conv2d(num_inputs[2], 16, 4, stride=2, padding=1),
            nn.ReLU(),
            self.dropout,
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            self.dropout,
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            self.dropout,
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            self.dropout
        )
        fake_out = encoder(fake_in)
        encoded_dim = fake_out.numel()
        return encoder, encoded_dim

    def _build_shared_network(self, encoded_dim, hidden_dim, drop_p):
        return nn.Sequential(
            nn.Linear(encoded_dim, 2 * hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout
        )

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(BasePolicy, self).to(device)

class DeterministicPolicy(BasePolicy):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, action_space=None, drop_p=0.1):
        super(DeterministicPolicy, self).__init__(num_inputs, num_actions, hidden_dim, observation_type, action_space, drop_p)
        self.epsilon = 1.0
        self.epsilon_decay = 2e-5
        self.ou_process = OUProcess(theta=0.3, mu=0., sigma=0.2, dt=5e-2, size=num_actions)
        self.action_high = torch.FloatTensor(action_space.high)
        self.action_low = torch.FloatTensor(action_space.low)

    def forward(self, state):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = h.view(h.size(0), -1)
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        noise = torch.clamp(self.ou_process.sample(), -0.5, 0.5) * self.epsilon
        action = torch.tanh(mean) * self.action_scale + self.action_bias + noise
        action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action, torch.tensor(0.), mean, torch.zeros(action.size(0)), mean

    def evaluate(self, state, action_sample):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = h.view(h.size(0), -1)
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        log_prob = (mean - action_sample) ** 2
        return log_prob

    def decay_eps(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0)

    def to(self, device):
        self.action_high = self.action_high.to(device)
        self.action_low = self.action_low.to(device)
        self.ou_process = self.ou_process.to(device)
        return super(DeterministicPolicy, self).to(device)
    
class StochasticPolicy(BasePolicy):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, action_space=None, drop_p=0.1):
        super(StochasticPolicy, self).__init__(num_inputs, num_actions, hidden_dim, observation_type, action_space, drop_p)

    def forward(self, state):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = h.view(h.size(0), -1)
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        log_std = torch.clamp(self.log_std(shared), -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_sample = normal.rsample()
        action_normalize = torch.tanh(action_sample)
        action = action_normalize * self.action_scale + self.action_bias
        log_prob = normal.log_prob(action_sample)
        log_prob -= torch.log(self.action_scale * (1 - action_normalize.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, std, action_sample

    def evaluate(self, state, action_sample):
        if self.obs_type == 'vector':
            shared = self.shared(state)
        elif self.obs_type == 'box':
            state = state.permute(0, 3, 1, 2).contiguous()
            h = self.encoder(state)
            h = h.view(h.size(0), -1)
            shared = self.shared(h)
        else:
            raise NotImplementedError

        mean = self.mean(shared)
        log_std = torch.clamp(self.log_std(shared), -10, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_normalize = torch.tanh(action_sample)
        log_prob = normal.log_prob(action_sample)
        log_prob -= torch.log(self.action_scale * (1 - action_normalize.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob

class SeROPolicy(StochasticPolicy):
    def __init__(self, num_inputs, num_actions, hidden_dim, observation_type, action_space=None, drop_p=0.1):
        super(SeROPolicy, self).__init__(num_inputs, num_actions, hidden_dim, observation_type, action_space, drop_p)
        self.num_sample = 30
        self.fix_uncertainty = False
        self.do_variance = -1e10 * torch.ones([hidden_dim])

    def forward(self, state):
        if not self.fix_uncertainty:
            deg_uncertainty = self.uncertainty(state)
        return super().forward(state)
    
    def uncertainty(self, state):
        with torch.no_grad():
            do_sum = 0
            do_square_sum = 0

            for _ in range(self.num_sample):
                if self.obs_type == 'vector':
                    x = self.shared(state)
                elif self.obs_type == 'box':
                    state = state.permute(0, 3, 1, 2).contiguous()
                    h = self.encoder(state)
                    h = h.view(h.size(0), -1)
                    x = self.shared(h)
                else:
                    raise NotImplementedError
                
                do_sum += x
                do_square_sum += x ** 2

            variance = (do_square_sum - (do_sum ** 2) / self.num_sample) / (self.num_sample - 1)
            if not self.fix_uncertainty:
                self.do_variance = torch.max(self.do_variance, variance)
            variance = variance / (self.do_variance + 1e-10)

            weight = variance / (torch.sum(variance, dim=1, keepdim=True) + 1e-10)
            uncertainty = torch.sum(variance * weight, dim=1, keepdim=True)
            return torch.clamp(uncertainty, 0.0, 1.0)

    def to(self, device):
        self.do_variance = self.do_variance.to(device)
        return super(SeROPolicy, self).to(device)