import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOPolicy(nn.Module):
    """
    Proximal Policy Optimization (PPO) policy network.
    """
    def _init_(self, state_dim, action_dim):
        """
        Initialize the PPO policy network with the given state and action dimensions.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        """
            
        super()._init_()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        """
        Forward pass through the policy network.
        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            probs (torch.Tensor): Action probabilities from the actor network.
            value (torch.Tensor): State value from the critic network.
        """
        value = self.critic(state)
        probs = self.actor(state)
        return probs,value