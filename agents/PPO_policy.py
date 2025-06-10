import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOPolicy(nn.Module):
    """
    Shared actor-critic network for Proximal Policy Optimization (PPO).
    """
    def __init__(self, state_dim=10, action_dim=4, hidden_dim=128):
        """Initialize the PPO shared actor-critic network with the given state and action dimensions.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Hidden layer size.
        """
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)  # logits (no softmax here)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)

    def forward(self, states):
        """Compute logits and value for a batch of states.
        Args:
            states (torch.Tensor): Tensor of shape [B, state_dim].

        Returns:
            torch.Tensor: Logits of shape [B, action_dim].
            torch.Tensor: Values of shape [B].
        """
        h = self.shared(states)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, state_tensor):
        """Sample an action for a single state (no grad).
        Args:
            state_tensor (torch.Tensor): Tensor of shape [1, state_dim].

        Returns:
            int: Sampled action (for env.step).
            torch.Tensor: Log-probability of the sampled action.
            torch.Tensor: Predicted state value.
        """
        logits, value = self.forward(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate log-prob, entropy and value on a minibatch.
        Args:
            states (torch.Tensor): Tensor of shape [B, state_dim].
            actions (torch.Tensor): Tensor of shape [B].

        Returns:
            torch.Tensor: New log-probs of shape [B].
            torch.Tensor: Entropy of shape [B].
            torch.Tensor: Predicted values of shape [B].
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values