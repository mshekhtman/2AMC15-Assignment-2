"""
PPO Policy Network for realistic 8D continuous state space.
Optimized for restaurant delivery robot with realistic sensor inputs.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOPolicy(nn.Module):
    """
    Shared actor-critic network for Proximal Policy Optimization (PPO).
    Optimized for 8D realistic state space: position(2) + clearance(4) + mission(2).
    """
    
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        """Initialize the PPO shared actor-critic network.
        
        Args:
            state_dim (int): Dimension of the state space (8 for realistic environment)
            action_dim (int): Dimension of the action space (4 for movement)
            hidden_dim (int): Hidden layer size
        """
        super().__init__()
        
        # Validate input dimensions
        if state_dim != 8:
            print(f"Warning: Expected state_dim=8 for realistic environment, got {state_dim}")
        if action_dim != 4:
            print(f"Warning: Expected action_dim=4 for movement, got {action_dim}")
        
        # Shared feature extraction layers
        # Slightly deeper network for better representation learning
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),  # ReLU generally works better than Tanh for RL
            nn.LayerNorm(hidden_dim),  # Layer normalization for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Light dropout for regularization
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim)
        )
        
        # Critic head (value network)  
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
        
        # Improved weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Special initialization for actor output (smaller initial policy variance)
        nn.init.xavier_uniform_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
    
    def forward(self, states):
        """Compute logits and value for a batch of states.
        
        Args:
            states (torch.Tensor): Tensor of shape [B, state_dim] where state_dim=8
                                  [0-1]: Normalized position (x, y)
                                  [2-5]: Clear directions (front, left, right, back)
                                  [6-7]: Mission status (remaining targets, progress)
        
        Returns:
            torch.Tensor: Action logits of shape [B, action_dim]
            torch.Tensor: State values of shape [B]
        """
        # Validate input shape
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        assert states.shape[-1] == 8, f"Expected 8D state, got {states.shape[-1]}D"
        
        # Shared feature extraction
        shared_features = self.shared(states)
        
        # Actor and critic outputs
        action_logits = self.actor(shared_features)
        state_values = self.critic(shared_features).squeeze(-1)
        
        return action_logits, state_values
    
    @torch.no_grad()
    def act(self, state_tensor):
        """Sample an action for a single state (no gradient computation).
        
        Args:
            state_tensor (torch.Tensor): Tensor of shape [1, 8] or [8] for single state
        
        Returns:
            int: Sampled action (0=down, 1=up, 2=left, 3=right)
            torch.Tensor: Log-probability of the sampled action
            torch.Tensor: Predicted state value
        """
        # Ensure proper shape
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        logits, value = self.forward(state_tensor)
        
        # Create categorical distribution from logits
        dist = Categorical(logits=logits)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate log-probabilities, entropy and values on a minibatch.
        
        Args:
            states (torch.Tensor): Tensor of shape [B, 8]
            actions (torch.Tensor): Tensor of shape [B] with action indices
        
        Returns:
            torch.Tensor: New log-probabilities of shape [B]
            torch.Tensor: Policy entropy of shape [B]
            torch.Tensor: Predicted state values of shape [B]
        """
        logits, values = self.forward(states)
        
        # Create categorical distribution
        dist = Categorical(logits=logits)
        
        # Compute log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, values
    
    def get_action_probabilities(self, state_tensor):
        """Get action probabilities for analysis/debugging.
        
        Args:
            state_tensor (torch.Tensor): Tensor of shape [1, 8] or [8]
            
        Returns:
            torch.Tensor: Action probabilities of shape [4]
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        logits, _ = self.forward(state_tensor)
        return torch.softmax(logits, dim=-1).squeeze(0)