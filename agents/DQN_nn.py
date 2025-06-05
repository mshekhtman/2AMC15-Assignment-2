"""
DQN Neural Network for 10D continuous state space.
This file contains the neural network architecture.
"""
import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    """Deep Q-Network for restaurant delivery robot with 10D state space."""
    
    def __init__(self, state_dim=10, action_dim=4, hidden_dim=128):
        """Initialize the DQN network.
        
        Args:
            state_dim: Input state dimension (10 for our simplified environment)
            action_dim: Output action dimension (4 for up/down/left/right)
            hidden_dim: Hidden layer size
        """
        super(DQNetwork, self).__init__()
        
        # Network architecture optimized for 10D state space
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),  # Slightly smaller final hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
        # Initialize weights for better training stability
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.network(x)