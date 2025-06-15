"""
DQN Neural Network for realistic 8D continuous state space.
This file contains the neural network architecture optimized for realistic robot sensing.
"""
import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    """Deep Q-Network for restaurant delivery robot with realistic 8D state space."""
    
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        """Initialize the DQN network.
        
        Args:
            state_dim: Input state dimension (8 for realistic environment)
            action_dim: Output action dimension (4 for up/down/left/right)
            hidden_dim: Hidden layer size
        """
        super(DQNetwork, self).__init__()
        
        # Network architecture optimized for 8D realistic state space
        # Slightly smaller network since we have fewer input features
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Light dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),  # Bottleneck layer
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
               For 8D realistic state:
               [0-1]: Normalized position (x, y)
               [2-5]: Clear directions (front, left, right, back)
               [6-7]: Mission status (remaining targets, progress)
            
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.network(x)


class DuelingDQNetwork(nn.Module):
    """Dueling DQN architecture for better value estimation."""
    
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        """Initialize the Dueling DQN network.
        
        Args:
            state_dim: Input state dimension (8 for realistic environment)
            action_dim: Output action dimension (4 for movement)
            hidden_dim: Hidden layer size
        """
        super(DuelingDQNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the dueling network.
        
        Args:
            x: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage using dueling formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values