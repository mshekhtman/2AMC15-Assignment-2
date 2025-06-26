"""
DQN Agent for realistic 8D continuous state space restaurant delivery robot.
UPDATED: Now automatically uses optimal hyperparameters based on grid.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pathlib import Path

# Handle imports like other agent files
try:
    from agents.base_agent import BaseAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.append(root_path)
    from agents.base_agent import BaseAgent

# Import the network (adjust path as needed)
from agents.DQN_nn import DQNetwork


class DQNAgent(BaseAgent):
    """DQN Agent for realistic 8D continuous state space restaurant delivery."""
    
    # ADDED: Optimal hyperparameters found through optimization
    OPTIMAL_CONFIGS = {
        # Best performing configuration (experiment_id 14)
        'default': {
            'lr': 0.001,
            'batch_size': 128,
            'buffer_size': 50000,
            'target_update_freq': 500,
            'epsilon_decay': 0.985,
            'epsilon_min': 0.05,
            'gamma': 0.99
        },
        # Grid-specific overrides if needed
        'maze': {
            'lr': 0.0005,
            'epsilon_min': 0.01,
            'target_update_freq': 1000
        }
    }
    
    def __init__(self, 
                 state_dim=8,
                 action_dim=4, 
                 state_type='continuous_vector',
                 gamma=None,  # UPDATED: Allow None to use optimal
                 lr=None,     # UPDATED: Allow None to use optimal
                 batch_size=None,  # UPDATED: Allow None to use optimal
                 buffer_size=None,  # UPDATED: Allow None to use optimal
                 min_replay_size=1000, 
                 target_update_freq=None,  # UPDATED: Allow None to use optimal
                 epsilon_start=1.0,
                 epsilon_min=None,  # UPDATED: Allow None to use optimal
                 epsilon_decay=None,  # UPDATED: Allow None to use optimal
                 verbose=True,
                 grid_path=None):  # ADDED: Optional grid path for optimal config
        """Initialize DQN Agent with optimal hyperparameters.
        
        Args:
            grid_path: Optional path to grid file for automatic optimal config selection
            Other parameters: If None, will use optimal values from hyperparameter search
        """
        super().__init__(state_dim, action_dim, state_type)
        
        # ADDED: Get optimal configuration
        optimal_config = self._get_optimal_config(grid_path)
        
        # UPDATED: Use optimal hyperparameters if not specified
        self.gamma = gamma if gamma is not None else optimal_config['gamma']
        self.lr = lr if lr is not None else optimal_config['lr']
        self.batch_size = batch_size if batch_size is not None else optimal_config['batch_size']
        self.buffer_size = buffer_size if buffer_size is not None else optimal_config['buffer_size']
        self.target_update_freq = target_update_freq if target_update_freq is not None else optimal_config['target_update_freq']
        self.epsilon_min = epsilon_min if epsilon_min is not None else optimal_config['epsilon_min']
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else optimal_config['epsilon_decay']
        
        self.min_replay_size = min_replay_size
        self.verbose = verbose
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"DQN Agent using device: {self.device}")
            if any(param is None for param in [gamma, lr, batch_size, buffer_size, target_update_freq, epsilon_min, epsilon_decay]):
                print(f"Using optimal hyperparameters: lr={self.lr}, batch_size={self.batch_size}")
        
        # Neural networks
        self.q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = DQNetwork(state_dim, action_dim).to(self.device)
        
        # Initialize target network with same weights as main network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer with optimal learning rate
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Exploration parameters
        self.epsilon = epsilon_start
        
        # Training tracking
        self.training_steps = 0
        self.episode_count = 0
        self.losses = []
        
        if self.verbose:
            print(f"DQN Agent initialized with {state_dim}D realistic state space")
            print(f"State features: position(2) + clearance(4) + mission(2) = {state_dim}D")
    
    def _get_optimal_config(self, grid_path):
        """ADDED: Get optimal configuration based on grid path."""
        config = self.OPTIMAL_CONFIGS['default'].copy()
        
        if grid_path is not None:
            grid_name = str(grid_path).lower()
            # Apply grid-specific overrides
            if 'maze' in grid_name or 'challenge' in grid_name:
                config.update(self.OPTIMAL_CONFIGS['maze'])
        
        return config
    
    def take_training_action(self, state, training=True):
        """Take action using epsilon-greedy policy."""
        state = self.preprocess_state(state)
        
        # Epsilon-greedy action selection
        if training and (np.random.rand() < self.epsilon):
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: choose best action according to Q-network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            
            return action
        
    def take_action(self, state):
        """Take action using current policy (for evaluation)."""
        return self.take_training_action(state, training=False)
    
    def update(self, state, reward, action, next_state=None, done=False):
        """Update the agent with new experience."""
        # Preprocess states
        if next_state is not None:
            state = self.preprocess_state(state)
            next_state = self.preprocess_state(next_state)
            
            # Store experience in replay buffer
            self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Don't train until we have enough experiences
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        # Train the network
        self._train_step()
        
        # Update target network periodically with optimal frequency
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            if self.verbose and self.training_steps % (self.target_update_freq * 5) == 0:
                print(f"Target network updated at step {self.training_steps}")
        
        # Decay epsilon at episode end with optimal schedule
        if done:
            self.episode_count += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Print progress occasionally and only if verbose
            if self.verbose and self.episode_count % 25 == 0:
                avg_loss = np.mean(self.losses[-50:]) if self.losses else 0
                print(f"Episode {self.episode_count}: epsilon={self.epsilon:.3f}, "
                      f"avg_loss={avg_loss:.4f}, buffer_size={len(self.replay_buffer)}")
    
    def _train_step(self):
        """Perform one training step on a batch of experiences with optimal batch size."""
        # Sample optimal batch size from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q_values = self.q_net(states).gather(1, actions)
        
        # Target Q-values: r + γ * max_a Q_target(s', a)
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track loss for monitoring
        self.losses.append(loss.item())
    
    def reset_episode(self):
        """Reset episode-specific state."""
        pass
    
    def save_agent(self, filepath: str):
        """Save DQN agent state."""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'replay_buffer_size': len(self.replay_buffer),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        if self.verbose:
            print(f"DQN Agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load DQN agent state."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            self.training_steps = checkpoint.get('training_steps', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            
            if self.verbose:
                print(f"DQN Agent loaded from {filepath}")
                print(f"Epsilon: {self.epsilon:.3f}, Episodes: {self.episode_count}, Steps: {self.training_steps}")
            
        except FileNotFoundError:
            if self.verbose:
                print(f"No saved DQN agent found at {filepath}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading DQN agent: {e}")
    
    def get_training_stats(self) -> dict:
        """Get training statistics for monitoring."""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'buffer_size': len(self.replay_buffer),
            'avg_loss_last_100': np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0,
            'total_losses_recorded': len(self.losses)
        }
    
    def analyze_state(self, state):
        """Analyze and print state information for debugging."""
        state = self.preprocess_state(state)
        state_info = self.get_state_info(state)
        
        if self.verbose:
            print(f"=== State Analysis ===")
            print(f"Position: {state_info['position']}")
            print(f"Grid Position: {state_info['position_grid']}")
            print(f"Clear Directions: {state_info['clear_directions']}")
            print(f"Remaining Targets: {state_info['remaining_targets']:.2f}")
            print(f"Mission Progress: {state_info['progress']:.2f}")
            
            # Get Q-values for current state
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor).squeeze().cpu().numpy()
            
            action_names = ["down", "up", "left", "right"]
            print(f"Q-values: {dict(zip(action_names, q_values))}")
            print(f"Best action: {action_names[np.argmax(q_values)]}")
            print("=" * 20)