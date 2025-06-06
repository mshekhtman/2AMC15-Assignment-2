"""
DQN Agent for 10D continuous state space restaurant delivery robot.
This file contains the complete DQN algorithm implementation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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
    """DQN Agent for 10D continuous state space restaurant delivery."""
    
    def __init__(self, 
                 state_dim=10,  # Updated for 10D state space
                 action_dim=4, 
                 state_type='continuous_vector',
                 gamma=0.99, 
                 lr=1e-3, 
                 batch_size=64, 
                 buffer_size=50000,  # Reduced for simpler environment
                 min_replay_size=1000, 
                 target_update_freq=500,  # More frequent updates for faster learning
                 epsilon_start=1.0,
                 epsilon_min=0.01,  # Lower minimum for better exploitation
                 epsilon_decay=0.995):
        """Initialize DQN Agent for 10D state space.
        
        Args:
            state_dim: Dimension of state space (10 for simplified environment)
            action_dim: Number of actions (4 for movement)
            gamma: Discount factor for future rewards
            lr: Learning rate for neural network
            batch_size: Size of training batches
            buffer_size: Size of experience replay buffer
            min_replay_size: Minimum experiences before training starts
            target_update_freq: How often to update target network
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        super().__init__(state_dim, action_dim, state_type)
        
        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        
        # Experience replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")
        
        # Neural networks
        self.q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = DQNetwork(state_dim, action_dim).to(self.device)
        
        # Initialize target network with same weights as main network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Training tracking
        self.training_steps = 0
        self.episode_count = 0
        self.losses = []
        
        print(f"DQN Agent initialized with {state_dim}D state space")
    
    def take_training_action(self, state, training=True):
        """Take action using epsilon-greedy policy.
        
        Args:
            state: Current state (10D vector or compatible format)
            training: Whether in training mode (affects exploration)
            
        Returns:
            action: Integer action in [0, 3]
        """
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
        """Take action using current policy (for evaluation).
        
        Args:
            state: Current state (10D vector or compatible format)
            
        Returns:
            action: Integer action in [0, 3]
        """
        return self.take_training_action(state, training=False)
    
    def update(self, state, reward, action, next_state=None, done=False):
        """Update the agent with new experience.
        
        Args:
            state: Current state
            reward: Reward received
            action: Action taken
            next_state: Next state
            done: Whether episode terminated
        """
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
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            print(f"Target network updated at step {self.training_steps}")
        
        # Decay epsilon at episode end
        if done:
            self.episode_count += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Print progress occasionally
            if self.episode_count % 50 == 0:
                avg_loss = np.mean(self.losses[-50:]) if self.losses else 0
                print(f"Episode {self.episode_count}: epsilon={self.epsilon:.3f}, "
                      f"avg_loss={avg_loss:.4f}, buffer_size={len(self.replay_buffer)}")
    
    def _train_step(self):
        """Perform one training step on a batch of experiences."""
        # Sample random batch from replay buffer
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
            'replay_buffer_size': len(self.replay_buffer)
        }, filepath)
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
            
            print(f"DQN Agent loaded from {filepath}")
            print(f"Epsilon: {self.epsilon:.3f}, Episodes: {self.episode_count}, Steps: {self.training_steps}")
            
        except FileNotFoundError:
            print(f"No saved DQN agent found at {filepath}")
        except Exception as e:
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