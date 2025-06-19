"""
PPO Agent for realistic 8D continuous state space restaurant delivery robot.
Implements Proximal Policy Optimization with improvements for Assignment 2.
"""
import numpy as np
import torch
import torch.optim as optim
from collections import deque

try:
    from agents.base_agent import BaseAgent
    from agents.PPO_policy import PPOPolicy
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
    from agents.PPO_policy import PPOPolicy


class PPOAgent(BaseAgent):
    """
    Clipped-surrogate Proximal Policy Optimization (PPO) agent.
    Optimized for 8D realistic state space: position(2) + clearance(4) + mission(2).
    """

    def __init__(
            self,
            state_dim: int = 8,  # Updated for realistic 8D state space
            action_dim: int = 4,
            state_type: str = 'continuous_vector',
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            rollout_steps: int = 256,
            ppo_epochs: int = 4,
            batch_size: int = 64,
            hidden_dim: int = 128,
            verbose: bool = True
    ):
        """Initialize the PPO agent for realistic restaurant delivery.
        
        Args:
            state_dim (int): Dimension of the state space (8 for realistic environment)
            action_dim (int): Dimension of the action space (4 for movement)
            state_type (str): Type of the state space
            lr (float): Learning rate for the optimizer
            gamma (float): Discount factor for future rewards
            gae_lambda (float): GAE lambda parameter for advantage estimation
            clip_eps (float): PPO clip range for policy updates
            entropy_coef (float): Coefficient for entropy regularization
            value_coef (float): Critic loss weight
            max_grad_norm (float): Gradient clipping norm
            rollout_steps (int): Steps per on-policy rollout
            ppo_epochs (int): SGD epochs per update
            batch_size (int): Minibatch size for training
            hidden_dim (int): Hidden layer size for policy network
            verbose (bool): Whether to print training progress
        """
        super().__init__(state_dim, action_dim, state_type)

        # Validate state dimension for realistic environment
        if state_dim != 8:
            print(f"Warning: Expected state_dim=8 for realistic environment, got {state_dim}")

        # Hyperparameters
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"PPO Agent using device: {self.device}")
            print(f"PPO Agent initialized for 8D realistic state space")
            print(f"State features: position(2) + clearance(4) + mission(2) = 8D")

        # Policy network and optimizer
        self.policy = PPOPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        # Learning rate scheduler for better convergence
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.95
        )

        # On-policy experience buffer (memory)
        self.buffer = []
        
        # Training statistics
        self.episode_count = 0
        self.update_count = 0
        self.total_steps = 0
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_values = deque(maxlen=100)

    @torch.no_grad()
    def take_action(self, state):
        """Select an action based on the current state using the policy network.
        
        Args:
            state (np.array): Current 8D state of the environment
                             [0-1]: Normalized position (x, y)
                             [2-5]: Clear directions (front, left, right, back)
                             [6-7]: Mission status (remaining targets, progress)
        
        Returns:
            int: Selected action (0=down, 1=up, 2=left, 3=right)
        """
        # Preprocess state to ensure consistent format
        state = self.preprocess_state(state)
        
        # Convert to tensor
        state_tensor = torch.from_numpy(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Get action from policy
        action, logp, value = self.policy.act(state_tensor)

        # Store experience in buffer (without reward and done for now)
        self.buffer.append({
            "state": state.copy(),
            "action": action,
            "logp": logp.cpu().item(),
            "value": value.cpu().item(),
            "reward": None,  # Will be set in update()
            "done": False    # Will be set in update()
        })
        
        self.total_steps += 1
        return action

    def take_training_action(self, state, training=True):
        """Take action during training (alias for take_action for consistency)."""
        return self.take_action(state)

    def update(self, state, reward, action, next_state=None, done=False):
        """Update the agent's memory with the latest experience.
        
        Args:
            state (np.array): Current state (unused, already stored in take_action)
            reward (float): Reward received from the environment
            action (int): Action taken (unused, already stored)
            next_state (np.array): Next state (unused for PPO)
            done (bool): Whether the episode has ended
        """
        if not self.buffer:
            return  # No experience to update
            
        # Update the last experience with reward and done flag
        self.buffer[-1]["reward"] = reward
        self.buffer[-1]["done"] = done

        # Trigger learning when rollout is complete or episode ends
        if done:
            self.episode_count += 1
            
        if done or len(self.buffer) >= self.rollout_steps:
            # Only learn if we have complete experiences (with rewards)
            complete_experiences = [exp for exp in self.buffer if exp["reward"] is not None]
            if len(complete_experiences) >= 8:  # Minimum steps for meaningful updates
                self._learn()
            self.buffer.clear()  # Reset memory

    def _learn(self):
        """Compute GAE advantages and run the PPO clipped-surrogate update."""
        if len(self.buffer) < 2:
            return  # Not enough data for learning
        
        # Filter out incomplete experiences and check if all experiences have rewards
        complete_buffer = [exp for exp in self.buffer if exp["reward"] is not None]
        
        if len(complete_buffer) < 2:
            if self.verbose:
                print(f"Warning: Only {len(complete_buffer)} complete experiences, skipping learning")
            return
            
        # Convert buffer to tensors using only complete experiences
        states = torch.tensor(
            np.stack([b["state"] for b in complete_buffer]), 
            dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [b["action"] for b in complete_buffer], 
            dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            [b["reward"] for b in complete_buffer], 
            dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [b["done"] for b in complete_buffer], 
            dtype=torch.float32, device=self.device
        )
        old_logp = torch.tensor(
            [b["logp"] for b in complete_buffer], 
            dtype=torch.float32, device=self.device
        )
        values = torch.tensor(
            [b["value"] for b in complete_buffer], 
            dtype=torch.float32, device=self.device
        )

        # Bootstrap the last state value if rollout ended mid-episode
        with torch.no_grad():
            if complete_buffer[-1]["done"]:
                next_v = 0.0  # Episode finished
            else:
                last_state = states[-1].unsqueeze(0)
                _, next_v = self.policy.forward(last_state)
                next_v = next_v.item()

        # Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * mask - values[t]
            gae = delta + self.gamma * self.lmbda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_v = values[t]

        # Normalize advantages for stability
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        
        if adv_std < 1e-8:
            advantages = advantages - adv_mean
        else:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # PPO clipped-objective optimization
        effective_bs = min(self.batch_size, states.size(0))
        n_samples = states.size(0)
        
        # Store losses for monitoring
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_values = []
        
        for epoch in range(self.ppo_epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(n_samples, device=self.device)
            
            for start_idx in range(0, n_samples, effective_bs):
                end_idx = min(start_idx + effective_bs, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_logp = old_logp[batch_indices]

                # Forward pass
                new_logp, entropy, new_values = self.policy.evaluate(batch_states, batch_actions)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_logp - batch_old_logp)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = (batch_returns - new_values).pow(2).mean()
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropy_values.append(entropy.mean().item())

        # Update learning rate
        self.lr_scheduler.step()
        
        # Store training statistics
        self.update_count += 1
        self.policy_losses.extend(epoch_policy_losses)
        self.value_losses.extend(epoch_value_losses)
        self.entropy_values.extend(epoch_entropy_values)

        # Print training progress
        if self.verbose and self.update_count % 10 == 0:
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)
            avg_entropy = np.mean(epoch_entropy_values)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"PPO Update {self.update_count}: "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, "
                  f"Entropy: {avg_entropy:.4f}, "
                  f"LR: {current_lr:.6f}")

    def reset_episode(self):
        """Reset episode-specific state."""
        # Clear any remaining buffer if episode ended abruptly
        if self.buffer:
            self.buffer.clear()

    def save_agent(self, filepath: str):
        """Save PPO agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'hyperparameters': {
                'gamma': self.gamma,
                'lmbda': self.lmbda,
                'clip_eps': self.clip_eps,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'rollout_steps': self.rollout_steps,
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size
            }
        }, filepath)
        
        if self.verbose:
            print(f"PPO Agent saved to {filepath}")

    def load_agent(self, filepath: str):
        """Load PPO agent state."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self.episode_count = checkpoint.get('episode_count', 0)
            self.update_count = checkpoint.get('update_count', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            
            if self.verbose:
                print(f"PPO Agent loaded from {filepath}")
                print(f"Episodes: {self.episode_count}, Updates: {self.update_count}, Steps: {self.total_steps}")
                
        except FileNotFoundError:
            if self.verbose:
                print(f"No saved PPO agent found at {filepath}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading PPO agent: {e}")

    def get_training_stats(self) -> dict:
        """Get training statistics for monitoring."""
        return {
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'avg_policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses) if self.value_losses else 0,
            'avg_entropy': np.mean(self.entropy_values) if self.entropy_values else 0,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'buffer_size': len(self.buffer)
        }

    def analyze_state(self, state):
        """Analyze and print state information for debugging."""
        state = self.preprocess_state(state)
        state_info = self.get_state_info(state)
        
        if self.verbose:
            print(f"=== PPO State Analysis ===")
            print(f"Position: {state_info['position']}")
            print(f"Grid Position: {state_info['position_grid']}")
            print(f"Clear Directions: {state_info['clear_directions']}")
            print(f"Remaining Targets: {state_info['remaining_targets']:.2f}")
            print(f"Mission Progress: {state_info['progress']:.2f}")
            
            # Get action probabilities
            state_tensor = torch.from_numpy(state).to(self.device)
            action_probs = self.policy.get_action_probabilities(state_tensor).cpu().numpy()
            
            action_names = ["down", "up", "left", "right"]
            print(f"Action Probabilities: {dict(zip(action_names, action_probs))}")
            print(f"Preferred action: {action_names[np.argmax(action_probs)]}")
            print("=" * 25)