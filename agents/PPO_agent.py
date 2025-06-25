"""
OPTIMIZED PPO Agent for Assignment 2 - Targeting DQN-level performance.
Key improvements: better hyperparameters, reward normalization, adaptive learning.
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
    """Optimized PPO Agent targeting DQN-level performance."""
    
    def __init__(
            self,
            state_dim: int = 8,
            action_dim: int = 4,
            state_type: str = 'continuous_vector',
            # OPTIMIZED HYPERPARAMETERS FOR RESTAURANT DELIVERY
            lr: float = 2e-3,           # Even higher learning rate
            gamma: float = 0.99,        
            gae_lambda: float = 0.95,   
            clip_eps: float = 0.25,     # Moderate clipping
            entropy_coef: float = 0.03, # Much higher exploration
            value_coef: float = 1.0,    # Higher value learning
            max_grad_norm: float = 1.0, # Higher gradient norm
            rollout_steps: int = 64,    # Shorter rollouts for faster updates
            ppo_epochs: int = 8,        # More training epochs
            batch_size: int = 32,       # Smaller batches for more updates
            hidden_dim: int = 128,      
            verbose: bool = False       # Keep quiet
    ):
        super().__init__(state_dim, action_dim, state_type)

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

        # Policy network and optimizer
        self.policy = PPOPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        # More aggressive learning rate scheduling
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=25, gamma=0.85  # Faster decay
        )

        # On-policy experience buffer
        self.buffer = []
        
        # REWARD NORMALIZATION for better learning
        self.reward_normalizer = RunningMeanStd()
        
        # Training statistics
        self.episode_count = 0
        self.update_count = 0
        self.total_steps = 0
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_values = deque(maxlen=100)
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.best_avg_reward = float('-inf')

    @torch.no_grad()
    def take_action(self, state):
        """Select action with the optimized policy."""
        state = self.preprocess_state(state)
        
        state_tensor = torch.from_numpy(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        action, logp, value = self.policy.act(state_tensor)

        self.buffer.append({
            "state": state.copy(),
            "action": action,
            "logp": logp.cpu().item(),
            "value": value.cpu().item(),
            "reward": None,
            "done": False
        })
        
        self.total_steps += 1
        return action

    def take_training_action(self, state, training=True):
        """Training action (same as take_action for PPO)."""
        return self.take_action(state)

    def update(self, state, reward, action, next_state=None, done=False):
        """Update with reward normalization and adaptive learning."""
        if not self.buffer:
            return
            
        # REWARD NORMALIZATION - key for PPO performance
        normalized_reward = self.reward_normalizer.normalize(reward)
        
        self.buffer[-1]["reward"] = normalized_reward
        self.buffer[-1]["done"] = done

        if done:
            self.episode_count += 1
            
            # Track episode performance
            episode_reward = sum(exp["reward"] for exp in self.buffer if exp["reward"] is not None)
            self.episode_rewards.append(episode_reward)
            
            # Adaptive hyperparameters based on performance
            self._adapt_hyperparameters()
            
        if done or len(self.buffer) >= self.rollout_steps:
            complete_experiences = [exp for exp in self.buffer if exp["reward"] is not None]
            if len(complete_experiences) >= 4:  # Lower threshold for more frequent updates
                self._learn()
            self.buffer.clear()

    def _adapt_hyperparameters(self):
        """Adapt hyperparameters based on performance."""
        if len(self.episode_rewards) >= 20:
            current_avg = np.mean(list(self.episode_rewards)[-20:])
            
            # Increase exploration if performance is stagnating
            if current_avg <= self.best_avg_reward:
                self.entropy_coef = min(0.1, self.entropy_coef * 1.05)
            else:
                self.best_avg_reward = current_avg
                self.entropy_coef = max(0.01, self.entropy_coef * 0.98)

    def _learn(self):
        """Enhanced PPO learning with normalization."""
        if len(self.buffer) < 2:
            return
        
        complete_buffer = [exp for exp in self.buffer if exp["reward"] is not None]
        if len(complete_buffer) < 2:
            return
            
        # Convert to tensors
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

        # Bootstrap value
        with torch.no_grad():
            if complete_buffer[-1]["done"]:
                next_v = 0.0
            else:
                last_state = states[-1].unsqueeze(0)
                _, next_v = self.policy.forward(last_state)
                next_v = next_v.item()

        # IMPROVED GAE computation
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

        # ROBUST advantage normalization
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO optimization with more epochs
        effective_bs = min(self.batch_size, states.size(0))
        n_samples = states.size(0)
        
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_values = []
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(n_samples, device=self.device)
            
            for start_idx in range(0, n_samples, effective_bs):
                end_idx = min(start_idx + effective_bs, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_logp = old_logp[batch_indices]

                new_logp, entropy, new_values = self.policy.evaluate(batch_states, batch_actions)
                
                # PPO loss with early stopping if ratio too large
                ratio = torch.exp(new_logp - batch_old_logp)
                
                # ADAPTIVE CLIPPING based on ratio distribution
                if ratio.max() > 5.0 or ratio.min() < 0.2:
                    # Skip this batch if ratios are too extreme
                    continue
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # VALUE CLIPPING for stability
                value_pred_clipped = batch_returns + torch.clamp(
                    new_values - batch_returns, -self.clip_eps, self.clip_eps
                )
                value_loss1 = (new_values - batch_returns).pow(2)
                value_loss2 = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                entropy_loss = -entropy.mean()
                
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropy_values.append(entropy.mean().item())

        self.lr_scheduler.step()
        
        self.update_count += 1
        if epoch_policy_losses:  # Only update if we had valid batches
            self.policy_losses.extend(epoch_policy_losses)
            self.value_losses.extend(epoch_value_losses)
            self.entropy_values.extend(epoch_entropy_values)

        # Progress reporting
        if self.verbose and self.update_count % 20 == 0:
            avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            recent_reward = np.mean(list(self.episode_rewards)[-10:]) if self.episode_rewards else 0
            
            print(f"PPO Update {self.update_count}: "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Recent Reward: {recent_reward:.1f}, "
                  f"Entropy Coef: {self.entropy_coef:.3f}, "
                  f"LR: {current_lr:.2e}")

    def reset_episode(self):
        """Reset episode state."""
        if self.buffer:
            self.buffer.clear()

    def get_training_stats(self) -> dict:
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'avg_policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses) if self.value_losses else 0,
            'avg_entropy': np.mean(self.entropy_values) if self.entropy_values else 0,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'entropy_coef': self.entropy_coef,
            'recent_avg_reward': np.mean(list(self.episode_rewards)[-20:]) if len(self.episode_rewards) >= 20 else 0,
            'buffer_size': len(self.buffer)
        }

    def save_agent(self, filepath: str):
        """Save PPO agent."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'reward_normalizer': self.reward_normalizer.get_state(),
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'best_avg_reward': self.best_avg_reward,
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
        """Load PPO agent."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            if 'reward_normalizer' in checkpoint:
                self.reward_normalizer.set_state(checkpoint['reward_normalizer'])
            
            self.episode_count = checkpoint.get('episode_count', 0)
            self.update_count = checkpoint.get('update_count', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
            
            if self.verbose:
                print(f"PPO Agent loaded from {filepath}")
                
        except FileNotFoundError:
            if self.verbose:
                print(f"No saved PPO agent found at {filepath}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading PPO agent: {e}")


class RunningMeanStd:
    """Running mean and standard deviation for reward normalization."""
    
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x):
        """Update running statistics with new value."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count
    
    def normalize(self, x):
        """Normalize value and update statistics."""
        self.update(x)
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
    
    def get_state(self):
        """Get current state for saving."""
        return {'mean': self.mean, 'var': self.var, 'count': self.count}
    
    def set_state(self, state):
        """Set state from loaded data."""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']