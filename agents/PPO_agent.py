"""
FIXED PPO Agent for Assignment 2 with critical performance issues resolved.
This implementation fixes:
1. Reward normalization issues
2. Configuration application problems
3. Episode tracking errors
4. Buffer management issues
5. Learning skip problems
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
    """Fixed PPO Agent with resolved performance issues."""
    
    # CORRECTED: Updated optimal hyperparameters to match optimization results
    OPTIMAL_CONFIGS = {
        # Open space grids - CORRECTED from optimization results
        'open_space': {
            'lr': 0.0005,               # Conservative LR from optimization
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.25,
            'entropy_coef': 0.03,       # Moderate exploration
            'value_coef': 1.0,
            'rollout_steps': 128,       # Long rollouts for planning
            'ppo_epochs': 6,
            'batch_size': 32,
            'hidden_dim': 128
        },
        # Simple restaurant - CORRECTED from optimization results
        'simple_restaurant': {
            'lr': 0.001,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.03,
            'value_coef': 0.5,
            'rollout_steps': 32,        # Shorter rollouts for quick decisions
            'ppo_epochs': 8,
            'batch_size': 16,
            'hidden_dim': 128
        },
        # Corridor test - CORRECTED from optimization results
        'corridor_test': {
            'lr': 0.001,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.25,
            'entropy_coef': 0.01,       # Low exploration for corridor navigation
            'value_coef': 1.0,
            'rollout_steps': 64,
            'ppo_epochs': 4,
            'batch_size': 32,
            'hidden_dim': 128
        },
        # A1 grid - CORRECTED from optimization results
        'A1_grid': {
            'lr': 0.001,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'rollout_steps': 64,
            'ppo_epochs': 4,
            'batch_size': 32,
            'hidden_dim': 128
        },
        # Assignment 2 main grid
        'assignment2_main': {
            'lr': 0.001,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.03,
            'value_coef': 0.5,
            'rollout_steps': 64,
            'ppo_epochs': 4,
            'batch_size': 32,
            'hidden_dim': 128
        },
        # Maze challenge - CORRECTED from optimization results
        'maze_challenge': {
            'lr': 0.001,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.01,       # Low exploration for maze navigation
            'value_coef': 0.5,
            'rollout_steps': 32,
            'ppo_epochs': 8,
            'batch_size': 32,
            'hidden_dim': 128
        },
        # Default configuration for unknown grids
        'default': {
            'lr': 0.001,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.03,
            'value_coef': 0.5,
            'rollout_steps': 64,
            'ppo_epochs': 4,
            'batch_size': 32,
            'hidden_dim': 128
        }
    }
    
    def __init__(
            self,
            state_dim: int = 8,
            action_dim: int = 4,
            state_type: str = 'continuous_vector',
            # UPDATED: Allow None to use optimal hyperparameters
            lr: float = None,
            gamma: float = None,
            gae_lambda: float = None,
            clip_eps: float = None,
            entropy_coef: float = None,
            value_coef: float = None,
            max_grad_norm: float = 1.0,
            rollout_steps: int = None,
            ppo_epochs: int = None,
            batch_size: int = None,
            hidden_dim: int = None,
            verbose: bool = False,
            grid_path=None  # ADDED: Optional grid path for automatic optimal config selection
    ):
        """Initialize PPO Agent with optimal hyperparameters.
        
        Args:
            grid_path: Optional path to grid file for automatic optimal config selection
            Other parameters: If None, will use optimal values from hyperparameter search
        """
        super().__init__(state_dim, action_dim, state_type)

        # FIXED: Set verbose first before calling _get_optimal_config
        self.verbose = verbose

        # ADDED: Get optimal configuration based on grid
        optimal_config = self._get_optimal_config(grid_path)
        
        # UPDATED: Use optimal hyperparameters if not specified
        self.gamma = gamma if gamma is not None else optimal_config['gamma']
        self.lmbda = gae_lambda if gae_lambda is not None else optimal_config['gae_lambda']
        self.clip_eps = clip_eps if clip_eps is not None else optimal_config['clip_eps']
        self.entropy_coef = entropy_coef if entropy_coef is not None else optimal_config['entropy_coef']
        self.value_coef = value_coef if value_coef is not None else optimal_config['value_coef']
        self.rollout_steps = rollout_steps if rollout_steps is not None else optimal_config['rollout_steps']
        self.ppo_epochs = ppo_epochs if ppo_epochs is not None else optimal_config['ppo_epochs']
        self.batch_size = batch_size if batch_size is not None else optimal_config['batch_size']
        optimal_lr = lr if lr is not None else optimal_config['lr']
        optimal_hidden_dim = hidden_dim if hidden_dim is not None else optimal_config['hidden_dim']
        
        self.max_grad_norm = max_grad_norm

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.verbose:
            print(f"PPO Agent using device: {self.device}")
            if any(param is None for param in [lr, gamma, gae_lambda, clip_eps, entropy_coef, value_coef, rollout_steps, ppo_epochs, batch_size, hidden_dim]):
                print(f"Using optimal hyperparameters: lr={optimal_lr}, entropy_coef={self.entropy_coef}, rollout_steps={self.rollout_steps}")

        # Policy network and optimizer with optimal parameters
        self.policy = PPOPolicy(state_dim, action_dim, optimal_hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=optimal_lr, eps=1e-5)
        
        # Learning rate scheduling (adapted based on optimal config)
        decay_factor = 0.9 if optimal_lr <= 0.001 else 0.85  # More aggressive for higher LR
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=25, gamma=decay_factor
        )

        # On-policy experience buffer
        self.buffer = []
        
        # FIXED: REMOVED reward normalization - this was breaking learning!
        # self.reward_normalizer = RunningMeanStd()  # REMOVED
        
        # Training statistics
        self.episode_count = 0
        self.update_count = 0
        self.total_steps = 0
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_values = deque(maxlen=100)
        
        # FIXED: Performance tracking with proper episode reward tracking
        self.episode_rewards = deque(maxlen=100)
        self.current_episode_reward = 0  # Track raw episode reward
        self.best_avg_reward = float('-inf')
        
        # FIXED: Store initial entropy coefficient for adaptive adjustment
        self.initial_entropy_coef = self.entropy_coef
        
        if self.verbose:
            print(f"PPO Agent initialized with {state_dim}D state space")
            print(f"Optimal config applied for grid: {self._get_grid_type(grid_path)}")
            print(f"Reward normalization: DISABLED (using raw rewards)")
    
    def _get_optimal_config(self, grid_path):
        """ADDED: Get optimal configuration based on grid path."""
        grid_type = self._get_grid_type(grid_path)
        
        if grid_type in self.OPTIMAL_CONFIGS:
            config = self.OPTIMAL_CONFIGS[grid_type].copy()
            if self.verbose:
                print(f"Using optimized config for {grid_type}")
        else:
            config = self.OPTIMAL_CONFIGS['default'].copy()
            if self.verbose:
                print(f"Using default config for unknown grid type: {grid_type}")
        
        return config
    
    def _get_grid_type(self, grid_path):
        """ADDED: Determine grid type from path."""
        if grid_path is None:
            return 'default'
        
        grid_name = str(grid_path).lower()
        
        # Map grid paths to configuration keys
        if 'open_space' in grid_name:
            return 'open_space'
        elif 'simple_restaurant' in grid_name:
            return 'simple_restaurant'
        elif 'corridor_test' in grid_name:
            return 'corridor_test'
        elif 'a1_grid' in grid_name or 'a1grid' in grid_name:
            return 'A1_grid'
        elif 'assignment2_main' in grid_name or 'assignment_2' in grid_name:
            return 'assignment2_main'
        elif 'maze_challenge' in grid_name or 'maze' in grid_name or 'challenge' in grid_name:
            return 'maze_challenge'
        else:
            return 'default'

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
        """FIXED: Update with proper reward handling and episode tracking."""
        if not self.buffer:
            return
            
        # FIXED: Use raw reward directly - NO NORMALIZATION
        self.buffer[-1]["reward"] = reward  # Raw reward for learning
        self.buffer[-1]["done"] = done
        
        # FIXED: Track raw episode reward separately
        self.current_episode_reward += reward
        
        if done:
            self.episode_count += 1
            
            # FIXED: Track episode performance using raw rewards
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Reset for next episode
            
            # Adaptive hyperparameters based on performance
            self._adapt_hyperparameters()
            
        # FIXED: Improved buffer management and learning conditions
        if done or len(self.buffer) >= self.rollout_steps:
            complete_experiences = [exp for exp in self.buffer if exp["reward"] is not None]
            # FIXED: Better minimum threshold based on batch size
            min_experiences = max(self.batch_size // 2, 8)
            
            if len(complete_experiences) >= min_experiences:
                self._learn()
            # FIXED: Always clear buffer after checking learning condition
            self.buffer.clear()

    def _adapt_hyperparameters(self):
        """FIXED: Adapt hyperparameters based on performance (grid-aware)."""
        if len(self.episode_rewards) >= 20:
            current_avg = np.mean(list(self.episode_rewards)[-20:])
            
            # FIXED: Grid-specific adaptation ranges based on initial entropy
            if self.initial_entropy_coef >= 0.03:
                max_entropy = 0.1
                min_entropy = 0.01
            else:
                max_entropy = 0.05
                min_entropy = 0.005
            
            # Increase exploration if performance is stagnating
            if current_avg <= self.best_avg_reward:
                self.entropy_coef = min(max_entropy, self.entropy_coef * 1.02)
            else:
                self.best_avg_reward = current_avg
                self.entropy_coef = max(min_entropy, self.entropy_coef * 0.99)

    def _learn(self):
        """FIXED: Enhanced PPO learning with reduced early exits."""
        # FIXED: Reduced early return conditions
        if len(self.buffer) < 1:
            return
        
        complete_buffer = [exp for exp in self.buffer if exp["reward"] is not None]
        if len(complete_buffer) < 1:
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

        # GAE computation with optimal lambda
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

        # PPO optimization with optimal epochs and batch size
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
                
                # PPO loss with optimal clipping
                ratio = torch.exp(new_logp - batch_old_logp)
                
                # FIXED: Less aggressive ratio checks to reduce learning skips
                max_ratio = 20.0 if self.clip_eps >= 0.25 else 10.0
                if ratio.max() > max_ratio or ratio.min() < (1.0 / max_ratio):
                    # Still continue learning but with warning
                    if self.verbose and self.update_count % 50 == 0:
                        print(f"Warning: Large policy ratio detected (max: {ratio.max():.2f})")
                
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
        # FIXED: Reset episode reward tracking
        self.current_episode_reward = 0

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
            'buffer_size': len(self.buffer),
            'optimal_config_used': True,
            'reward_normalization': 'DISABLED'  # Indicate fix applied
        }

    def save_agent(self, filepath: str):
        """Save PPO agent."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            # REMOVED: reward_normalizer (no longer used)
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'best_avg_reward': self.best_avg_reward,
            'current_episode_reward': self.current_episode_reward,  # Save episode tracking
            'hyperparameters': {
                'gamma': self.gamma,
                'lmbda': self.lmbda,
                'clip_eps': self.clip_eps,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'rollout_steps': self.rollout_steps,
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size,
                'initial_entropy_coef': self.initial_entropy_coef
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
            
            # REMOVED: reward_normalizer loading (no longer used)
            
            self.episode_count = checkpoint.get('episode_count', 0)
            self.update_count = checkpoint.get('update_count', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
            self.current_episode_reward = checkpoint.get('current_episode_reward', 0)
            
            # Load hyperparameters if available
            if 'hyperparameters' in checkpoint:
                hyp = checkpoint['hyperparameters']
                self.initial_entropy_coef = hyp.get('initial_entropy_coef', self.entropy_coef)
            
            if self.verbose:
                print(f"PPO Agent loaded from {filepath}")
                print(f"Episodes: {self.episode_count}, Updates: {self.update_count}, Steps: {self.total_steps}")
                
        except FileNotFoundError:
            if self.verbose:
                print(f"No saved PPO agent found at {filepath}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading PPO agent: {e}")


# REMOVED: RunningMeanStd class (no longer needed)