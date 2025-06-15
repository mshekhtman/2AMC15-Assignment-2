"""
Logger class for tracking and visualizing training progress.
Implemented by team member for saving learning curves with greedy policy evaluation.
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

class Logger:
    def __init__(self,
                 grid,
                 sigma,
                 gamma=0.99,
                 lr=1e-3,
                 batch_size=64,
                 buffer_size=50000,
                 min_replay_size=1000,
                 target_update_freq=500,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):
        """
        Initialize logger for tracking training progress.
        
        Args:
            grid: Grid file path being used for training
            sigma: Environment stochasticity
            gamma: Discount factor
            lr: Learning rate
            batch_size: DQN batch size
            buffer_size: Replay buffer size
            min_replay_size: Minimum replay size before training
            target_update_freq: Target network update frequency
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
        """
        self.grid = grid
        self.sigma = sigma
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_rewards = []
        self.DQN_rewards = {}
        
        # Ensure results directory exists
        Path("results").mkdir(exist_ok=True)
    
    def log_target_rewards(self, reward):
        """Log reward from training with epsilon-greedy policy."""
        self.target_rewards.append(reward)
    
    def log_DQN_rewards(self, episode, DQNreward):
        """Log reward from DQN evaluation with greedy policy (no exploration)."""
        self.DQN_rewards[episode] = DQNreward
    
    def plot_target_rewards(self):
        """Plot and save target rewards over episodes."""
        if not self.target_rewards:
            print("No target rewards to plot")
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        filename = f"results/{timestamp}_targetrewardsplot.png"
        
        plt.figure(figsize=(12, 6))
        
        # Plot raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.target_rewards, alpha=0.6, label='Episode Rewards')
        
        # Plot moving average for clearer trend
        if len(self.target_rewards) > 10:
            window = min(20, len(self.target_rewards) // 5)
            moving_avg = np.convolve(self.target_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.target_rewards)), moving_avg, 
                    color='red', linewidth=2, label=f'Moving Avg ({window} episodes)')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards (ε-greedy Policy)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot reward distribution
        plt.subplot(1, 2, 2)
        plt.hist(self.target_rewards, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Target rewards plot saved: {filename}")
    
    def plot_DQN_rewards(self):
        """Plot and save DQN evaluation rewards over episodes."""
        if not self.DQN_rewards:
            print("No DQN rewards to plot")
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        filename = f"results/{timestamp}_DQNrewardsplot.png"
        
        episodes = list(self.DQN_rewards.keys())
        rewards = list(self.DQN_rewards.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, 'o-', linewidth=2, markersize=6, label='DQN Evaluation Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DQN Performance (Greedy Policy Evaluation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add trend line if enough data points
        if len(episodes) > 2:
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            plt.plot(episodes, p(episodes), "--", alpha=0.8, color='red', 
                    label=f'Trend (slope: {z[0]:.2f})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"DQN rewards plot saved: {filename}")
    
    def get_summary_stats(self):
        """Get summary statistics of training."""
        stats = {
            'total_episodes': len(self.target_rewards),
            'avg_reward': np.mean(self.target_rewards) if self.target_rewards else 0,
            'best_reward': np.max(self.target_rewards) if self.target_rewards else 0,
            'worst_reward': np.min(self.target_rewards) if self.target_rewards else 0,
            'final_10_avg': np.mean(self.target_rewards[-10:]) if len(self.target_rewards) >= 10 else 0,
        }
        
        if self.DQN_rewards:
            dqn_rewards = list(self.DQN_rewards.values())
            stats.update({
                'dqn_evaluations': len(dqn_rewards),
                'dqn_avg_reward': np.mean(dqn_rewards),
                'dqn_best_reward': np.max(dqn_rewards),
                'dqn_latest_reward': dqn_rewards[-1] if dqn_rewards else 0
            })
        
        return stats
    
    def print_summary(self):
        """Print training summary."""
        stats = self.get_summary_stats()
        
        print(f"\n=== TRAINING SUMMARY ===")
        print(f"Grid: {self.grid}")
        print(f"Episodes: {stats['total_episodes']}")
        print(f"Average Reward: {stats['avg_reward']:.2f}")
        print(f"Best Reward: {stats['best_reward']:.2f}")
        print(f"Final 10 Episodes Avg: {stats['final_10_avg']:.2f}")
        
        if 'dqn_evaluations' in stats:
            print(f"\nDQN Evaluations: {stats['dqn_evaluations']}")
            print(f"DQN Average Reward: {stats['dqn_avg_reward']:.2f}")
            print(f"DQN Best Reward: {stats['dqn_best_reward']:.2f}")
            print(f"DQN Latest Reward: {stats['dqn_latest_reward']:.2f}")
        
        print("=" * 25)