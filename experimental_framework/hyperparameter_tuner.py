"""
Hyperparameter tuning framework for DQN experiments.
This systematically tests different hyperparameter combinations.
"""
import itertools
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

try:
    from world import Environment
    from agents.DQN_agent import DQNAgent
    from logger import Logger
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from world import Environment
    from agents.DQN_agent import DQNAgent
    from logger import Logger


class HyperparameterTuner:
    """Systematic hyperparameter optimization for DQN."""
    
    def __init__(self, base_config):
        """Initialize tuner with base configuration."""
        self.base_config = base_config
        self.results = []
        self.experiment_dir = Path(f"experiments/hyperparam_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def define_search_space(self):
        """Define hyperparameter search space based on RL literature."""
        return {
            # Learning rate: Critical for convergence
            'lr': [1e-4, 5e-4, 1e-3, 2e-3],
            
            # Network architecture: Affects representation capacity
            'hidden_dim': [64, 128, 256],
            
            # Experience replay: Memory and sample efficiency
            'batch_size': [32, 64, 128],
            'buffer_size': [25000, 50000, 100000],
            
            # Target network: Stability vs adaptation speed
            'target_update_freq': [250, 500, 1000],
            
            # Exploration: Exploration-exploitation tradeoff
            'epsilon_decay': [0.995, 0.99, 0.985],
            'epsilon_min': [0.01, 0.05, 0.1],
            
            # Training dynamics
            'gamma': [0.95, 0.99, 0.995],
        }
    
    def generate_configurations(self, method='grid', max_configs=50):
        """Generate hyperparameter configurations."""
        search_space = self.define_search_space()
        
        if method == 'grid':
            # Full grid search (potentially many combinations)
            configs = list(itertools.product(*search_space.values()))
            keys = list(search_space.keys())
            configurations = [dict(zip(keys, config)) for config in configs]
            
        elif method == 'random':
            # Random search - more efficient for high-dimensional spaces
            configurations = []
            keys = list(search_space.keys())
            
            for _ in range(max_configs):
                config = {}
                for key in keys:
                    config[key] = np.random.choice(search_space[key])
                configurations.append(config)
                
        # Limit number of configurations
        if len(configurations) > max_configs:
            configurations = np.random.choice(configurations, max_configs, replace=False)
            
        return configurations
    
    def run_experiment(self, config, experiment_id, episodes=100):
        """Run single experiment with given configuration."""
        print(f"\n=== Experiment {experiment_id} ===")
        print(f"Config: {config}")
        
        # Create agent with specific hyperparameters
        agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            lr=config['lr'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            target_update_freq=config['target_update_freq'],
            epsilon_decay=config['epsilon_decay'],
            epsilon_min=config['epsilon_min'],
            gamma=config['gamma']
        )
        
        # Update network architecture if specified
        if 'hidden_dim' in config:
            from agents.DQN_nn import DQNetwork
            agent.q_net = DQNetwork(8, 4, config['hidden_dim']).to(agent.device)
            agent.target_q_net = DQNetwork(8, 4, config['hidden_dim']).to(agent.device)
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())
        
        # Set up environment
        env = Environment(
            grid_fp=self.base_config['grid_path'],
            no_gui=True,
            sigma=self.base_config['sigma'],
            agent_start_pos=self.base_config['agent_start_pos'],
            random_seed=self.base_config['random_seed'],
            state_representation='continuous_vector'
        )
        
        # Training loop
        episode_rewards = []
        evaluation_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(1000):  # Max steps per episode
                action = agent.take_training_action(state, training=True)
                next_state, reward, terminated, info = env.step(action)
                agent.update(state, reward, action, next_state, terminated)
                
                episode_reward += reward
                state = next_state
                
                if terminated:
                    break
                    
            episode_rewards.append(episode_reward)
            
            # Periodic evaluation
            if episode % 20 == 0 and episode > 0:
                eval_reward = self._evaluate_agent(env, agent)
                evaluation_rewards.append((episode, eval_reward))
        
        # Calculate metrics
        results = self._calculate_metrics(episode_rewards, evaluation_rewards, config)
        results['experiment_id'] = experiment_id
        results['config'] = config
        
        return results
    
    def _evaluate_agent(self, env, agent, eval_episodes=5):
        """Evaluate agent performance with greedy policy."""
        total_rewards = []
        
        for _ in range(eval_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = agent.take_action(state)  # Greedy policy
                state, reward, terminated, info = env.step(action)
                episode_reward += reward
                
                if terminated:
                    break
                    
            total_rewards.append(episode_reward)
            
        return np.mean(total_rewards)
    
    def _calculate_metrics(self, episode_rewards, evaluation_rewards, config):
        """Calculate comprehensive performance metrics."""
        episode_rewards = np.array(episode_rewards)
        
        # Learning metrics
        early_performance = np.mean(episode_rewards[:20])  # First 20 episodes
        late_performance = np.mean(episode_rewards[-20:])  # Last 20 episodes
        improvement = late_performance - early_performance
        
        # Stability metrics
        variance = np.var(episode_rewards[-20:])  # Variance in final performance
        
        # Convergence metrics
        moving_avg = pd.Series(episode_rewards).rolling(window=20).mean()
        convergence_episode = self._find_convergence_point(moving_avg)
        
        # Evaluation performance
        if evaluation_rewards:
            eval_rewards = [r[1] for r in evaluation_rewards]
            final_eval_performance = eval_rewards[-1] if eval_rewards else 0
        else:
            final_eval_performance = 0
            
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'early_performance': early_performance,
            'late_performance': late_performance,
            'improvement': improvement,
            'variance': variance,
            'convergence_episode': convergence_episode,
            'final_eval_performance': final_eval_performance,
            'best_episode_reward': np.max(episode_rewards),
            'worst_episode_reward': np.min(episode_rewards)
        }
    
    def _find_convergence_point(self, moving_avg, threshold=50):
        """Find episode where performance converged (stabilized)."""
        if len(moving_avg) < 40:
            return len(moving_avg)
            
        for i in range(20, len(moving_avg) - 20):
            window = moving_avg[i:i+20]
            if window.std() < threshold:  # Low variance = convergence
                return i
                
        return len(moving_avg)
    
    def run_hyperparameter_sweep(self, method='random', max_configs=20, episodes=100):
        """Run complete hyperparameter sweep."""
        configurations = self.generate_configurations(method, max_configs)
        
        print(f"Running {len(configurations)} hyperparameter experiments...")
        
        for i, config in enumerate(configurations):
            try:
                results = self.run_experiment(config, i, episodes)
                self.results.append(results)
                
                # Save intermediate results
                self._save_results()
                
            except Exception as e:
                print(f"Experiment {i} failed: {e}")
                continue
        
        # Analyze and save final results
        self._analyze_results()
        return self.results
    
    def _save_results(self):
        """Save results to JSON file."""
        results_file = self.experiment_dir / "hyperparameter_results.json"
        
        # Convert numpy types to JSON serializable
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _analyze_results(self):
        """Analyze hyperparameter sweep results."""
        if not self.results:
            return
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Find best configurations
        best_configs = {
            'mean_reward': df.loc[df['mean_reward'].idxmax()],
            'improvement': df.loc[df['improvement'].idxmax()],
            'stability': df.loc[df['variance'].idxmin()],
            'convergence': df.loc[df['convergence_episode'].idxmin()],
            'eval_performance': df.loc[df['final_eval_performance'].idxmax()]
        }
        
        # Save analysis
        analysis_file = self.experiment_dir / "best_configurations.json"
        with open(analysis_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in best_configs.items()}, f, indent=2, default=str)
        
        # Create visualizations
        self._create_hyperparameter_plots(df)
        
        print(f"\n=== HYPERPARAMETER ANALYSIS COMPLETE ===")
        print(f"Results saved to: {self.experiment_dir}")
        print(f"Best mean reward: {df['mean_reward'].max():.2f}")
        print(f"Best improvement: {df['improvement'].max():.2f}")
        print(f"Best stability (low variance): {df['variance'].min():.2f}")
        
        return best_configs
    
    def _create_hyperparameter_plots(self, df):
        """Create visualization plots for hyperparameter analysis."""
        # Performance vs hyperparameters
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Analysis', fontsize=16)
        
        hyperparams = ['lr', 'batch_size', 'target_update_freq', 'epsilon_decay', 'gamma', 'hidden_dim']
        
        for i, param in enumerate(hyperparams):
            if i >= 6:
                break
                
            ax = axes[i//3, i%3]
            
            # Group by hyperparameter value and plot mean performance
            grouped = df.groupby(param)['mean_reward'].agg(['mean', 'std'])
            
            x_values = list(grouped.index)
            y_values = grouped['mean'].values
            y_errors = grouped['std'].values
            
            ax.errorbar(x_values, y_values, yerr=y_errors, marker='o', capsize=5)
            ax.set_xlabel(param)
            ax.set_ylabel('Mean Reward')
            ax.set_title(f'Performance vs {param}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "hyperparameter_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Correlation matrix
        numeric_cols = ['lr', 'batch_size', 'target_update_freq', 'epsilon_decay', 'gamma', 'hidden_dim', 'mean_reward']
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title('Hyperparameter Correlation Matrix')
        
        # Add correlation values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "correlation_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    base_config = {
        'grid_path': Path('grid_configs/A1_grid.npy'),
        'sigma': 0.1,
        'agent_start_pos': (3, 11),
        'random_seed': 42
    }
    
    tuner = HyperparameterTuner(base_config)
    results = tuner.run_hyperparameter_sweep(method='random', max_configs=15, episodes=50)