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
import torch

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
        
        # Convert NumPy types to native Python types
        python_config = {}
        for key, value in config.items():
            if isinstance(value, (np.integer, np.floating)):
                python_config[key] = float(value) if isinstance(value, np.floating) else int(value)
            else:
                python_config[key] = value
        
        # Create agent with specific hyperparameters using converted config
        agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            lr=python_config['lr'],
            batch_size=python_config['batch_size'],
            buffer_size=python_config['buffer_size'],
            target_update_freq=python_config['target_update_freq'],
            epsilon_decay=python_config['epsilon_decay'],
            epsilon_min=python_config['epsilon_min'],
            gamma=python_config['gamma'],
            verbose=False  # Reduce verbosity during hyperparameter search
        )
        
        # Update network architecture if specified
        if 'hidden_dim' in python_config:
            from agents.DQN_nn import DQNetwork
            agent.q_net = DQNetwork(8, 4, python_config['hidden_dim']).to(agent.device)
            agent.target_q_net = DQNetwork(8, 4, python_config['hidden_dim']).to(agent.device)
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())
            # Update optimizer with new network
            agent.optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=agent.lr)
        
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
            
            # Progress indicator
            if episode % 10 == 0:
                print(f"  Episode {episode}/{episodes} completed (reward: {episode_reward:.1f})")
        
        # Calculate metrics
        results = self._calculate_metrics(episode_rewards, evaluation_rewards, python_config)
        results['experiment_id'] = int(experiment_id)  # Ensure it's int, not numpy type
        results['config'] = python_config  # Use clean config
        
        print(f"Experiment {experiment_id} completed: Mean reward = {results['mean_reward']:.2f}")
        
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
        
        # Convert all metrics to native Python types (fix for JSON serialization)
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'early_performance': float(early_performance),
            'late_performance': float(late_performance),
            'improvement': float(improvement),
            'variance': float(variance),
            'convergence_episode': int(convergence_episode),
            'final_eval_performance': float(final_eval_performance),
            'best_episode_reward': float(np.max(episode_rewards)),
            'worst_episode_reward': float(np.min(episode_rewards))
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
                import traceback
                traceback.print_exc()
                continue
        
        # Analyze and save final results
        self._analyze_results()
        return self.results
    
    def _save_results(self):
        """Save results to JSON file with proper type conversion."""
        results_file = self.experiment_dir / "hyperparameter_results.json"
        
        # Convert numpy types to JSON serializable - comprehensive conversion
        def convert_for_json(obj):
            """Recursively convert numpy types to JSON serializable types."""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_for_json(item) for item in obj)
            else:
                return obj
        
        # Convert all results
        serializable_results = [convert_for_json(result) for result in self.results]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved {len(serializable_results)} results to {results_file}")
    
    def _analyze_results(self):
        """Analyze hyperparameter sweep results."""
        if not self.results:
         return
    
        # Convert to DataFrame and extract config parameters as separate columns
        df_data = []
        for result in self.results:
            # Flatten the config into the main result dict
            flattened_result = result.copy()
            if 'config' in result:
                flattened_result.update(result['config'])
            df_data.append(flattened_result)
        
        df = pd.DataFrame(df_data)
        
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
        
        # Convert best configs to JSON serializable format
        serializable_best_configs = {}
        for key, config in best_configs.items():
            serializable_best_configs[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                            for k, v in config.to_dict().items()}
        
        with open(analysis_file, 'w') as f:
            json.dump(serializable_best_configs, f, indent=2, default=str)
        
        # Create visualizations
        self._create_hyperparameter_plots(df)
        
        print(f"\n=== HYPERPARAMETER ANALYSIS COMPLETE ===")
        print(f"Results saved to: {self.experiment_dir}")
        print(f"Best mean reward: {df['mean_reward'].max():.2f}")
        print(f"Best improvement: {df['improvement'].max():.2f}")
        print(f"Best stability (low variance): {df['variance'].min():.2f}")
        
        # Print best configuration
        best_idx = df['mean_reward'].idxmax()
        best_config = df.loc[best_idx]
        print(f"\nBest Configuration (Mean Reward: {best_config['mean_reward']:.2f}):")
        hyperparams = ['lr', 'batch_size', 'target_update_freq', 'epsilon_decay', 'gamma', 'hidden_dim']
        for param in hyperparams:
            if param in best_config:
                print(f"  {param}: {best_config[param]}")
        
        return best_configs
    
    def _create_hyperparameter_plots(self, df):
        """Create visualization plots for hyperparameter analysis."""
        # Check which hyperparameters are available in the dataframe
        hyperparams = ['lr', 'batch_size', 'target_update_freq', 'epsilon_decay', 'gamma', 'hidden_dim']
        available_hyperparams = [param for param in hyperparams if param in df.columns]
        
        if not available_hyperparams:
            print("Warning: No hyperparameter columns found for plotting")
            return
        
        # Performance vs hyperparameters
        n_params = len(available_hyperparams)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle('Hyperparameter Analysis', fontsize=16)
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(available_hyperparams):
            ax = axes[i]
            
            try:
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
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {param}:\n{str(e)}', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Error: {param}')
        
        # Hide unused subplots
        for i in range(len(available_hyperparams), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "hyperparameter_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Correlation matrix (only if we have multiple hyperparameters)
        if len(available_hyperparams) > 1:
            try:
                numeric_cols = available_hyperparams + ['mean_reward']
                # Only include columns that actually exist and are numeric
                valid_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                
                if len(valid_cols) > 1:
                    correlation_matrix = df[valid_cols].corr()
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                    plt.colorbar()
                    plt.xticks(range(len(valid_cols)), valid_cols, rotation=45)
                    plt.yticks(range(len(valid_cols)), valid_cols)
                    plt.title('Hyperparameter Correlation Matrix')
                    
                    # Add correlation values
                    for i in range(len(valid_cols)):
                        for j in range(len(valid_cols)):
                            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                                    ha='center', va='center', color='black')
                    
                    plt.tight_layout()
                    plt.savefig(self.experiment_dir / "correlation_matrix.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    print("Correlation matrix plot saved")
                else:
                    print("Not enough numeric columns for correlation matrix")
            except Exception as e:
                print(f"Could not create correlation matrix: {e}")
        
        print(f"Hyperparameter analysis plots saved to {self.experiment_dir}")


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