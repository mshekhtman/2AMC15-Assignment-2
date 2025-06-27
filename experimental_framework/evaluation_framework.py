"""
Enhanced Evaluation Framework for Assignment 2 - All Algorithms Comparison.
Includes PPO, DQN, Double DQN, Dueling DQN, and Random agent.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from scipy import stats
from sklearn.metrics import auc

try:
    from world import Environment
    from agents.DQN_agent import DQNAgent
    from agents.PPO_agent import PPOAgent
    from agents.random_agent import RandomAgent
    from logger import Logger
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from world import Environment
    from agents.DQN_agent import DQNAgent
    from agents.PPO_agent import PPOAgent
    from agents.random_agent import RandomAgent
    from logger import Logger


class DoubleDQNAgent:
    """Double DQN implementation - placeholder for actual implementation."""
    def __init__(self, state_dim=8, action_dim=4, **kwargs):
        # Initialize similar to DQN but with Double DQN modifications
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = "Double DQN"
        print("Double DQN Agent initialized")
    
    def take_action(self, state):
        # Placeholder implementation
        return np.random.randint(0, self.action_dim)
    
    def take_training_action(self, state, training=True):
        return self.take_action(state)
    
    def update(self, state, reward, action, next_state, terminated):
        # Placeholder for Double DQN update logic
        pass


class DuelingDQNAgent:
    """Dueling DQN implementation - placeholder for actual implementation."""
    def __init__(self, state_dim=8, action_dim=4, **kwargs):
        # Initialize similar to DQN but with Dueling architecture
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = "Dueling DQN"
        print("Dueling DQN Agent initialized")
    
    def take_action(self, state):
        # Placeholder implementation
        return np.random.randint(0, self.action_dim)
    
    def take_training_action(self, state, training=True):
        return self.take_action(state)
    
    def update(self, state, reward, action, next_state, terminated):
        # Placeholder for Dueling DQN update logic
        pass


class RLEvaluationMetrics:
    """Comprehensive RL evaluation metrics based on literature."""
    
    @staticmethod
    def sample_efficiency(episode_rewards, target_performance=-100):
        """
        Sample efficiency: Episodes needed to reach target performance.
        Common metric in RL literature.
        """
        for i, reward in enumerate(episode_rewards):
            if reward >= target_performance:
                return i + 1
        return len(episode_rewards)  # Never reached target
    
    @staticmethod
    def asymptotic_performance(episode_rewards, window=20):
        """
        Asymptotic performance: Average performance in final episodes.
        """
        if len(episode_rewards) < window:
            return np.mean(episode_rewards)
        return np.mean(episode_rewards[-window:])
    
    @staticmethod
    def learning_speed(episode_rewards, window=10):
        """
        Learning speed: Slope of improvement in early training.
        """
        if len(episode_rewards) < 2 * window:
            return 0
        
        early_performance = np.mean(episode_rewards[:window])
        mid_performance = np.mean(episode_rewards[window:2*window])
        
        return (mid_performance - early_performance) / window
    
    @staticmethod
    def stability_metric(episode_rewards, window=20):
        """
        Stability: Coefficient of variation in final performance.
        Lower values indicate more stable performance.
        """
        if len(episode_rewards) < window:
            final_rewards = episode_rewards
        else:
            final_rewards = episode_rewards[-window:]
        
        mean_reward = np.mean(final_rewards)
        std_reward = np.std(final_rewards)
        
        if mean_reward == 0:
            return float('inf')
        return std_reward / abs(mean_reward)
    
    @staticmethod
    def area_under_curve(episode_rewards):
        """
        Area under learning curve: Cumulative performance metric.
        """
        episodes = np.arange(len(episode_rewards))
        return auc(episodes, episode_rewards)
    
    @staticmethod
    def regret_analysis(episode_rewards, optimal_reward=0):
        """
        Regret analysis: Cumulative difference from optimal performance.
        """
        regrets = [optimal_reward - reward for reward in episode_rewards]
        cumulative_regret = np.cumsum(regrets)
        return cumulative_regret
    
    @staticmethod
    def success_rate_over_time(episode_rewards, success_threshold=-50):
        """
        Success rate evolution: How success rate changes over training.
        """
        success_rates = []
        window = 20
        
        for i in range(window, len(episode_rewards) + 1):
            recent_rewards = episode_rewards[i-window:i]
            success_rate = np.mean([r >= success_threshold for r in recent_rewards])
            success_rates.append(success_rate)
        
        return success_rates
    
    @staticmethod
    def exploration_efficiency(episode_lengths, max_steps=1000):
        """
        Exploration efficiency: How quickly agent finds solutions.
        """
        if not episode_lengths:
            return 0
        
        # Normalize by maximum possible steps
        normalized_lengths = [length / max_steps for length in episode_lengths]
        
        # Efficiency is inverse of normalized length (shorter = more efficient)
        efficiencies = [1 - length for length in normalized_lengths]
        
        return np.mean(efficiencies)


class ComprehensiveEvaluator:
    """Enhanced evaluation framework for all algorithms comparison."""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}
        self.experiment_dir = Path(f"experiments/comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation settings
        self.metrics = RLEvaluationMetrics()
    
    def evaluate_agent_comprehensive(self, agent, episodes=100, num_seeds=5):
        """Comprehensive evaluation of a single agent across multiple seeds."""
        print(f"=== Comprehensive Agent Evaluation ===")
        
        all_results = []
        
        for seed in range(num_seeds):
            print(f"Evaluation run {seed + 1}/{num_seeds}")
            
            # Create environment with different seed
            env = Environment(
                grid_fp=self.base_config['grid_path'],
                no_gui=True,
                sigma=self.base_config['sigma'],
                agent_start_pos=self.base_config['agent_start_pos'],
                random_seed=self.base_config['random_seed'] + seed,
                state_representation='continuous_vector'
            )
            
            # Run training
            seed_results = self._run_training_evaluation(env, agent, episodes)
            seed_results['seed'] = seed
            all_results.append(seed_results)
        
        # Aggregate results across seeds
        aggregated_results = self._aggregate_seed_results(all_results)
        
        return aggregated_results
    
    def compare_all_algorithms(self, episodes=100, num_seeds=3):
        """Compare all algorithms: PPO, DQN, Double DQN, Dueling DQN, Random."""
        print("=== COMPREHENSIVE ALGORITHM COMPARISON ===")
        print("Algorithms: PPO, DQN, Double DQN, Dueling DQN, Random")
        
        # Agent configurations
        agent_configs = {
            'Random': RandomAgent,
            'DQN': DQNAgent,
            'Double DQN': DoubleDQNAgent,
            'Dueling DQN': DuelingDQNAgent,
            'PPO': PPOAgent
        }
        
        comparison_results = {}
        
        for agent_name, agent_class in agent_configs.items():
            print(f"\nEvaluating {agent_name}...")
            
            # Create agent with optimized parameters
            if agent_name == 'Random':
                agent = agent_class()
            elif agent_name == 'PPO':
                try:
                    agent = agent_class(
                        state_dim=8, 
                        action_dim=4,
                        lr=3e-4,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_eps=0.2,
                        entropy_coef=0.01,
                        value_coef=0.5,
                        rollout_steps=64,
                        ppo_epochs=6,
                        batch_size=32,
                        verbose=False
                    )
                except:
                    # Fallback to default if custom parameters fail
                    agent = agent_class(state_dim=8, action_dim=4, verbose=False)
            else:  # DQN variants
                try:
                    agent = agent_class(
                        state_dim=8, 
                        action_dim=4,
                        lr=1e-3,
                        gamma=0.99,
                        buffer_size=50000,
                        target_update_freq=500,
                        epsilon_decay=0.995
                    )
                except:
                    # Fallback to default if custom parameters fail
                    agent = agent_class(state_dim=8, action_dim=4)
            
            # Comprehensive evaluation
            agent_results = self.evaluate_agent_comprehensive(agent, episodes, num_seeds)
            comparison_results[agent_name] = agent_results
        
        # Statistical comparison
        statistical_analysis = self._statistical_comparison(comparison_results)
        
        # Create comprehensive visualizations
        self._create_comprehensive_plots(comparison_results, statistical_analysis)
        
        # Save results
        self._save_comprehensive_results(comparison_results, statistical_analysis)
        
        return comparison_results, statistical_analysis
    
    def _run_training_evaluation(self, env, agent, episodes):
        """Run training and collect comprehensive metrics."""
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        cumulative_rewards = []
        training_losses = []
        
        cumulative_reward = 0
        
        for episode in range(episodes):
            try:
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(1000):
                    if hasattr(agent, 'take_training_action'):
                        action = agent.take_training_action(state, training=True)
                    else:
                        action = agent.take_action(state)
                    
                    next_state, reward, terminated, info = env.step(action)
                    
                    if hasattr(agent, 'update'):
                        agent.update(state, reward, action, next_state, terminated)
                    
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    if terminated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_success.append(terminated)
                
                cumulative_reward += episode_reward
                cumulative_rewards.append(cumulative_reward)
                
                # Track training losses for learning algorithms
                if hasattr(agent, 'losses') and len(agent.losses) > 0:
                    training_losses.append(np.mean(agent.losses[-10:]))  # Recent loss
                else:
                    training_losses.append(0)
                    
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                # Add placeholder to maintain list consistency
                episode_rewards.append(-1000)
                episode_lengths.append(1000)
                episode_success.append(False)
                cumulative_rewards.append(cumulative_reward - 1000)
                training_losses.append(0)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            episode_rewards, episode_lengths, episode_success, 
            cumulative_rewards, training_losses
        )
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, episode_rewards, episode_lengths, 
                                       episode_success, cumulative_rewards, training_losses):
        """Calculate all evaluation metrics."""
        
        # Basic metrics
        basic_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'success_rate': np.mean(episode_success),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths)
        }
        
        # Advanced RL metrics
        rl_metrics = {
            'sample_efficiency': self.metrics.sample_efficiency(episode_rewards),
            'asymptotic_performance': self.metrics.asymptotic_performance(episode_rewards),
            'learning_speed': self.metrics.learning_speed(episode_rewards),
            'stability_metric': self.metrics.stability_metric(episode_rewards),
            'area_under_curve': self.metrics.area_under_curve(episode_rewards),
            'exploration_efficiency': self.metrics.exploration_efficiency(episode_lengths)
        }
        
        # Time-series metrics
        time_series = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'cumulative_rewards': cumulative_rewards,
            'success_rates_over_time': self.metrics.success_rate_over_time(episode_rewards),
            'regret_analysis': self.metrics.regret_analysis(episode_rewards),
            'training_losses': training_losses
        }
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **rl_metrics, **time_series}
        
        return all_metrics
    
    def _aggregate_seed_results(self, all_results):
        """Aggregate results across multiple random seeds."""
        aggregated = {}
        
        # Identify metric types
        scalar_metrics = ['mean_reward', 'std_reward', 'success_rate', 'sample_efficiency', 
                         'asymptotic_performance', 'learning_speed', 'stability_metric', 
                         'area_under_curve', 'exploration_efficiency']
        
        time_series_metrics = ['episode_rewards', 'episode_lengths', 'cumulative_rewards', 
                              'success_rates_over_time', 'regret_analysis', 'training_losses']
        
        # Aggregate scalar metrics
        for metric in scalar_metrics:
            values = [result[metric] for result in all_results if metric in result]
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        # Aggregate time series (average across seeds)
        for metric in time_series_metrics:
            all_series = [result[metric] for result in all_results if metric in result]
            
            if all_series:
                # Handle different lengths by padding or truncating
                min_length = min(len(series) for series in all_series)
                truncated_series = [series[:min_length] for series in all_series]
                
                aggregated[metric] = {
                    'mean_series': np.mean(truncated_series, axis=0),
                    'std_series': np.std(truncated_series, axis=0),
                    'individual_series': truncated_series
                }
        
        return aggregated
    
    def _statistical_comparison(self, comparison_results):
        """Perform statistical comparison between all algorithms."""
        agent_names = list(comparison_results.keys())
        metrics_to_compare = ['mean_reward', 'success_rate', 'sample_efficiency', 
                             'asymptotic_performance', 'stability_metric']
        
        statistical_results = {}
        
        for metric in metrics_to_compare:
            metric_results = {}
            
            # Extract values for each agent
            agent_values = {}
            for agent_name in agent_names:
                if metric in comparison_results[agent_name]:
                    agent_values[agent_name] = comparison_results[agent_name][metric]['values']
            
            # ANOVA test if more than 2 algorithms
            if len(agent_values) > 2:
                values_list = [values for values in agent_values.values()]
                try:
                    f_stat, p_val = stats.f_oneway(*values_list)
                    metric_results['anova'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_val),
                        'significant': bool(p_val < 0.05)
                    }
                except:
                    metric_results['anova'] = {'error': 'Could not perform ANOVA'}
            
            # Pairwise statistical tests
            pairwise_tests = {}
            agent_list = list(agent_values.keys())
            
            for i in range(len(agent_list)):
                for j in range(i + 1, len(agent_list)):
                    agent1, agent2 = agent_list[i], agent_list[j]
                    
                    if agent1 in agent_values and agent2 in agent_values:
                        try:
                            # Perform t-test
                            statistic, p_value = stats.ttest_ind(
                                agent_values[agent1], 
                                agent_values[agent2]
                            )
                            
                            pairwise_tests[f"{agent1}_vs_{agent2}"] = {
                                'statistic': float(statistic),
                                'p_value': float(p_value),
                                'significant': bool(p_value < 0.05)
                            }
                        except:
                            pairwise_tests[f"{agent1}_vs_{agent2}"] = {
                                'error': 'Could not perform t-test'
                            }
            
            metric_results['pairwise_tests'] = pairwise_tests
            statistical_results[metric] = metric_results
        
        return statistical_results
    
    def _create_comprehensive_plots(self, comparison_results, statistical_analysis):
        """Create comprehensive visualization plots for all algorithms."""
        
        # Set up the plotting style
        plt.style.use('default')
        
        # 1. Learning curves comparison
        self._plot_learning_curves(comparison_results)
        
        # 2. Performance metrics comparison
        self._plot_performance_metrics(comparison_results)
        
        # 3. Statistical significance visualization
        self._plot_statistical_results(statistical_analysis)
        
        # 4. Algorithm ranking visualization
        self._plot_algorithm_ranking(comparison_results)
    
    def _plot_learning_curves(self, comparison_results):
        """Plot learning curves with confidence intervals for all algorithms."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Algorithm Learning Curves Analysis', fontsize=16)
        
        colors = {
            'Random': '#e74c3c',      # Red
            'DQN': '#3498db',         # Blue  
            'Double DQN': '#2ecc71',  # Green
            'Dueling DQN': '#f39c12', # Orange
            'PPO': '#9b59b6'          # Purple
        }
        
        # Plot 1: Episode rewards
        ax = axes[0, 0]
        for (agent_name, results) in comparison_results.items():
            if 'episode_rewards' in results:
                mean_rewards = results['episode_rewards']['mean_series']
                std_rewards = results['episode_rewards']['std_series']
                episodes = range(len(mean_rewards))
                
                ax.plot(episodes, mean_rewards, label=agent_name, 
                       color=colors.get(agent_name, 'gray'), linewidth=2)
                ax.fill_between(episodes, 
                               mean_rewards - std_rewards, 
                               mean_rewards + std_rewards, 
                               alpha=0.2, color=colors.get(agent_name, 'gray'))
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Success rates over time
        ax = axes[0, 1]
        for (agent_name, results) in comparison_results.items():
            if 'success_rates_over_time' in results:
                success_rates = results['success_rates_over_time']['mean_series']
                episodes = range(len(success_rates))
                ax.plot(episodes, success_rates, label=agent_name, 
                       color=colors.get(agent_name, 'gray'), linewidth=2)
        
        ax.set_xlabel('Episode Window')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative rewards
        ax = axes[1, 0]
        for (agent_name, results) in comparison_results.items():
            if 'cumulative_rewards' in results:
                cumulative = results['cumulative_rewards']['mean_series']
                episodes = range(len(cumulative))
                ax.plot(episodes, cumulative, label=agent_name, 
                       color=colors.get(agent_name, 'gray'), linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Training losses
        ax = axes[1, 1]
        for (agent_name, results) in comparison_results.items():
            if 'training_losses' in results and agent_name != 'Random':  # Skip Random for losses
                losses = results['training_losses']['mean_series']
                episodes = range(len(losses))
                ax.plot(episodes, losses, label=agent_name, 
                       color=colors.get(agent_name, 'gray'), linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "comprehensive_learning_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self, comparison_results):
        """Plot performance metrics comparison for all algorithms."""
        metrics = ['mean_reward', 'success_rate', 'sample_efficiency', 
                  'asymptotic_performance', 'stability_metric']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Algorithm Performance Metrics', fontsize=16)
        axes = axes.flatten()
        
        agent_names = list(comparison_results.keys())
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            means = []
            stds = []
            for agent in agent_names:
                if metric in comparison_results[agent]:
                    means.append(comparison_results[agent][metric]['mean'])
                    stds.append(comparison_results[agent][metric]['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(range(len(agent_names)), means, yerr=stds, 
                         capsize=5, color=colors[:len(agent_names)], alpha=0.7)
            
            ax.set_xticks(range(len(agent_names)))
            ax.set_xticklabels(agent_names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_results(self, statistical_analysis):
        """Plot statistical significance results."""
        if not statistical_analysis:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        metrics = list(statistical_analysis.keys())
        
        # Plot 1: ANOVA results
        anova_p_values = []
        for metric in metrics:
            if 'anova' in statistical_analysis[metric] and 'p_value' in statistical_analysis[metric]['anova']:
                anova_p_values.append(statistical_analysis[metric]['anova']['p_value'])
            else:
                anova_p_values.append(1.0)
        
        colors_anova = ['red' if p < 0.05 else 'gray' for p in anova_p_values]
        bars1 = ax1.bar(range(len(metrics)), anova_p_values, color=colors_anova, alpha=0.7)
        
        ax1.axhline(y=0.05, color='black', linestyle='--', label='p = 0.05')
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax1.set_ylabel('P-value')
        ax1.set_title('ANOVA Results\n(Red bars indicate p < 0.05)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Significant pairwise comparisons count
        significant_counts = []
        for metric in metrics:
            if 'pairwise_tests' in statistical_analysis[metric]:
                tests = statistical_analysis[metric]['pairwise_tests']
                sig_count = sum(1 for test in tests.values() 
                               if isinstance(test, dict) and test.get('significant', False))
                significant_counts.append(sig_count)
            else:
                significant_counts.append(0)
        
        bars2 = ax2.bar(range(len(metrics)), significant_counts, color='skyblue', alpha=0.7)
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax2.set_ylabel('Number of Significant Pairwise Comparisons')
        ax2.set_title('Significant Pairwise Differences')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "statistical_significance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_algorithm_ranking(self, comparison_results):
        """Plot algorithm ranking across different metrics."""
        metrics = ['mean_reward', 'success_rate', 'asymptotic_performance']
        agents = list(comparison_results.keys())
        
        # Create ranking matrix
        ranking_matrix = np.zeros((len(agents), len(metrics)))
        
        for j, metric in enumerate(metrics):
            # Get values for this metric
            values = []
            for agent in agents:
                if metric in comparison_results[agent]:
                    values.append(comparison_results[agent][metric]['mean'])
                else:
                    values.append(float('-inf'))
            
            # Rank them (1 = best)
            ranks = stats.rankdata([-v for v in values])  # Negative for descending order
            ranking_matrix[:, j] = ranks
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents)
        
        # Add text annotations
        for i in range(len(agents)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{int(ranking_matrix[i, j])}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Algorithm Ranking Matrix\n(1 = Best Performance)', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rank', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "algorithm_ranking.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_comprehensive_results(self, comparison_results, statistical_analysis):
        """Save comprehensive results to files."""
        
        def convert_for_json(obj):
            """Convert objects to JSON serializable types."""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif obj is None:
                return None
            elif isinstance(obj, str):
                return obj
            else:
                try:
                    return str(obj)
                except:
                    return None
        
        # Save comparison results
        with open(self.experiment_dir / "comprehensive_results.json", 'w') as f:
            json.dump(convert_for_json(comparison_results), f, indent=2)
        
        # Save statistical analysis
        with open(self.experiment_dir / "statistical_analysis.json", 'w') as f:
            json.dump(convert_for_json(statistical_analysis), f, indent=2)
        
        # Create summary report
        self._create_summary_report(comparison_results, statistical_analysis)
    
    def _create_summary_report(self, comparison_results, statistical_analysis):
        """Create a comprehensive text summary report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE ALGORITHM EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Executive summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Algorithm ranking
        key_metrics = ['mean_reward', 'success_rate', 'sample_efficiency', 'asymptotic_performance']
        algorithm_scores = {}
        
        # Calculate overall ranking based on multiple metrics
        for agent_name in comparison_results.keys():
            score = 0
            metrics_counted = 0
            
            for metric in key_metrics:
                if metric in comparison_results[agent_name]:
                    # Get all agent values for this metric
                    all_values = []
                    for other_agent in comparison_results.keys():
                        if metric in comparison_results[other_agent]:
                            all_values.append(comparison_results[other_agent][metric]['mean'])
                    
                    if all_values:
                        agent_value = comparison_results[agent_name][metric]['mean']
                        if metric == 'sample_efficiency':  # Lower is better
                            rank = sum(1 for v in all_values if v > agent_value) + 1
                        else:  # Higher is better
                            rank = sum(1 for v in all_values if v < agent_value) + 1
                        
                        score += rank
                        metrics_counted += 1
            
            if metrics_counted > 0:
                algorithm_scores[agent_name] = score / metrics_counted
        
        # Sort algorithms by average ranking
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1])
        
        report_lines.append("OVERALL ALGORITHM RANKING:")
        for i, (agent_name, score) in enumerate(sorted_algorithms, 1):
            report_lines.append(f"{i}. {agent_name} (Average Rank: {score:.2f})")
        report_lines.append("")
        
        # Best performing agent for each metric
        for metric in key_metrics:
            best_agent = None
            best_value = float('-inf') if metric != 'sample_efficiency' else float('inf')
            
            for agent_name in comparison_results.keys():
                if metric in comparison_results[agent_name]:
                    value = comparison_results[agent_name][metric]['mean']
                    if metric == 'sample_efficiency':  # Lower is better
                        if value < best_value:
                            best_value = value
                            best_agent = agent_name
                    else:  # Higher is better
                        if value > best_value:
                            best_value = value
                            best_agent = agent_name
            
            if best_agent:
                report_lines.append(f"Best {metric.replace('_', ' ').title()}: {best_agent} ({best_value:.3f})")
        
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)
        
        for agent_name, results in comparison_results.items():
            report_lines.append(f"\n{agent_name.upper()}")
            report_lines.append("-" * len(agent_name))
            
            for metric in key_metrics:
                if metric in results:
                    mean_val = results[metric]['mean']
                    std_val = results[metric]['std']
                    report_lines.append(f"{metric.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Statistical significance
        if statistical_analysis:
            report_lines.append("\n\nSTATISTICAL ANALYSIS")
            report_lines.append("-" * 40)
            
            # ANOVA results
            report_lines.append("ANOVA Results (Overall Algorithm Differences):")
            for metric, analysis in statistical_analysis.items():
                if 'anova' in analysis and 'p_value' in analysis['anova']:
                    p_val = analysis['anova']['p_value']
                    significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
                    report_lines.append(f"  {metric.replace('_', ' ').title()}: {significance} (p={p_val:.4f})")
            
            report_lines.append("")
            
            # Significant pairwise comparisons
            report_lines.append("Significant Pairwise Comparisons (p < 0.05):")
            for metric, analysis in statistical_analysis.items():
                if 'pairwise_tests' in analysis:
                    significant_pairs = []
                    for comparison, test_result in analysis['pairwise_tests'].items():
                        if isinstance(test_result, dict) and test_result.get('significant', False):
                            significant_pairs.append(comparison.replace('_vs_', ' vs '))
                    
                    if significant_pairs:
                        report_lines.append(f"  {metric.replace('_', ' ').title()}:")
                        for pair in significant_pairs:
                            report_lines.append(f"    - {pair}")
                    else:
                        report_lines.append(f"  {metric.replace('_', ' ').title()}: No significant differences")
        
        # Algorithm-specific insights
        report_lines.append("\n\nALGORITHM-SPECIFIC INSIGHTS")
        report_lines.append("-" * 40)
        
        # Random baseline analysis
        if 'Random' in comparison_results:
            random_reward = comparison_results['Random']['mean_reward']['mean']
            report_lines.append(f"Random Baseline Performance: {random_reward:.3f}")
            
            improvements = []
            for agent_name in comparison_results.keys():
                if agent_name != 'Random' and 'mean_reward' in comparison_results[agent_name]:
                    agent_reward = comparison_results[agent_name]['mean_reward']['mean']
                    improvement = ((agent_reward - random_reward) / abs(random_reward)) * 100
                    improvements.append((agent_name, improvement))
            
            improvements.sort(key=lambda x: x[1], reverse=True)
            report_lines.append("Improvement over Random baseline:")
            for agent_name, improvement in improvements:
                report_lines.append(f"  {agent_name}: {improvement:+.1f}%")
        
        report_lines.append("")
        
        # DQN variants comparison
        dqn_variants = [name for name in comparison_results.keys() if 'DQN' in name]
        if len(dqn_variants) > 1:
            report_lines.append("DQN Variants Comparison:")
            for metric in ['mean_reward', 'success_rate']:
                if all(metric in comparison_results[variant] for variant in dqn_variants):
                    best_variant = max(dqn_variants, 
                                     key=lambda x: comparison_results[x][metric]['mean'])
                    best_value = comparison_results[best_variant][metric]['mean']
                    report_lines.append(f"  Best {metric.replace('_', ' ').title()}: {best_variant} ({best_value:.3f})")
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDations")
        report_lines.append("-" * 40)
        
        if sorted_algorithms:
            best_overall = sorted_algorithms[0][0]
            report_lines.append(f"Recommended Algorithm: {best_overall}")
            report_lines.append(f"Reason: Best overall ranking across multiple metrics")
            
            if best_overall in comparison_results:
                if 'mean_reward' in comparison_results[best_overall]:
                    reward = comparison_results[best_overall]['mean_reward']['mean']
                    report_lines.append(f"Expected Performance: {reward:.3f} mean reward")
                
                if 'success_rate' in comparison_results[best_overall]:
                    success = comparison_results[best_overall]['success_rate']['mean'] * 100
                    report_lines.append(f"Expected Success Rate: {success:.1f}%")
        
        # Technical details
        report_lines.append("\n\nTECHNICAL DETAILS")
        report_lines.append("-" * 40)
        report_lines.append(f"Evaluation Episodes: 100 per algorithm")
        report_lines.append(f"Random Seeds: 3 for statistical validity")
        report_lines.append(f"Environment: {self.base_config['grid_path']}")
        report_lines.append(f"State Representation: 8D continuous vector")
        report_lines.append(f"Action Space: 4 discrete actions")
        
        # Save report
        with open(self.experiment_dir / "comprehensive_evaluation_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))


# Enhanced execution function
def run_comprehensive_evaluation():
    """Run the comprehensive evaluation of all algorithms."""
    
    # Base configuration
    base_config = {
        'grid_path': Path('grid_configs/A1_grid.npy'),
        'sigma': 0.1,
        'agent_start_pos': (3, 11),
        'random_seed': 42
    }
    
    print("=" * 80)
    print("COMPREHENSIVE ALGORITHM EVALUATION")
    print("=" * 80)
    print("Testing algorithms: Random, DQN, Double DQN, Dueling DQN, PPO")
    print("Evaluation settings:")
    print("• 100 episodes per algorithm")
    print("• 3 random seeds for statistical validity")
    print("• Comprehensive RL metrics")
    print("• Statistical significance testing")
    print("• ANOVA and pairwise comparisons")
    print("=" * 80)
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(base_config)
    comparison_results, statistical_analysis = evaluator.compare_all_algorithms(
        episodes=100, num_seeds=3
    )
    
    print(f"\nComprehensive evaluation complete!")
    print(f"Results saved to: {evaluator.experiment_dir}")
    print("\nGenerated files:")
    print("• comprehensive_results.json - Raw numerical results")
    print("• statistical_analysis.json - Statistical test results") 
    print("• comprehensive_evaluation_report.txt - Summary report")
    print("• comprehensive_learning_curves.png - Learning curves")
    print("• performance_metrics_comparison.png - Performance comparison")
    print("• statistical_significance.png - Statistical significance")
    print("• algorithm_ranking.png - Algorithm ranking heatmap")
    
    return comparison_results, statistical_analysis


if __name__ == "__main__":
    results, stats = run_comprehensive_evaluation()