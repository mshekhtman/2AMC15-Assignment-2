"""
Modified Evaluation Framework for Assignment 2 - PPO and DQN Focus.
Adapted from the existing evaluation_framework.py to focus on PPO vs DQN comparison.

SIMPLIFIED: Minimal changes to existing structure, just removing Random/Heuristic agents
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
    from logger import Logger
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from world import Environment
    from agents.DQN_agent import DQNAgent
    from agents.PPO_agent import PPOAgent
    from logger import Logger


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


class PPODQNEvaluator:
    """Evaluation framework focused on PPO vs DQN comparison - adapted from existing code."""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}
        self.experiment_dir = Path(f"experiments/ppo_dqn_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    
    def compare_ppo_dqn(self, episodes=100, num_seeds=3):
        """Compare PPO and DQN agents with comprehensive evaluation."""
        print("=== PPO vs DQN Comprehensive Comparison ===")
        
        # Agent configurations - ONLY PPO and DQN
        agent_configs = {
            'PPO': PPOAgent,
            'DQN': DQNAgent
        }
        
        comparison_results = {}
        
        for agent_name, agent_class in agent_configs.items():
            print(f"\nEvaluating {agent_name}...")
            
            # Create agent with optimized parameters
            if agent_name == 'PPO':
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
            else:  # DQN
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
        """Run training and collect comprehensive metrics - adapted from existing."""
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
        """Calculate all evaluation metrics - from existing code."""
        
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
        """Aggregate results across multiple random seeds - from existing code."""
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
        """Perform statistical comparison between agents - adapted from existing."""
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
            
            # Pairwise statistical tests
            pairwise_tests = {}
            if len(agent_names) == 2:  # PPO vs DQN
                agent1, agent2 = agent_names[0], agent_names[1]
                if agent1 in agent_values and agent2 in agent_values:
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
            
            metric_results['pairwise_tests'] = pairwise_tests
            statistical_results[metric] = metric_results
        
        return statistical_results
    
    def _create_comprehensive_plots(self, comparison_results, statistical_analysis):
        """Create comprehensive visualization plots - simplified from existing."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Learning curves comparison
        self._plot_learning_curves(comparison_results)
        
        # 2. Performance metrics comparison
        self._plot_performance_metrics(comparison_results)
        
        # 3. Statistical significance visualization
        self._plot_statistical_results(statistical_analysis)
    
    def _plot_learning_curves(self, comparison_results):
        """Plot learning curves with confidence intervals."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('PPO vs DQN Learning Curves Analysis', fontsize=16)
        
        colors = {'PPO': '#1f77b4', 'DQN': '#ff7f0e'}
        
        # Plot 1: Episode rewards
        ax = axes[0, 0]
        for (agent_name, results) in comparison_results.items():
            if 'episode_rewards' in results:
                mean_rewards = results['episode_rewards']['mean_series']
                std_rewards = results['episode_rewards']['std_series']
                episodes = range(len(mean_rewards))
                
                ax.plot(episodes, mean_rewards, label=agent_name, color=colors.get(agent_name, 'gray'), linewidth=2)
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
                ax.plot(episodes, success_rates, label=agent_name, color=colors.get(agent_name, 'gray'), linewidth=2)
        
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
                ax.plot(episodes, cumulative, label=agent_name, color=colors.get(agent_name, 'gray'), linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Training losses
        ax = axes[1, 1]
        for (agent_name, results) in comparison_results.items():
            if 'training_losses' in results:
                losses = results['training_losses']['mean_series']
                episodes = range(len(losses))
                ax.plot(episodes, losses, label=agent_name, color=colors.get(agent_name, 'gray'), linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "learning_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self, comparison_results):
        """Plot performance metrics comparison."""
        metrics = ['mean_reward', 'success_rate', 'sample_efficiency', 
                  'asymptotic_performance', 'stability_metric']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPO vs DQN Performance Metrics Comparison', fontsize=16)
        axes = axes.flatten()
        
        agent_names = list(comparison_results.keys())
        colors = ['#1f77b4', '#ff7f0e']
        
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
            ax.set_xticklabels(agent_names)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.2f}', ha='center', va='bottom')
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_results(self, statistical_analysis):
        """Plot statistical significance results."""
        if not statistical_analysis:
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = list(statistical_analysis.keys())
        p_values = []
        significances = []
        
        for metric in metrics:
            if 'pairwise_tests' in statistical_analysis[metric]:
                tests = statistical_analysis[metric]['pairwise_tests']
                if tests:
                    # Get first (and likely only) test result
                    test_key = list(tests.keys())[0]
                    p_val = tests[test_key]['p_value']
                    sig = tests[test_key]['significant']
                    p_values.append(p_val)
                    significances.append(sig)
                else:
                    p_values.append(1.0)
                    significances.append(False)
        
        # Create bar plot
        colors = ['red' if sig else 'gray' for sig in significances]
        bars = ax.bar(range(len(metrics)), p_values, color=colors, alpha=0.7)
        
        # Add significance line
        ax.axhline(y=0.05, color='black', linestyle='--', label='p = 0.05')
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_ylabel('P-value')
        ax.set_title('Statistical Significance: PPO vs DQN\n(Red bars indicate p < 0.05)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "statistical_significance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_comprehensive_results(self, comparison_results, statistical_analysis):
        """Save comprehensive results to files - from existing code."""
        
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
        with open(self.experiment_dir / "ppo_dqn_results.json", 'w') as f:
            json.dump(convert_for_json(comparison_results), f, indent=2)
        
        # Save statistical analysis
        with open(self.experiment_dir / "statistical_analysis.json", 'w') as f:
            json.dump(convert_for_json(statistical_analysis), f, indent=2)
        
        # Create summary report
        self._create_summary_report(comparison_results, statistical_analysis)
    
    def _create_summary_report(self, comparison_results, statistical_analysis):
        """Create a text summary report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PPO vs DQN EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Executive summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Find best performing agent for each metric
        key_metrics = ['mean_reward', 'success_rate', 'sample_efficiency', 'asymptotic_performance']
        
        for metric in key_metrics:
            if metric in comparison_results['PPO'] and metric in comparison_results['DQN']:
                ppo_val = comparison_results['PPO'][metric]['mean']
                dqn_val = comparison_results['DQN'][metric]['mean']
                
                if metric == 'sample_efficiency':  # Lower is better
                    best_agent = 'PPO' if ppo_val < dqn_val else 'DQN'
                else:  # Higher is better
                    best_agent = 'PPO' if ppo_val > dqn_val else 'DQN'
                
                report_lines.append(f"{metric.replace('_', ' ').title()}: {best_agent} performs better")
                report_lines.append(f"  PPO: {ppo_val:.2f}, DQN: {dqn_val:.2f}")
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
                    report_lines.append(f"{metric.replace('_', ' ').title()}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Statistical significance
        if statistical_analysis:
            report_lines.append("\n\nSTATISTICAL SIGNIFICANCE")
            report_lines.append("-" * 40)
            
            for metric, analysis in statistical_analysis.items():
                if 'pairwise_tests' in analysis and analysis['pairwise_tests']:
                    test_key = list(analysis['pairwise_tests'].keys())[0]
                    test_result = analysis['pairwise_tests'][test_key]
                    
                    significance = "statistically significant" if test_result['significant'] else "not statistically significant"
                    report_lines.append(f"{metric.replace('_', ' ').title()}: {significance} (p={test_result['p_value']:.4f})")
        
        # Save report
        with open(self.experiment_dir / "evaluation_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))


# Simplified execution function
def run_ppo_dqn_evaluation():
    """Run the PPO vs DQN evaluation - adapted from existing run_comprehensive_evaluation."""
    
    # Base configuration - using same structure as existing
    base_config = {
        'grid_path': Path('grid_configs/A1_grid.npy'),
        'sigma': 0.1,
        'agent_start_pos': (3, 11),
        'random_seed': 42
    }
    
    print("=" * 80)
    print("PPO vs DQN COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print("Testing PPO and DQN agents with:")
    print("• 100 episodes per run")
    print("• 5 random seeds for statistical validity")
    print("• Comprehensive RL metrics")
    print("• Statistical significance testing")
    print("=" * 80)
    
    # Run evaluation
    evaluator = PPODQNEvaluator(base_config)
    comparison_results, statistical_analysis = evaluator.compare_ppo_dqn(
        episodes=100, num_seeds=5
    )
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {evaluator.experiment_dir}")
    
    return comparison_results, statistical_analysis


if __name__ == "__main__":
    results, stats = run_ppo_dqn_evaluation()