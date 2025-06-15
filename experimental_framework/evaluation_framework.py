"""
Comprehensive Evaluation Framework for Assignment 2.
Implements RL evaluation metrics from literature and creates detailed analysis.
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
    from logger import Logger
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from world import Environment
    from agents.DQN_agent import DQNAgent
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


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for RL agents."""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}
        self.experiment_dir = Path(f"experiments/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    
    def compare_multiple_agents(self, agent_configs, episodes=100, num_seeds=3):
        """Compare multiple agents with comprehensive evaluation."""
        print("=== Multi-Agent Comprehensive Comparison ===")
        
        comparison_results = {}
        
        for agent_name, agent_class in agent_configs.items():
            print(f"\nEvaluating {agent_name}...")
            
            # Create agent
            if agent_name in ['Random', 'Heuristic']:
                agent = agent_class()
            else:
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
            if hasattr(agent, 'losses') and agent.losses:
                training_losses.append(np.mean(agent.losses[-10:]))  # Recent loss
            else:
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
            values = [result[metric] for result in all_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Aggregate time series (average across seeds)
        for metric in time_series_metrics:
            all_series = [result[metric] for result in all_results]
            
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
        """Perform statistical comparison between agents."""
        agent_names = list(comparison_results.keys())
        metrics_to_compare = ['mean_reward', 'success_rate', 'sample_efficiency', 
                             'asymptotic_performance', 'stability_metric']
        
        statistical_results = {}
        
        for metric in metrics_to_compare:
            metric_results = {}
            
            # Extract values for each agent
            agent_values = {}
            for agent_name in agent_names:
                agent_values[agent_name] = comparison_results[agent_name][metric]['values']
            
            # Pairwise statistical tests
            pairwise_tests = {}
            for i, agent1 in enumerate(agent_names):
                for agent2 in agent_names[i+1:]:
                    # Perform t-test
                    statistic, p_value = stats.ttest_ind(
                        agent_values[agent1], 
                        agent_values[agent2]
                    )
                    
                    pairwise_tests[f"{agent1}_vs_{agent2}"] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            # ANOVA test if more than 2 agents
            if len(agent_names) > 2:
                f_statistic, anova_p_value = stats.f_oneway(*agent_values.values())
                metric_results['anova'] = {
                    'f_statistic': f_statistic,
                    'p_value': anova_p_value,
                    'significant': anova_p_value < 0.05
                }
            
            metric_results['pairwise_tests'] = pairwise_tests
            statistical_results[metric] = metric_results
        
        return statistical_results
    
    def _create_comprehensive_plots(self, comparison_results, statistical_analysis):
        """Create comprehensive visualization plots."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Learning curves comparison
        self._plot_learning_curves(comparison_results)
        
        # 2. Performance metrics comparison
        self._plot_performance_metrics(comparison_results)
        
        # 3. Statistical significance heatmap
        self._plot_statistical_heatmap(statistical_analysis)
        
        # 4. Distribution plots
        self._plot_performance_distributions(comparison_results)
        
        # 5. Radar chart for multi-metric comparison
        self._plot_radar_chart(comparison_results)
    
    def _plot_learning_curves(self, comparison_results):
        """Plot learning curves with confidence intervals."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learning Curves Analysis', fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(comparison_results)))
        
        # Plot 1: Episode rewards
        ax = axes[0, 0]
        for (agent_name, results), color in zip(comparison_results.items(), colors):
            mean_rewards = results['episode_rewards']['mean_series']
            std_rewards = results['episode_rewards']['std_series']
            episodes = range(len(mean_rewards))
            
            ax.plot(episodes, mean_rewards, label=agent_name, color=color, linewidth=2)
            ax.fill_between(episodes, 
                           mean_rewards - std_rewards, 
                           mean_rewards + std_rewards, 
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Success rates over time
        ax = axes[0, 1]
        for (agent_name, results), color in zip(comparison_results.items(), colors):
            if 'success_rates_over_time' in results:
                success_rates = results['success_rates_over_time']['mean_series']
                episodes = range(len(success_rates))
                ax.plot(episodes, success_rates, label=agent_name, color=color, linewidth=2)
        
        ax.set_xlabel('Episode Window')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative regret
        ax = axes[1, 0]
        for (agent_name, results), color in zip(comparison_results.items(), colors):
            if 'regret_analysis' in results:
                regret = results['regret_analysis']['mean_series']
                episodes = range(len(regret))
                ax.plot(episodes, regret, label=agent_name, color=color, linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Training losses (for learning agents)
        ax = axes[1, 1]
        for (agent_name, results), color in zip(comparison_results.items(), colors):
            if 'training_losses' in results and agent_name not in ['Random', 'Heuristic']:
                losses = results['training_losses']['mean_series']
                episodes = range(len(losses))
                ax.plot(episodes, losses, label=agent_name, color=color, linewidth=2)
        
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
        fig.suptitle('Performance Metrics Comparison', fontsize=16)
        axes = axes.flatten()
        
        agent_names = list(comparison_results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(agent_names)))
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            means = [comparison_results[agent][metric]['mean'] for agent in agent_names]
            stds = [comparison_results[agent][metric]['std'] for agent in agent_names]
            
            bars = ax.bar(range(len(agent_names)), means, yerr=stds, 
                         capsize=5, color=colors, alpha=0.7)
            
            ax.set_xticks(range(len(agent_names)))
            ax.set_xticklabels(agent_names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.1f}', ha='center', va='bottom')
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_heatmap(self, statistical_analysis):
        """Plot statistical significance heatmap."""
        if not statistical_analysis:
            return
            
        # Create significance matrix
        metrics = list(statistical_analysis.keys())
        pairwise_comparisons = []
        p_values_matrix = []
        
        # Extract pairwise comparisons
        first_metric = list(statistical_analysis.values())[0]
        if 'pairwise_tests' in first_metric:
            pairwise_comparisons = list(first_metric['pairwise_tests'].keys())
        
        # Build p-value matrix
        for metric in metrics:
            metric_p_values = []
            for comparison in pairwise_comparisons:
                if 'pairwise_tests' in statistical_analysis[metric]:
                    p_value = statistical_analysis[metric]['pairwise_tests'][comparison]['p_value']
                    metric_p_values.append(p_value)
                else:
                    metric_p_values.append(1.0)
            p_values_matrix.append(metric_p_values)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(p_values_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(pairwise_comparisons)))
        ax.set_xticklabels(pairwise_comparisons, rotation=45, ha='right')
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('P-value')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(pairwise_comparisons)):
                text = f'{p_values_matrix[i][j]:.3f}'
                color = 'white' if p_values_matrix[i][j] < 0.05 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color)
        
        ax.set_title('Statistical Significance (P-values)\nWhite text indicates p < 0.05')
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "statistical_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distributions(self, comparison_results):
        """Plot performance distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Distributions', fontsize=16)
        
        metrics = ['mean_reward', 'success_rate', 'sample_efficiency', 'asymptotic_performance']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Prepare data for violin plot
            data_for_plot = []
            labels = []
            
            for agent_name, results in comparison_results.items():
                values = results[metric]['values']
                data_for_plot.extend(values)
                labels.extend([agent_name] * len(values))
            
            # Create DataFrame for seaborn
            df = pd.DataFrame({'Agent': labels, 'Value': data_for_plot})
            
            # Create violin plot
            sns.violinplot(data=df, x='Agent', y='Value', ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_chart(self, comparison_results):
        """Plot radar chart for multi-metric comparison."""
        metrics = ['mean_reward', 'success_rate', 'asymptotic_performance', 
                  'learning_speed', 'stability_metric', 'exploration_efficiency']
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized_data = {}
        
        for metric in metrics:
            all_values = []
            for agent_results in comparison_results.values():
                if metric in agent_results:
                    all_values.append(agent_results[metric]['mean'])
            
            if all_values:
                min_val, max_val = min(all_values), max(all_values)
                if max_val != min_val:
                    for agent_name, agent_results in comparison_results.items():
                        if agent_name not in normalized_data:
                            normalized_data[agent_name] = {}
                        
                        value = agent_results[metric]['mean']
                        # For stability_metric, lower is better, so invert
                        if metric == 'stability_metric':
                            normalized_value = 1 - (value - min_val) / (max_val - min_val)
                        else:
                            normalized_value = (value - min_val) / (max_val - min_val)
                        
                        normalized_data[agent_name][metric] = normalized_value
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(comparison_results)))
        
        for (agent_name, agent_data), color in zip(normalized_data.items(), colors):
            values = [agent_data.get(metric, 0) for metric in metrics]
            values += [values[0]]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=agent_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Performance Comparison\n(Normalized to 0-1 scale)', 
                    fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "radar_chart.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_comprehensive_results(self, comparison_results, statistical_analysis):
        """Save comprehensive results to files."""
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Save comparison results
        with open(self.experiment_dir / "comprehensive_results.json", 'w') as f:
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
        report_lines.append("COMPREHENSIVE EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Executive summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        # Find best performing agent for each metric
        key_metrics = ['mean_reward', 'success_rate', 'sample_efficiency', 'asymptotic_performance']
        
        for metric in key_metrics:
            best_agent = max(comparison_results.items(), 
                           key=lambda x: x[1][metric]['mean'])
            worst_agent = min(comparison_results.items(), 
                            key=lambda x: x[1][metric]['mean'])
            
            report_lines.append(f"{metric.replace('_', ' ').title()}:")
            report_lines.append(f"  Best: {best_agent[0]} ({best_agent[1][metric]['mean']:.2f})")
            report_lines.append(f"  Worst: {worst_agent[0]} ({worst_agent[1][metric]['mean']:.2f})")
            report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)
        
        for agent_name, results in comparison_results.items():
            report_lines.append(f"\n{agent_name.upper()}")
            report_lines.append("-" * len(agent_name))
            
            for metric in key_metrics:
                mean_val = results[metric]['mean']
                std_val = results[metric]['std']
                report_lines.append(f"{metric.replace('_', ' ').title()}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Statistical significance
        if statistical_analysis:
            report_lines.append("\n\nSTATISTICAL SIGNIFICANCE")
            report_lines.append("-" * 40)
            
            for metric, analysis in statistical_analysis.items():
                if 'anova' in analysis and analysis['anova']['significant']:
                    report_lines.append(f"{metric.replace('_', ' ').title()}: Significant differences detected (p={analysis['anova']['p_value']:.4f})")
                
                if 'pairwise_tests' in analysis:
                    significant_pairs = [pair for pair, test in analysis['pairwise_tests'].items() 
                                       if test['significant']]
                    if significant_pairs:
                        report_lines.append(f"  Significant pairwise differences: {', '.join(significant_pairs)}")
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        # Find overall best agent
        overall_scores = {}
        for agent_name, results in comparison_results.items():
            score = 0
            for metric in key_metrics:
                # Normalize and weight metrics
                score += results[metric]['mean']
            overall_scores[agent_name] = score
        
        best_overall = max(overall_scores.items(), key=lambda x: x[1])
        report_lines.append(f"Best overall performer: {best_overall[0]}")
        
        # Identify trade-offs
        report_lines.append("\nKey trade-offs identified:")
        if 'DQN' in comparison_results and 'Heuristic' in comparison_results:
            dqn_sample_eff = comparison_results['DQN']['sample_efficiency']['mean']
            heuristic_perf = comparison_results['Heuristic']['asymptotic_performance']['mean']
            report_lines.append(f"- DQN requires {dqn_sample_eff:.0f} episodes to converge")
            report_lines.append(f"- Heuristic achieves {heuristic_perf:.2f} performance immediately")
        
        # Save report
        with open(self.experiment_dir / "evaluation_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))


# Example usage and experiment runner
def run_comprehensive_evaluation():
    """Run the complete evaluation suite."""
    
    # Base configuration
    base_config = {
        'grid_path': Path('grid_configs/A1_grid.npy'),
        'sigma': 0.1,
        'agent_start_pos': (3, 11),
        'random_seed': 42
    }
    
    # Agent configurations for comparison
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    
    agent_configs = {
        'Random': RandomAgent,
        'Heuristic': HeuristicAgent,
        'DQN': DQNAgent,
        # 'PPO': PPOAgent  # TODO: Add when implemented
    }
    
    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator(base_config)
    comparison_results, statistical_analysis = evaluator.compare_multiple_agents(
        agent_configs, episodes=100, num_seeds=5
    )
    
    print(f"\nComprehensive evaluation complete!")
    print(f"Results saved to: {evaluator.experiment_dir}")
    
    return comparison_results, statistical_analysis


if __name__ == "__main__":
    results, stats = run_comprehensive_evaluation()