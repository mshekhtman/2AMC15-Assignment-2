"""
Master Experiment Runner for Assignment 2.
Coordinates all experimental phases for comprehensive analysis.
"""
import argparse
from pathlib import Path
from datetime import datetime
import json

# Import all experimental frameworks
try:
    from hyperparameter_tuner import HyperparameterTuner
    from algorithm_comparison import AlgorithmComparison, DoubleDQNAgent, DuelingDQNAgent
    from ablation_studies import AblationStudy
    from evaluation_framework import ComprehensiveEvaluator
    from agents.DQN_agent import DQNAgent
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from hyperparameter_tuner import HyperparameterTuner
    from algorithm_comparison import AlgorithmComparison, DoubleDQNAgent, DuelingDQNAgent
    from ablation_studies import AblationStudy
    from evaluation_framework import ComprehensiveEvaluator
    from agents.DQN_agent import DQNAgent
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent

class MasterExperimentRunner:
    """Coordinates all experimental phases for Assignment 2."""
    
    def __init__(self, config):
        self.config = config
        self.master_dir = Path(f"experiments/master_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.master_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'hyperparameter_tuning': None,
            'algorithm_comparison': None,
            'ablation_studies': None,
            'comprehensive_evaluation': None,
            'best_configurations': None
        }
        
        print(f"Master experiment directory: {self.master_dir}")
    
    def run_complete_experimental_suite(self, quick_mode=False):
        """Run the complete experimental suite for Assignment 2."""
        
        print("=" * 80)
        print("ASSIGNMENT 2 - COMPREHENSIVE EXPERIMENTAL ANALYSIS")
        print("=" * 80)
        
        # Adjust parameters for quick vs full mode
        if quick_mode:
            episodes = 50
            num_seeds = 3
            max_configs = 10
            print("Running in QUICK MODE (reduced parameters for testing)")
        else:
            episodes = 100
            num_seeds = 5
            max_configs = 20
            print("Running in FULL MODE (complete analysis)")
        
        print(f"Episodes per experiment: {episodes}")
        print(f"Random seeds per comparison: {num_seeds}")
        print(f"Max hyperparameter configurations: {max_configs}")
        print()
        
        # Phase 1: Hyperparameter Optimization
        print("\n" + "="*50)
        print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
        print("="*50)
        
        self.results['hyperparameter_tuning'] = self._run_hyperparameter_optimization(
            episodes, max_configs
        )
        
        # Phase 2: Algorithm Comparison  
        print("\n" + "="*50)
        print("PHASE 2: ALGORITHM COMPARISON")
        print("="*50)
        
        self.results['algorithm_comparison'] = self._run_algorithm_comparison(
            episodes, num_seeds
        )
        
        # Phase 3: Ablation Studies
        print("\n" + "="*50)
        print("PHASE 3: ABLATION STUDIES")
        print("="*50)
        
        self.results['ablation_studies'] = self._run_ablation_studies(episodes)
        
        # Phase 4: Comprehensive Evaluation
        print("\n" + "="*50)
        print("PHASE 4: COMPREHENSIVE EVALUATION")
        print("="*50)
        
        self.results['comprehensive_evaluation'] = self._run_comprehensive_evaluation(
            episodes, num_seeds
        )
        
        # Phase 5: Final Analysis and Recommendations
        print("\n" + "="*50)
        print("PHASE 5: FINAL ANALYSIS")
        print("="*50)
        
        self._generate_final_analysis()
        
        print("\n" + "="*80)
        print("EXPERIMENTAL SUITE COMPLETE!")
        print(f"All results saved to: {self.master_dir}")
        print("="*80)
        
        return self.results
    
    def _run_hyperparameter_optimization(self, episodes, max_configs):
        """Phase 1: Hyperparameter optimization."""
        print("Optimizing DQN hyperparameters...")
        
        tuner = HyperparameterTuner(self.config)
        results = tuner.run_hyperparameter_sweep(
            method='random', 
            max_configs=max_configs, 
            episodes=episodes
        )
        
        # Copy results to master directory
        import shutil
        shutil.copytree(tuner.experiment_dir, self.master_dir / "hyperparameter_tuning")
        
        print(f"✓ Hyperparameter optimization complete")
        print(f"  - Tested {len(results)} configurations")
        print(f"  - Results in: hyperparameter_tuning/")
        
        return results
    
    def _run_algorithm_comparison(self, episodes, num_seeds):
        """Phase 2: Algorithm comparison."""
        print("Comparing different RL algorithms...")
        
        comparison = AlgorithmComparison(self.config)
        results = comparison.run_comparison(episodes=episodes, num_runs=num_seeds)
        
        # Copy results to master directory
        import shutil
        shutil.copytree(comparison.experiment_dir, self.master_dir / "algorithm_comparison")
        
        algorithms_tested = list(results.keys())
        print(f"✓ Algorithm comparison complete")
        print(f"  - Algorithms tested: {', '.join(algorithms_tested)}")
        print(f"  - {num_seeds} runs per algorithm")
        print(f"  - Results in: algorithm_comparison/")
        
        return results
    
    def _run_ablation_studies(self, episodes):
        """Phase 3: Ablation studies."""
        print("Running ablation studies...")
        
        ablation = AblationStudy(self.config)
        results = ablation.run_all_ablation_studies(episodes=episodes)
        
        # Copy results to master directory
        import shutil
        shutil.copytree(ablation.experiment_dir, self.master_dir / "ablation_studies")
        
        studies_completed = list(results.keys())
        print(f"✓ Ablation studies complete")
        print(f"  - Studies: {', '.join(studies_completed)}")
        print(f"  - Results in: ablation_studies/")
        
        return results
    
    def _run_comprehensive_evaluation(self, episodes, num_seeds):
        """Phase 4: Comprehensive evaluation."""
        print("Running comprehensive evaluation with RL metrics...")
        
        # Define agents for evaluation
        agent_configs = {
            'Random': RandomAgent,
            'Heuristic': HeuristicAgent,
            'DQN': DQNAgent,
            'Double DQN': DoubleDQNAgent,
            'Dueling DQN': DuelingDQNAgent
            # TODO: Add PPO when available
        }
        
        evaluator = ComprehensiveEvaluator(self.config)
        comparison_results, statistical_analysis = evaluator.compare_multiple_agents(
            agent_configs, episodes=episodes, num_seeds=num_seeds
        )
        
        # Copy results to master directory
        import shutil
        shutil.copytree(evaluator.experiment_dir, self.master_dir / "comprehensive_evaluation")
        
        print(f"✓ Comprehensive evaluation complete")
        print(f"  - Agents evaluated: {', '.join(agent_configs.keys())}")
        print(f"  - Advanced RL metrics computed")
        print(f"  - Statistical significance tested")
        print(f"  - Results in: comprehensive_evaluation/")
        
        return {
            'comparison_results': comparison_results,
            'statistical_analysis': statistical_analysis
        }
    
    def _generate_final_analysis(self):
        """Phase 5: Generate final analysis and recommendations."""
        print("Generating final analysis and recommendations...")
        
        # Collect key findings from all phases
        findings = self._extract_key_findings()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings)
        
        # Create final report
        self._create_final_report(findings, recommendations)
        
        print(f"✓ Final analysis complete")
        print(f"  - Key findings extracted")
        print(f"  - Recommendations generated")
        print(f"  - Final report created: final_analysis_report.md")
    
    def _extract_key_findings(self):
        """Extract key findings from all experimental phases."""
        findings = {
            'hyperparameter_optimization': {},
            'algorithm_comparison': {},
            'ablation_studies': {},
            'comprehensive_evaluation': {}
        }
        
        # Hyperparameter findings
        if self.results['hyperparameter_tuning']:
            hp_results = self.results['hyperparameter_tuning']
            if hp_results:
                best_config = max(hp_results, key=lambda x: x['mean_reward'])
                findings['hyperparameter_optimization'] = {
                    'best_configuration': best_config['config'],
                    'best_performance': best_config['mean_reward'],
                    'most_important_hyperparameters': self._identify_important_hyperparameters(hp_results)
                }
        
        # Algorithm comparison findings
        if self.results['algorithm_comparison']:
            alg_results = self.results['algorithm_comparison']
            findings['algorithm_comparison'] = {
                'best_algorithm': self._find_best_algorithm(alg_results),
                'performance_ranking': self._rank_algorithms(alg_results),
                'key_tradeoffs': self._identify_algorithm_tradeoffs(alg_results)
            }
        
        # Ablation study findings
        if self.results['ablation_studies']:
            ablation_results = self.results['ablation_studies']
            findings['ablation_studies'] = {
                'most_important_components': self._identify_important_components(ablation_results),
                'component_impacts': self._calculate_component_impacts(ablation_results)
            }
        
        # Comprehensive evaluation findings
        if self.results['comprehensive_evaluation']:
            eval_results = self.results['comprehensive_evaluation']
            findings['comprehensive_evaluation'] = {
                'statistical_significance': self._summarize_statistical_tests(eval_results),
                'performance_metrics': self._summarize_performance_metrics(eval_results)
            }
        
        return findings
    
    def _identify_important_hyperparameters(self, hp_results):
        """Identify most important hyperparameters from tuning results."""
        # Simple heuristic: parameters with highest variance in performance
        hyperparams = ['lr', 'batch_size', 'target_update_freq', 'epsilon_decay']
        importance = {}
        
        for param in hyperparams:
            param_performance = {}
            for result in hp_results:
                param_value = result['config'][param]
                if param_value not in param_performance:
                    param_performance[param_value] = []
                param_performance[param_value].append(result['mean_reward'])
            
            # Calculate variance across different values
            if len(param_performance) > 1:
                param_means = [np.mean(values) for values in param_performance.values()]
                importance[param] = np.var(param_means)
            else:
                importance[param] = 0
        
        # Sort by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_params[:3]  # Top 3 most important
    
    def _find_best_algorithm(self, alg_results):
        """Find best performing algorithm."""
        best_algorithm = None
        best_performance = float('-inf')
        
        for alg_name, runs in alg_results.items():
            avg_performance = np.mean([run['mean_reward'] for run in runs])
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_algorithm = alg_name
        
        return {
            'name': best_algorithm,
            'performance': best_performance
        }
    
    def _rank_algorithms(self, alg_results):
        """Rank algorithms by performance."""
        rankings = []
        
        for alg_name, runs in alg_results.items():
            avg_performance = np.mean([run['mean_reward'] for run in runs])
            avg_success_rate = np.mean([run['success_rate'] for run in runs])
            
            rankings.append({
                'algorithm': alg_name,
                'mean_reward': avg_performance,
                'success_rate': avg_success_rate
            })
        
        # Sort by mean reward
        rankings.sort(key=lambda x: x['mean_reward'], reverse=True)
        return rankings
    
    def _identify_algorithm_tradeoffs(self, alg_results):
        """Identify key tradeoffs between algorithms."""
        tradeoffs = []
        
        # Example: Learning speed vs final performance
        if 'DQN' in alg_results and 'Heuristic' in alg_results:
            dqn_perf = np.mean([run['mean_reward'] for run in alg_results['DQN']])
            heuristic_perf = np.mean([run['mean_reward'] for run in alg_results['Heuristic']])
            
            if heuristic_perf > dqn_perf:
                tradeoffs.append("Heuristic outperforms DQN, suggesting learning challenge")
            else:
                tradeoffs.append("DQN successfully learned to outperform heuristic")
        
        return tradeoffs
    
    def _identify_important_components(self, ablation_results):
        """Identify most important components from ablation studies."""
        important_components = {}
        
        for study_name, study_results in ablation_results.items():
            # Find component with largest performance impact
            performance_gaps = {}
            performances = [result['mean_reward'] for result in study_results.values()]
            
            if performances:
                max_perf = max(performances)
                min_perf = min(performances)
                performance_gaps[study_name] = max_perf - min_perf
        
        return performance_gaps
    
    def _calculate_component_impacts(self, ablation_results):
        """Calculate impact of each component."""
        impacts = {}
        
        for study_name, study_results in ablation_results.items():
            study_impacts = {}
            baseline = None
            
            # Find baseline (usually 'Standard' or 'Full' variant)
            for variant_name, results in study_results.items():
                if 'standard' in variant_name.lower() or 'full' in variant_name.lower():
                    baseline = results['mean_reward']
                    break
            
            # If no baseline found, use best performance
            if baseline is None:
                baseline = max(result['mean_reward'] for result in study_results.values())
            
            # Calculate impact relative to baseline
            for variant_name, results in study_results.items():
                impact = (results['mean_reward'] - baseline) / abs(baseline) if baseline != 0 else 0
                study_impacts[variant_name] = impact
            
            impacts[study_name] = study_impacts
        
        return impacts
    
    def _summarize_statistical_tests(self, eval_results):
        """Summarize statistical significance tests."""
        if 'statistical_analysis' not in eval_results:
            return {}
        
        stat_analysis = eval_results['statistical_analysis']
        summary = {}
        
        for metric, analysis in stat_analysis.items():
            significant_pairs = []
            if 'pairwise_tests' in analysis:
                for pair, test in analysis['pairwise_tests'].items():
                    if test['significant']:
                        significant_pairs.append(pair)
            
            summary[metric] = {
                'significant_differences': len(significant_pairs) > 0,
                'significant_pairs': significant_pairs
            }
        
        return summary
    
    def _summarize_performance_metrics(self, eval_results):
        """Summarize performance metrics."""
        if 'comparison_results' not in eval_results:
            return {}
        
        comparison_results = eval_results['comparison_results']
        summary = {}
        
        key_metrics = ['mean_reward', 'success_rate', 'sample_efficiency', 'asymptotic_performance']
        
        for metric in key_metrics:
            metric_summary = {}
            for agent_name, results in comparison_results.items():
                if metric in results:
                    metric_summary[agent_name] = {
                        'mean': results[metric]['mean'],
                        'std': results[metric]['std']
                    }
            summary[metric] = metric_summary
        
        return summary
    
    def _generate_recommendations(self, findings):
        """Generate actionable recommendations based on findings."""
        recommendations = {
            'best_approach': None,
            'hyperparameter_settings': None,
            'when_to_use_each_algorithm': {},
            'future_improvements': []
        }
        
        # Best overall approach
        if findings['algorithm_comparison'].get('best_algorithm'):
            best_alg = findings['algorithm_comparison']['best_algorithm']
            recommendations['best_approach'] = f"Use {best_alg['name']} for best performance ({best_alg['performance']:.2f} reward)"
        
        # Best hyperparameters
        if findings['hyperparameter_optimization'].get('best_configuration'):
            recommendations['hyperparameter_settings'] = findings['hyperparameter_optimization']['best_configuration']
        
        # When to use each algorithm
        if findings['algorithm_comparison'].get('performance_ranking'):
            rankings = findings['algorithm_comparison']['performance_ranking']
            
            recommendations['when_to_use_each_algorithm'] = {
                'Quick deployment': 'Heuristic - immediate good performance',
                'Best final performance': f"{rankings[0]['algorithm']} - highest reward",
                'Research/learning': 'DQN variants - for understanding RL'
            }
        
        # Future improvements
        recommendations['future_improvements'] = [
            "Implement PPO for policy-based comparison",
            "Test on more complex environments",
            "Add curriculum learning",
            "Investigate state representation improvements"
        ]
        
        return recommendations
    
    def _create_final_report(self, findings, recommendations):
        """Create comprehensive final report."""
        report_lines = []
        
        # Header
        report_lines.extend([
            "# Assignment 2 - Comprehensive Experimental Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Grid:** {self.config['grid_path']}",
            f"**Starting Position:** {self.config['agent_start_pos']}",
            "",
            "## Executive Summary",
            ""
        ])
        
        # Best approach
        if recommendations['best_approach']:
            report_lines.append(f"**Recommended Approach:** {recommendations['best_approach']}")
            report_lines.append("")
        
        # Key findings
        report_lines.extend([
            "## Key Findings",
            "",
            "### Algorithm Performance Ranking"
        ])
        
        if findings['algorithm_comparison'].get('performance_ranking'):
            rankings = findings['algorithm_comparison']['performance_ranking']
            for i, alg in enumerate(rankings, 1):
                report_lines.append(f"{i}. **{alg['algorithm']}**: {alg['mean_reward']:.2f} reward, {alg['success_rate']:.1%} success")
        
        report_lines.append("")
        
        # Hyperparameter insights
        if findings['hyperparameter_optimization'].get('most_important_hyperparameters'):
            report_lines.extend([
                "### Most Important Hyperparameters",
                ""
            ])
            
            for param, importance in findings['hyperparameter_optimization']['most_important_hyperparameters']:
                report_lines.append(f"- **{param}**: Impact score {importance:.2f}")
            
            report_lines.append("")
        
        # Component analysis
        if findings['ablation_studies'].get('most_important_components'):
            report_lines.extend([
                "### Component Importance (Ablation Studies)",
                ""
            ])
            
            for component, impact in findings['ablation_studies']['most_important_components'].items():
                report_lines.append(f"- **{component.replace('_', ' ').title()}**: {impact:.2f} reward impact")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### Best Configuration"
        ])
        
        if recommendations['hyperparameter_settings']:
            for param, value in recommendations['hyperparameter_settings'].items():
                report_lines.append(f"- {param}: {value}")
        
        report_lines.extend([
            "",
            "### When to Use Each Algorithm"
        ])
        
        for use_case, recommendation in recommendations['when_to_use_each_algorithm'].items():
            report_lines.append(f"- **{use_case}**: {recommendation}")
        
        # Future work
        report_lines.extend([
            "",
            "## Future Improvements",
            ""
        ])
        
        for improvement in recommendations['future_improvements']:
            report_lines.append(f"- {improvement}")
        
        # Technical details
        report_lines.extend([
            "",
            "## Technical Details",
            "",
            "### Experimental Setup",
            f"- Episodes per experiment: Varied by phase",
            f"- Random seeds: Multiple runs for statistical validity",
            f"- Environment: 8D realistic continuous state space",
            f"- Evaluation metrics: Sample efficiency, asymptotic performance, stability",
            "",
            "### Files Generated",
            "- `hyperparameter_tuning/`: Hyperparameter optimization results",
            "- `algorithm_comparison/`: Multi-algorithm comparison",
            "- `ablation_studies/`: Component importance analysis", 
            "- `comprehensive_evaluation/`: Advanced RL metrics and statistical tests",
            "- `final_analysis_report.md`: This comprehensive report",
            "",
            "## Statistical Significance",
            ""
        ])
        
        # Add statistical significance summary
        if findings['comprehensive_evaluation'].get('statistical_significance'):
            stat_summary = findings['comprehensive_evaluation']['statistical_significance']
            
            for metric, significance in stat_summary.items():
                if significance['significant_differences']:
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}**: Significant differences detected")
                    for pair in significance['significant_pairs']:
                        report_lines.append(f"  - {pair.replace('_', ' vs ')}")
                else:
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}**: No significant differences")
        
        report_lines.extend([
            "",
            "---",
            "*Generated by Assignment 2 Master Experiment Runner*"
        ])
        
        # Save the report
        with open(self.master_dir / "final_analysis_report.md", 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also save findings and recommendations as JSON
        with open(self.master_dir / "findings_and_recommendations.json", 'w') as f:
            json.dump({
                'findings': findings,
                'recommendations': recommendations
            }, f, indent=2, default=str)


def main():
    """Main function to run experiments with command line arguments."""
    parser = argparse.ArgumentParser(description='Run comprehensive RL experiments for Assignment 2')
    
    parser.add_argument('--grid', type=str, default='grid_configs/A1_grid.npy',
                       help='Grid file to use for experiments')
    parser.add_argument('--start_pos', type=int, nargs=2, default=[3, 11],
                       help='Agent starting position (x y)')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Environment stochasticity')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (reduced parameters for testing)')
    parser.add_argument('--phase', type=str, choices=['all', 'hyperparams', 'algorithms', 'ablation', 'evaluation'],
                       default='all', help='Which experimental phase to run')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'grid_path': Path(args.grid),
        'agent_start_pos': tuple(args.start_pos),
        'sigma': args.sigma,
        'random_seed': args.seed
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create experiment runner
    runner = MasterExperimentRunner(config)
    
    # Run experiments based on phase selection
    if args.phase == 'all':
        results = runner.run_complete_experimental_suite(quick_mode=args.quick)
    elif args.phase == 'hyperparams':
        episodes = 50 if args.quick else 100
        max_configs = 10 if args.quick else 20
        results = runner._run_hyperparameter_optimization(episodes, max_configs)
    elif args.phase == 'algorithms':
        episodes = 50 if args.quick else 100
        num_seeds = 3 if args.quick else 5
        results = runner._run_algorithm_comparison(episodes, num_seeds)
    elif args.phase == 'ablation':
        episodes = 50 if args.quick else 100
        results = runner._run_ablation_studies(episodes)
    elif args.phase == 'evaluation':
        episodes = 50 if args.quick else 100
        num_seeds = 3 if args.quick else 5
        results = runner._run_comprehensive_evaluation(episodes, num_seeds)
    
    print(f"\nExperiments complete! Results saved to: {runner.master_dir}")
    
    return results


if __name__ == "__main__":
    import numpy as np
    results = main()