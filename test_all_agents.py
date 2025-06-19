#!/usr/bin/env python3
"""
Comprehensive Agent Testing Script for Assignment 2
Tests all implemented agents on Assignment 2 specific grids with detailed analysis.

Usage:
python test_all_agents.py [--quick] [--agents AGENT1,AGENT2] [--grids GRID1,GRID2] [--episodes N]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import sys
from tqdm import tqdm

# Handle imports like other scripts
try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.DQN_agent import DQNAgent
    from logger import Logger
except ModuleNotFoundError:
    from os import path
    from os import pardir
    
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    
    if root_path not in sys.path:
        sys.path.append(root_path)
    
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.DQN_agent import DQNAgent
    from logger import Logger


class AgentTester:
    """Comprehensive testing framework for all agents on Assignment 2 grids."""
    
    def __init__(self, output_dir="test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Assignment 2 specific grids
        self.a2_grids = {
            'A1_grid': 'grid_configs/A1_grid.npy',
            'open_space': 'grid_configs/A2/open_space.npy',
            'simple_restaurant': 'grid_configs/A2/simple_restaurant.npy',
            'corridor_test': 'grid_configs/A2/corridor_test.npy',
            'maze_challenge': 'grid_configs/A2/maze_challenge.npy',
            'assignment2_main': 'grid_configs/A2/assignment2_main.npy'
        }
        
        # Available agents
        self.agents = {
            'random': RandomAgent,
            'heuristic': HeuristicAgent,
            'dqn': DQNAgent
        }
        
        # Recommended starting positions for each grid
        self.start_positions = {
            'A1_grid': (3, 11),
            'open_space': (2, 2),
            'simple_restaurant': (2, 8),
            'corridor_test': (2, 4), 
            'maze_challenge': (2, 6),
            'assignment2_main': (3, 9)
        }
        
        self.results = {}
        
    def create_grids_if_missing(self):
        """Create Assignment 2 grids if they don't exist."""
        try:
            from world.create_restaurant_grids import (
                create_open_space, create_simple_restaurant, 
                create_corridor_test, create_maze_challenge, create_assignment_grid
            )
            
            # Ensure A2 directory exists
            Path("grid_configs/A2").mkdir(parents=True, exist_ok=True)
            
            grid_creators = {
                'open_space': create_open_space,
                'simple_restaurant': create_simple_restaurant,
                'corridor_test': create_corridor_test,
                'maze_challenge': create_maze_challenge,
                'assignment2_main': create_assignment_grid
            }
            
            missing_grids = []
            for grid_name, grid_path in self.a2_grids.items():
                if not Path(grid_path).exists() and grid_name in grid_creators:
                    print(f"Creating missing grid: {grid_name}")
                    grid_creators[grid_name]()
                    missing_grids.append(grid_name)
                    
            if missing_grids:
                print(f"✓ Created {len(missing_grids)} missing grids: {', '.join(missing_grids)}")
                
        except ImportError:
            print("⚠️  Could not import grid creators. Make sure grid files exist.")
    
    def test_single_agent_grid(self, agent_name, agent_class, grid_name, grid_path, 
                              episodes=50, max_steps=1000, num_seeds=3):
        """Test single agent on single grid with multiple random seeds."""
        print(f"  Testing {agent_name} on {grid_name}...")
        
        all_results = []
        
        for seed in range(num_seeds):
            # Create agent
            if agent_name in ['random', 'heuristic']:
                agent = agent_class()
            elif agent_name == 'dqn':
                agent = agent_class(state_dim=8, action_dim=4, verbose=False)
            else:
                agent = agent_class()
            
            # Create environment
            env = Environment(
                grid_fp=Path(grid_path),
                no_gui=True,
                sigma=0.1,
                agent_start_pos=self.start_positions.get(grid_name, None),
                random_seed=42 + seed,
                state_representation='continuous_vector'
            )
            
            # Run episodes
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            
            for episode in range(episodes):
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(max_steps):
                    # Select action based on agent type
                    if hasattr(agent, 'take_training_action') and agent_name == 'dqn':
                        action = agent.take_training_action(state, training=True)
                    else:
                        action = agent.take_action(state)
                    
                    # Execute action
                    next_state, reward, terminated, info = env.step(action)
                    
                    # Update agent if it supports learning
                    if hasattr(agent, 'update'):
                        agent.update(state, reward, action, next_state, terminated)
                    
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    if terminated:
                        success_count += 1
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Final evaluation with greedy policy
            final_eval_reward = self._evaluate_agent_greedy(env, agent, max_steps)
            
            # Calculate metrics for this seed
            seed_results = {
                'seed': seed,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'success_rate': success_count / episodes,
                'mean_length': np.mean(episode_lengths),
                'final_evaluation': final_eval_reward,
                'best_episode': np.max(episode_rewards),
                'worst_episode': np.min(episode_rewards),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths
            }
            
            all_results.append(seed_results)
        
        # Aggregate results across seeds
        aggregated = self._aggregate_seed_results(all_results)
        aggregated['agent'] = agent_name
        aggregated['grid'] = grid_name
        aggregated['episodes'] = episodes
        aggregated['num_seeds'] = num_seeds
        
        return aggregated
    
    def _evaluate_agent_greedy(self, env, agent, max_steps):
        """Evaluate agent with greedy policy (no exploration)."""
        state = env.reset()
        total_reward = 0
        
        for _ in range(max_steps):
            action = agent.take_action(state)  # Greedy action
            state, reward, terminated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                break
                
        return total_reward
    
    def _aggregate_seed_results(self, seed_results):
        """Aggregate results across multiple seeds."""
        metrics = ['mean_reward', 'success_rate', 'mean_length', 'final_evaluation', 
                   'best_episode', 'worst_episode']
        
        aggregated = {}
        for metric in metrics:
            values = [result[metric] for result in seed_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Keep individual seed results for detailed analysis
        aggregated['seed_results'] = seed_results
        
        return aggregated
    
    def test_all_combinations(self, agent_names=None, grid_names=None, episodes=50, num_seeds=3):
        """Test all agent-grid combinations."""
        if agent_names is None:
            agent_names = list(self.agents.keys())
        if grid_names is None:
            grid_names = list(self.a2_grids.keys())
        
        # Filter available grids
        available_grids = {}
        for name in grid_names:
            if name in self.a2_grids and Path(self.a2_grids[name]).exists():
                available_grids[name] = self.a2_grids[name]
            else:
                print(f"⚠️  Grid {name} not found at {self.a2_grids.get(name, 'unknown path')}")
        
        if not available_grids:
            print("❌ No valid grids found!")
            return
        
        print(f"Testing {len(agent_names)} agents on {len(available_grids)} grids...")
        print(f"Agents: {', '.join(agent_names)}")
        print(f"Grids: {', '.join(available_grids.keys())}")
        print(f"Episodes per test: {episodes}, Seeds per test: {num_seeds}")
        print()
        
        # Test all combinations
        for agent_name in agent_names:
            if agent_name not in self.agents:
                print(f"⚠️  Unknown agent: {agent_name}")
                continue
                
            print(f"Testing {agent_name.upper()} agent:")
            agent_class = self.agents[agent_name]
            
            for grid_name, grid_path in available_grids.items():
                try:
                    result = self.test_single_agent_grid(
                        agent_name, agent_class, grid_name, grid_path, 
                        episodes, max_steps=1000, num_seeds=num_seeds
                    )
                    
                    # Store results
                    if agent_name not in self.results:
                        self.results[agent_name] = {}
                    self.results[agent_name][grid_name] = result
                    
                    # Print summary
                    mean_reward = result['mean_reward']['mean']
                    success_rate = result['success_rate']['mean']
                    print(f"    {grid_name}: {mean_reward:.1f} ± {result['mean_reward']['std']:.1f} reward, "
                          f"{success_rate:.1%} success")
                    
                except Exception as e:
                    print(f"    ❌ {grid_name}: Error - {e}")
                    continue
            
            print()
    
    def create_comparison_plots(self):
        """Create comprehensive comparison plots."""
        if not self.results:
            print("No results to plot!")
            return
        
        # Extract data for plotting
        agents = list(self.results.keys())
        grids = list(next(iter(self.results.values())).keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Assignment 2 - Agent Performance Comparison', fontsize=16)
        
        # Plot 1: Mean Rewards by Grid
        ax = axes[0, 0]
        grid_positions = np.arange(len(grids))
        bar_width = 0.25
        
        for i, agent in enumerate(agents):
            rewards = [self.results[agent][grid]['mean_reward']['mean'] for grid in grids]
            reward_stds = [self.results[agent][grid]['mean_reward']['std'] for grid in grids]
            
            ax.bar(grid_positions + i * bar_width, rewards, bar_width, 
                   label=agent.title(), yerr=reward_stds, capsize=3)
        
        ax.set_xlabel('Grid')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Mean Rewards by Grid')
        ax.set_xticks(grid_positions + bar_width)
        ax.set_xticklabels(grids, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Success Rates by Grid
        ax = axes[0, 1]
        for i, agent in enumerate(agents):
            success_rates = [self.results[agent][grid]['success_rate']['mean'] for grid in grids]
            success_stds = [self.results[agent][grid]['success_rate']['std'] for grid in grids]
            
            ax.bar(grid_positions + i * bar_width, success_rates, bar_width,
                   label=agent.title(), yerr=success_stds, capsize=3)
        
        ax.set_xlabel('Grid')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rates by Grid')
        ax.set_xticks(grid_positions + bar_width)
        ax.set_xticklabels(grids, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Episode Lengths by Grid
        ax = axes[0, 2]
        for i, agent in enumerate(agents):
            lengths = [self.results[agent][grid]['mean_length']['mean'] for grid in grids]
            length_stds = [self.results[agent][grid]['mean_length']['std'] for grid in grids]
            
            ax.bar(grid_positions + i * bar_width, lengths, bar_width,
                   label=agent.title(), yerr=length_stds, capsize=3)
        
        ax.set_xlabel('Grid')
        ax.set_ylabel('Mean Episode Length')
        ax.set_title('Episode Lengths by Grid')
        ax.set_xticks(grid_positions + bar_width)
        ax.set_xticklabels(grids, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Agent Performance Summary (Radar-style)
        ax = axes[1, 0]
        agent_positions = np.arange(len(agents))
        
        # Average across all grids
        avg_rewards = []
        avg_success = []
        for agent in agents:
            rewards = [self.results[agent][grid]['mean_reward']['mean'] for grid in grids]
            success = [self.results[agent][grid]['success_rate']['mean'] for grid in grids]
            avg_rewards.append(np.mean(rewards))
            avg_success.append(np.mean(success))
        
        ax.bar(agent_positions, avg_rewards, alpha=0.7, label='Avg Reward (scaled)')
        ax2 = ax.twinx()
        ax2.bar(agent_positions, avg_success, alpha=0.7, color='orange', label='Avg Success Rate')
        
        ax.set_xlabel('Agent')
        ax.set_ylabel('Average Reward', color='blue')
        ax2.set_ylabel('Average Success Rate', color='orange')
        ax.set_title('Overall Agent Performance')
        ax.set_xticks(agent_positions)
        ax.set_xticklabels([a.title() for a in agents])
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Grid Difficulty Analysis
        ax = axes[1, 1]
        
        # Use heuristic agent as baseline to assess grid difficulty
        if 'heuristic' in self.results:
            baseline_rewards = [self.results['heuristic'][grid]['mean_reward']['mean'] for grid in grids]
            baseline_success = [self.results['heuristic'][grid]['success_rate']['mean'] for grid in grids]
            
            # Sort grids by difficulty (lower reward = harder)
            grid_difficulty = list(zip(grids, baseline_rewards, baseline_success))
            grid_difficulty.sort(key=lambda x: x[1])  # Sort by reward
            
            sorted_grids = [x[0] for x in grid_difficulty]
            sorted_rewards = [x[1] for x in grid_difficulty]
            sorted_success = [x[2] for x in grid_difficulty]
            
            x_pos = np.arange(len(sorted_grids))
            ax.bar(x_pos, sorted_rewards, alpha=0.7, label='Heuristic Reward')
            ax2 = ax.twinx()
            ax2.plot(x_pos, sorted_success, 'ro-', label='Success Rate')
            
            ax.set_xlabel('Grid (sorted by difficulty)')
            ax.set_ylabel('Heuristic Agent Reward', color='blue')
            ax2.set_ylabel('Success Rate', color='red')
            ax.set_title('Grid Difficulty Analysis')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sorted_grids, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Final Evaluation vs Training Performance
        ax = axes[1, 2]
        
        for agent in agents:
            training_perfs = []
            final_evals = []
            
            for grid in grids:
                training_perfs.append(self.results[agent][grid]['mean_reward']['mean'])
                final_evals.append(self.results[agent][grid]['final_evaluation']['mean'])
            
            ax.scatter(training_perfs, final_evals, label=agent.title(), alpha=0.7, s=50)
        
        # Add diagonal line (perfect correlation)
        min_val = min(min(training_perfs), min(final_evals))
        max_val = max(max(training_perfs), max(final_evals))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
        
        ax.set_xlabel('Training Performance (Mean Reward)')
        ax.set_ylabel('Final Evaluation Performance')
        ax.set_title('Training vs Evaluation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.output_dir / f"agent_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plots saved to: {plot_path}")
    
    def save_results(self):
        """Save detailed results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_path = self.output_dir / f"test_results_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        with open(json_path, 'w') as f:
            json.dump(convert_for_json(self.results), f, indent=2)
        
        print(f"💾 Detailed results saved to: {json_path}")
        
        # Create summary report
        self._create_summary_report(timestamp)
    
    def _create_summary_report(self, timestamp):
        """Create human-readable summary report."""
        report_path = self.output_dir / f"summary_report_{timestamp}.md"
        
        lines = [
            "# Assignment 2 - Agent Testing Report",
            f"",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Grids Tested:** {len(next(iter(self.results.values())).keys())}",
            f"**Agents Tested:** {len(self.results)}",
            f"",
            "## Executive Summary",
            ""
        ]
        
        # Overall best performers
        if self.results:
            # Calculate overall scores
            agent_scores = {}
            for agent_name, agent_results in self.results.items():
                total_reward = 0
                total_success = 0
                grid_count = 0
                
                for grid_name, grid_results in agent_results.items():
                    total_reward += grid_results['mean_reward']['mean']
                    total_success += grid_results['success_rate']['mean']
                    grid_count += 1
                
                avg_reward = total_reward / grid_count
                avg_success = total_success / grid_count
                
                agent_scores[agent_name] = {
                    'avg_reward': avg_reward,
                    'avg_success': avg_success,
                    'composite_score': avg_reward + (avg_success * 100)  # Weighted composite
                }
            
            # Best agent overall
            best_agent = max(agent_scores.items(), key=lambda x: x[1]['composite_score'])
            lines.extend([
                f"**Best Overall Agent:** {best_agent[0].title()}",
                f"- Average Reward: {best_agent[1]['avg_reward']:.2f}",
                f"- Average Success Rate: {best_agent[1]['avg_success']:.1%}",
                ""
            ])
        
        # Per-agent summaries
        lines.append("## Agent Performance Summary")
        lines.append("")
        
        for agent_name, agent_results in self.results.items():
            lines.append(f"### {agent_name.upper()} Agent")
            lines.append("")
            
            for grid_name, grid_results in agent_results.items():
                reward = grid_results['mean_reward']['mean']
                reward_std = grid_results['mean_reward']['std']
                success = grid_results['success_rate']['mean']
                
                lines.append(f"- **{grid_name}**: {reward:.1f} ± {reward_std:.1f} reward, {success:.1%} success")
            
            lines.append("")
        
        # Grid difficulty ranking
        if 'heuristic' in self.results:
            lines.extend([
                "## Grid Difficulty Ranking",
                "*Based on heuristic agent performance*",
                ""
            ])
            
            heuristic_results = self.results['heuristic']
            grid_difficulty = [(name, results['mean_reward']['mean']) 
                              for name, results in heuristic_results.items()]
            grid_difficulty.sort(key=lambda x: x[1])  # Sort by reward (ascending = harder)
            
            for i, (grid_name, reward) in enumerate(grid_difficulty, 1):
                difficulty = "Hard" if i <= 2 else "Medium" if i <= 4 else "Easy"
                lines.append(f"{i}. **{grid_name}** ({difficulty}): {reward:.1f} reward")
            
            lines.append("")
        
        # Recommendations
        lines.extend([
            "## Recommendations",
            "",
            "### For Development:",
            "- Test new algorithms on `open_space` first (easiest grid)",
            "- Use `simple_restaurant` for intermediate testing",
            "- Validate on `maze_challenge` for robustness",
            "",
            "### For Evaluation:",
            "- Use `A1_grid` as the primary benchmark",
            "- Include `assignment2_main` for comprehensive assessment",
            "",
            "### Agent-Specific Notes:"
        ])
        
        if 'random' in self.results:
            lines.append("- **Random Agent**: Useful as baseline, expect ~10-20% success rate")
        if 'heuristic' in self.results:
            lines.append("- **Heuristic Agent**: Good immediate performance, 70-90% success rate")
        if 'dqn' in self.results:
            lines.append("- **DQN Agent**: Requires training, performance varies by starting position")
        
        lines.extend([
            "",
            "---",
            "*Generated by Assignment 2 Agent Testing Framework*"
        ])
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"📝 Summary report saved to: {report_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test all agents on Assignment 2 grids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_all_agents.py                                    # Test all agents on all grids
  python test_all_agents.py --quick                           # Quick test (fewer episodes/seeds)
  python test_all_agents.py --agents random,heuristic         # Test specific agents
  python test_all_agents.py --grids A1_grid,open_space        # Test specific grids
  python test_all_agents.py --episodes 100 --seeds 5          # Custom episode/seed counts
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (20 episodes, 2 seeds)')
    parser.add_argument('--agents', type=str, default='random,heuristic,dqn',
                       help='Comma-separated list of agents to test')
    parser.add_argument('--grids', type=str, default=None,
                       help='Comma-separated list of grids to test (default: all)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes per test (default: 50)')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds per test (default: 3)')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def main():
    """Main testing function."""
    args = parse_args()
    
    # Adjust parameters for quick mode
    if args.quick:
        episodes = 20
        seeds = 2
        print("🚀 Quick test mode enabled")
    else:
        episodes = args.episodes
        seeds = args.seeds
    
    # Parse agent and grid lists
    agent_names = [name.strip() for name in args.agents.split(',')]
    
    if args.grids:
        grid_names = [name.strip() for name in args.grids.split(',')]
    else:
        grid_names = None  # Use all grids
    
    print("🧪 Assignment 2 - Comprehensive Agent Testing")
    print("=" * 50)
    print(f"Episodes per test: {episodes}")
    print(f"Seeds per test: {seeds}")
    print(f"Agents: {', '.join(agent_names)}")
    print(f"Output directory: {args.output}")
    print()
    
    # Create tester
    tester = AgentTester(args.output)
    
    # Create grids if missing
    print("📁 Checking for Assignment 2 grids...")
    tester.create_grids_if_missing()
    print()
    
    # Run tests
    print("🔬 Running agent tests...")
    tester.test_all_combinations(
        agent_names=agent_names,
        grid_names=grid_names,
        episodes=episodes,
        num_seeds=seeds
    )
    
    # Save results
    print("💾 Saving results...")
    tester.save_results()
    
    # Create plots
    if not args.no_plots:
        print("📊 Creating comparison plots...")
        tester.create_comparison_plots()
    
    print()
    print("✅ Testing complete!")
    print(f"📂 Results saved to: {tester.output_dir}")
    
    # Print quick summary
    if tester.results:
        print("\n🏆 Quick Summary:")
        for agent_name, agent_results in tester.results.items():
            avg_reward = np.mean([r['mean_reward']['mean'] for r in agent_results.values()])
            avg_success = np.mean([r['success_rate']['mean'] for r in agent_results.values()])
            print(f"  {agent_name.title()}: {avg_reward:.1f} avg reward, {avg_success:.1%} avg success")


if __name__ == "__main__":
    main()