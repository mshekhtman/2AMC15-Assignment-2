#!/usr/bin/env python3
"""
Simplified Agent Testing Script for Assignment 2
Tests all implemented agents on Assignment 2 grids with table summary only.
Removes all visualizations to prevent freezing during DQN training.

Usage:
python test_all_agents_simple.py [--quick] [--agents AGENT1,AGENT2] [--episodes N]
"""

import argparse
import numpy as np
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
    # Try to import PPO agent
    try:
        from agents.PPO_agent import PPOAgent
        PPO_AVAILABLE = True
    except ImportError:
        PPOAgent = None
        PPO_AVAILABLE = False
        print("⚠️  PPO agent not available - will skip PPO tests")
    # Try to import PPO agent
    try:
        from agents.PPO_agent import PPOAgent
        PPO_AVAILABLE = True
    except ImportError:
        PPOAgent = None
        PPO_AVAILABLE = False
        print("⚠️  PPO agent not available - will skip PPO tests")
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


class SimpleAgentTester:
    """Simplified testing framework focusing on core functionality."""
    
    def __init__(self, output_dir="test_results_simple"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Assignment 2 specific grids (no A1_grid)
        self.test_grids = {
            'open_space': 'grid_configs/A2/open_space.npy',
            'simple_restaurant': 'grid_configs/A2/simple_restaurant.npy',
            'corridor_test': 'grid_configs/A2/corridor_test.npy',
            'maze_challenge': 'grid_configs/A2/maze_challenge.npy',
            'assignment2_main': 'grid_configs/A2/assignment2_main.npy'
        }
        
        # Available agents including PPO
        self.agents = {
            #'random': RandomAgent,
            #'heuristic': HeuristicAgent,
            #'dqn': DQNAgent,
            'ppo': None  # Will be imported conditionally
        }
        
        # Starting positions for A2 grids
        self.start_positions = {
            'open_space': (2, 2),
            'simple_restaurant': (2, 8),
            'corridor_test': (2, 4), 
            'maze_challenge': (2, 6),
            'assignment2_main': (3, 9)
        }
        # Set PPO agent if available
        if PPO_AVAILABLE:
            self.agents['ppo'] = PPOAgent
        else:
            # Remove PPO from agents dict if not available
            if 'ppo' in self.agents:
                del self.agents['ppo']
        
        self.results = {}
        
    def create_grids_if_missing(self):
        """Create Assignment 2 grids if they don't exist."""
        try:
            from world.create_restaurant_grids import (
                create_open_space, create_simple_restaurant, 
                create_corridor_test, create_maze_challenge, create_assignment_grid
            )
            
            Path("grid_configs/A2").mkdir(parents=True, exist_ok=True)
            
            grid_creators = {
                'open_space': create_open_space,
                'simple_restaurant': create_simple_restaurant,
                'corridor_test': create_corridor_test,
                'maze_challenge': create_maze_challenge,
                'assignment2_main': create_assignment_grid
            }
            
            missing_grids = []
            for grid_name, grid_path in self.test_grids.items():
                if not Path(grid_path).exists() and grid_name in grid_creators:
                    print(f"Creating missing grid: {grid_name}")
                    grid_creators[grid_name]()
                    missing_grids.append(grid_name)
                    
            if missing_grids:
                print(f"✓ Created {len(missing_grids)} missing grids: {', '.join(missing_grids)}")
                
        except ImportError:
            print("⚠️  Could not import grid creators. Make sure grid files exist.")
    
    def test_single_agent_grid(self, agent_name, agent_class, grid_name, grid_path, episodes=None):
        """Test single agent on single grid with appropriate episode count per algorithm."""
        
        # Set appropriate episode count based on agent type if not specified
        if episodes is None:
            episode_counts = {
                'random': 20,      # Fast baseline - doesn't learn
                'heuristic': 10,   # Deterministic - fewer episodes needed
                'dqn': 100,        # Needs more episodes to learn effectively
                'ppo': 150         # Needs most episodes for policy optimization
            }
            episodes = episode_counts.get(agent_name, 30)
        
        print(f"  Testing {agent_name} on {grid_name} ({episodes} episodes)...")
        
        # Create agent with minimal verbosity
        if agent_name in ['random', 'heuristic']:
            agent = agent_class()
        elif agent_name in ['dqn', 'ppo']:
            agent = agent_class(state_dim=8, action_dim=4, verbose=False)
        else:
            agent = agent_class()
        
        # Create environment
        env = Environment(
            grid_fp=Path(grid_path),
            no_gui=True,
            sigma=0.1,
            agent_start_pos=self.start_positions.get(grid_name, None),
            random_seed=42,
            state_representation='continuous_vector'
        )
        
        # Run episodes with progress bar
        episode_rewards = []
        success_count = 0
        max_steps = 500  # Reduced from 1000 to speed up testing
        
        for episode in tqdm(range(episodes), desc=f"{agent_name}-{grid_name}", leave=False):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Select action based on agent type
                if hasattr(agent, 'take_training_action') and agent_name in ['dqn', 'ppo']:
                    action = agent.take_training_action(state, training=True)
                else:
                    action = agent.take_action(state)
                
                # Execute action
                next_state, reward, terminated, info = env.step(action)
                
                # Update agent if it supports learning
                if hasattr(agent, 'update'):
                    agent.update(state, reward, action, next_state, terminated)
                
                episode_reward += reward
                state = next_state
                
                if terminated:
                    success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
        
        # Calculate simple metrics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / episodes,
            'best_reward': np.max(episode_rewards),
            'worst_reward': np.min(episode_rewards),
            'episodes_run': episodes
        }
        
        return results
    
    def test_all_agents(self, agent_names=None, episodes=None):
        """Test all specified agents on all grids with appropriate episode counts."""
        if agent_names is None:
            agent_names = list(self.agents.keys())
        
        # Filter out unavailable agents
        available_agents = []
        for name in agent_names:
            if name in self.agents:
                available_agents.append(name)
            else:
                print(f"⚠️  Agent {name} not available - skipping")
        
        if not available_agents:
            print("❌ No valid agents to test!")
            return
        
        # Filter available grids
        available_grids = {}
        for name, path in self.test_grids.items():
            if Path(path).exists():
                available_grids[name] = path
            else:
                print(f"⚠️  Grid {name} not found at {path}")
        
        if not available_grids:
            print("❌ No valid grids found!")
            return
        
        print(f"Testing {len(available_agents)} agents on {len(available_grids)} grids")
        print(f"Available agents: {', '.join(available_agents)}")
        if episodes:
            print(f"Episodes per test: {episodes} (override)")
        else:
            print("Episodes per test: adaptive (random=20, heuristic=10, dqn=100, ppo=150)")
        print()
        
        # Test all combinations
        for agent_name in available_agents:
            print(f"Testing {agent_name.upper()} agent:")
            agent_class = self.agents[agent_name]
            
            self.results[agent_name] = {}
            
            for grid_name, grid_path in available_grids.items():
                try:
                    result = self.test_single_agent_grid(
                        agent_name, agent_class, grid_name, grid_path, episodes
                    )
                    
                    self.results[agent_name][grid_name] = result
                    
                    # Print quick summary with episode count
                    episodes_run = result['episodes_run']
                    print(f"    {grid_name}: {result['mean_reward']:.1f} ± {result['std_reward']:.1f} reward, "
                          f"{result['success_rate']:.1%} success ({episodes_run} episodes)")
                    
                except Exception as e:
                    print(f"    ❌ {grid_name}: Error - {e}")
                    self.results[agent_name][grid_name] = {
                        'mean_reward': -999,
                        'std_reward': 0,
                        'success_rate': 0,
                        'best_reward': -999,
                        'worst_reward': -999,
                        'episodes_run': episodes or 0,
                        'error': str(e)
                    }
            
            print()
    
    def print_results_table(self):
        """Print a clean table summary of all results."""
        if not self.results:
            print("No results to display!")
            return
        
        agents = list(self.results.keys())
        grids = list(next(iter(self.results.values())).keys())
        
        print("\n" + "="*80)
        print("ASSIGNMENT 2 - AGENT PERFORMANCE SUMMARY")
        print("="*80)
        
        # Rewards table
        print("\nMEAN REWARDS:")
        print("-" * 60)
        header = f"{'Agent':<12}"
        for grid in grids:
            header += f"{grid[:10]:<12}"
        print(header)
        print("-" * 60)
        
        for agent in agents:
            row = f"{agent:<12}"
            for grid in grids:
                reward = self.results[agent][grid]['mean_reward']
                if reward == -999:
                    row += f"{'ERROR':<12}"
                else:
                    row += f"{reward:>8.1f}    "
            print(row)
        
        # Success rates table
        print("\nSUCCESS RATES:")
        print("-" * 60)
        header = f"{'Agent':<12}"
        for grid in grids:
            header += f"{grid[:10]:<12}"
        print(header)
        print("-" * 60)
        
        for agent in agents:
            row = f"{agent:<12}"
            for grid in grids:
                success = self.results[agent][grid]['success_rate']
                if self.results[agent][grid]['mean_reward'] == -999:
                    row += f"{'ERROR':<12}"
                else:
                    row += f"{success:>8.1%}    "
            print(row)
        
        # Overall summary
        print("\nOVERALL PERFORMANCE:")
        print("-" * 40)
        
        for agent in agents:
            valid_rewards = []
            valid_success = []
            
            for grid in grids:
                if self.results[agent][grid]['mean_reward'] != -999:
                    valid_rewards.append(self.results[agent][grid]['mean_reward'])
                    valid_success.append(self.results[agent][grid]['success_rate'])
            
            if valid_rewards:
                avg_reward = np.mean(valid_rewards)
                avg_success = np.mean(valid_success)
                print(f"{agent.upper():<12}: {avg_reward:>8.1f} avg reward, {avg_success:>7.1%} avg success")
            else:
                print(f"{agent.upper():<12}: All tests failed")
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.output_dir / f"simple_results_{timestamp}.json"
        
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
        
        print(f"\n💾 Results saved to: {json_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Simple agent testing for Assignment 2 (no visualizations)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (15 episodes)')
    parser.add_argument('--agents', type=str, default='random,heuristic,dqn,ppo',
                       help='Comma-separated list of agents to test')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes per test (default: adaptive per agent)')
    parser.add_argument('--output', type=str, default='test_results_simple',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main testing function."""
    args = parse_args()
    
    # Adjust parameters for quick mode
    if args.quick:
        episodes = {'random': 10, 'heuristic': 5, 'dqn': 30, 'ppo': 40}
        print("🚀 Quick test mode enabled")
        print("Quick mode episodes: random=10, heuristic=5, dqn=30, ppo=40")
    else:
        episodes = args.episodes  # None for adaptive, or user-specified value
    
    # Parse agent list
    agent_names = [name.strip() for name in args.agents.split(',')]
    
    print("🧪 Assignment 2 - Simple Agent Testing")
    print("=" * 50)
    if isinstance(episodes, dict):
        print("Quick mode episodes:")
        for agent, count in episodes.items():
            print(f"  {agent}: {count} episodes")
    elif episodes:
        print(f"Episodes per test: {episodes} (fixed)")
    else:
        print("Episodes per test: adaptive (random=20, heuristic=10, dqn=100, ppo=150)")
    print(f"Agents: {', '.join(agent_names)}")
    print(f"Output directory: {args.output}")
    print()
    
    # Create tester
    tester = SimpleAgentTester(args.output)
    
    # Create grids if missing
    print("📁 Checking for grids...")
    tester.create_grids_if_missing()
    print()
    
    # Run tests
    print("🔬 Running simplified agent tests...")
    if isinstance(episodes, dict):
        # Quick mode with per-agent episode counts
        for agent_name in agent_names:
            if agent_name in episodes:
                single_agent_episodes = episodes[agent_name]
                tester.test_all_agents(agent_names=[agent_name], episodes=single_agent_episodes)
    else:
        # Normal mode
        tester.test_all_agents(agent_names=agent_names, episodes=episodes)
    
    # Print results table
    tester.print_results_table()
    
    # Save results
    tester.save_results()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()