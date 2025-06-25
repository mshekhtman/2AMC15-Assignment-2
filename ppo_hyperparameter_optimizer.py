#!/usr/bin/env python3
"""
PPO Hyperparameter Optimization Script for Assignment 2
Systematically analyzes and optimizes PPO hyperparameters for each Assignment 2 grid.

This script:
1. Analyzes the current PPO implementation
2. Defines comprehensive hyperparameter search spaces
3. Tests different configurations on each grid
4. Finds optimal hyperparameters per grid
5. Provides detailed analysis and recommendations

Usage:
python ppo_hyperparameter_optimizer.py [--quick] [--grids GRID1,GRID2] [--trials N]
"""

import argparse
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import torch
from dataclasses import dataclass

# Import our modules with proper error handling
try:
    from world.environment import Environment
    from agents.PPO_agent import PPOAgent
    from logger import Logger
except ModuleNotFoundError as e:
    print(f"Import error: {e}")
    print("Trying alternative import method...")
    import sys
    from pathlib import Path
    
    # Add current directory and parent to path
    current_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(current_dir))
    
    try:
        from world.environment import Environment
        from agents.PPO_agent import PPOAgent
        from logger import Logger
        print("✓ Successfully imported modules")
    except ImportError as e2:
        print(f"❌ Failed to import modules: {e2}")
        print("Please ensure you're running from the project root directory")
        print("and that all required files exist:")
        print("  - world/environment.py")
        print("  - agents/PPO_agent.py") 
        print("  - logger.py")
        sys.exit(1)


@dataclass
class HyperparameterConfig:
    """PPO hyperparameter configuration."""
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    entropy_coef: float
    value_coef: float
    rollout_steps: int
    ppo_epochs: int
    batch_size: int
    hidden_dim: int


@dataclass
class ExperimentResult:
    """Results from a single hyperparameter experiment."""
    config: HyperparameterConfig
    grid_name: str
    mean_reward: float
    std_reward: float
    success_rate: float
    convergence_episode: int
    final_reward: float
    stability_score: float
    sample_efficiency: float
    episode_rewards: List[float]


class PPOHyperparameterOptimizer:
    """Comprehensive PPO hyperparameter optimization for Assignment 2 grids."""
    
    def __init__(self, output_dir: str = "ppo_optimization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Assignment 2 grids with difficulty estimates
        self.grids = {
            'open_space': {
                'path': 'grid_configs/A2/open_space.npy',
                'difficulty': 'easy',
                'start_pos': (2, 2),
                'description': 'Simple open environment with minimal obstacles'
            },
            'simple_restaurant': {
                'path': 'grid_configs/A2/simple_restaurant.npy', 
                'difficulty': 'medium',
                'start_pos': (2, 8),
                'description': 'Basic restaurant layout with tables and kitchen'
            },
            'corridor_test': {
                'path': 'grid_configs/A2/corridor_test.npy',
                'difficulty': 'medium',
                'start_pos': (2, 4),
                'description': 'Narrow corridors testing navigation skills'
            },
            'A1_grid': {
                'path': 'grid_configs/A1_grid.npy',
                'difficulty': 'medium-hard',
                'start_pos': (3, 11),
                'description': 'Main assignment grid with balanced complexity'
            },
            'assignment2_main': {
                'path': 'grid_configs/A2/assignment2_main.npy',
                'difficulty': 'hard',
                'start_pos': (3, 9),
                'description': 'Complex restaurant with multiple service areas'
            },
            'maze_challenge': {
                'path': 'grid_configs/A2/maze_challenge.npy',
                'difficulty': 'very_hard',
                'start_pos': (2, 6),
                'description': 'Complex maze requiring advanced pathfinding'
            }
        }
        
        self.results = []
        
    def analyze_current_implementation(self):
        """Analyze the current PPO implementation to understand its characteristics."""
        print("=== ANALYZING CURRENT PPO IMPLEMENTATION ===")
        
        # Create a test agent to analyze default hyperparameters
        test_agent = PPOAgent(verbose=False)
        
        current_config = {
            'lr': test_agent.optimizer.param_groups[0]['lr'],
            'gamma': test_agent.gamma,
            'gae_lambda': test_agent.lmbda,
            'clip_eps': test_agent.clip_eps,
            'entropy_coef': test_agent.entropy_coef,
            'value_coef': test_agent.value_coef,
            'rollout_steps': test_agent.rollout_steps,
            'ppo_epochs': test_agent.ppo_epochs,
            'batch_size': test_agent.batch_size,
            'hidden_dim': 128  # From network architecture
        }
        
        analysis = {
            'current_hyperparameters': current_config,
            'implementation_features': [
                'Reward normalization with RunningMeanStd',
                'Adaptive hyperparameters based on performance',
                'GAE for advantage estimation',
                'Clipped surrogate objective',
                'Value function clipping',
                'Gradient clipping',
                'Learning rate scheduling'
            ],
            'potential_improvements': [
                'Learning rate may be too high for stable learning',
                'Entropy coefficient might need adjustment per grid difficulty',
                'Rollout steps could be optimized per grid complexity',
                'Batch size affects sample efficiency vs stability trade-off'
            ]
        }
        
        print(f"Current Learning Rate: {current_config['lr']}")
        print(f"Current Entropy Coefficient: {current_config['entropy_coef']}")
        print(f"Current Rollout Steps: {current_config['rollout_steps']}")
        print(f"Current PPO Epochs: {current_config['ppo_epochs']}")
        print()
        
        # Save analysis
        with open(self.output_dir / "implementation_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
    
    def define_search_spaces(self, search_type: str = 'comprehensive'):
        """Define hyperparameter search spaces based on RL literature and grid characteristics."""
        
        if search_type == 'quick':
            # Quick search for testing
            return {
                'lr': [1e-3, 2e-3],
                'gamma': [0.99],
                'gae_lambda': [0.95],
                'clip_eps': [0.2, 0.25],
                'entropy_coef': [0.01, 0.03],
                'value_coef': [0.5, 1.0],
                'rollout_steps': [32, 64],
                'ppo_epochs': [4, 8],
                'batch_size': [16, 32],
                'hidden_dim': [128]
            }
        
        elif search_type == 'comprehensive':
            # Comprehensive search based on PPO literature
            return {
                'lr': [5e-4, 1e-3, 2e-3, 3e-3, 5e-3],  # Learning rate range
                'gamma': [0.95, 0.99, 0.995],  # Discount factor
                'gae_lambda': [0.9, 0.95, 0.98],  # GAE lambda
                'clip_eps': [0.1, 0.2, 0.25, 0.3],  # PPO clipping
                'entropy_coef': [0.005, 0.01, 0.03, 0.05, 0.1],  # Exploration
                'value_coef': [0.25, 0.5, 1.0, 1.5],  # Value learning weight
                'rollout_steps': [16, 32, 64, 128],  # Episode length
                'ppo_epochs': [3, 4, 6, 8, 10],  # Training epochs
                'batch_size': [8, 16, 32, 64],  # Batch size
                'hidden_dim': [64, 128, 256]  # Network size
            }
        
        else:  # 'grid_specific'
            # Grid-specific search focusing on key parameters
            return {
                'lr': [1e-3, 2e-3, 3e-3],
                'gamma': [0.99],  # Keep standard
                'gae_lambda': [0.95],  # Keep standard
                'clip_eps': [0.2, 0.25, 0.3],
                'entropy_coef': [0.01, 0.03, 0.05, 0.1],  # Varies by grid difficulty
                'value_coef': [0.5, 1.0],
                'rollout_steps': [32, 64, 128],  # Varies by grid size
                'ppo_epochs': [4, 6, 8],
                'batch_size': [16, 32],
                'hidden_dim': [128, 256]  # Varies by grid complexity
            }
    
    def generate_configurations(self, search_space: Dict, max_configs: int = 50, 
                               strategy: str = 'random') -> List[HyperparameterConfig]:
        """Generate hyperparameter configurations to test."""
        
        if strategy == 'grid':
            # Full grid search (can be very large)
            configs = list(itertools.product(*search_space.values()))
            keys = list(search_space.keys())
            configurations = [
                HyperparameterConfig(**dict(zip(keys, config))) 
                for config in configs
            ]
        
        elif strategy == 'random':
            # Random search - more efficient for high-dimensional spaces
            configurations = []
            keys = list(search_space.keys())
            
            for _ in range(max_configs):
                config_dict = {}
                for key in keys:
                    config_dict[key] = np.random.choice(search_space[key])
                configurations.append(HyperparameterConfig(**config_dict))
        
        elif strategy == 'smart':
            # Smart sampling based on known good combinations
            configurations = []
            
            # Start with some known good baselines
            baseline_configs = [
                # Conservative PPO
                {'lr': 1e-3, 'entropy_coef': 0.01, 'rollout_steps': 64, 'ppo_epochs': 4},
                # Aggressive exploration
                {'lr': 2e-3, 'entropy_coef': 0.05, 'rollout_steps': 32, 'ppo_epochs': 8},
                # Large rollouts
                {'lr': 5e-4, 'entropy_coef': 0.03, 'rollout_steps': 128, 'ppo_epochs': 6},
            ]
            
            # Add baseline configurations
            for base in baseline_configs:
                config_dict = {
                    'lr': base.get('lr', 2e-3),
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_eps': 0.25,
                    'entropy_coef': base.get('entropy_coef', 0.03),
                    'value_coef': 1.0,
                    'rollout_steps': base.get('rollout_steps', 64),
                    'ppo_epochs': base.get('ppo_epochs', 8),
                    'batch_size': 32,
                    'hidden_dim': 128
                }
                configurations.append(HyperparameterConfig(**config_dict))
            
            # Fill remaining with random search
            remaining = max_configs - len(configurations)
            if remaining > 0:
                random_configs = self.generate_configurations(
                    search_space, remaining, 'random'
                )
                configurations.extend(random_configs)
        
        # Limit to max_configs
        return configurations[:max_configs]
    
    def run_single_experiment(self, config: HyperparameterConfig, grid_name: str, 
                             episodes: int = 100) -> ExperimentResult:
        """Run a single hyperparameter experiment."""
        
        grid_info = self.grids[grid_name]
        
        # Create environment
        env = Environment(
            grid_fp=Path(grid_info['path']),
            no_gui=True,
            sigma=0.1,
            agent_start_pos=grid_info['start_pos'],
            random_seed=42,
            state_representation='continuous_vector'
        )
        
        # Create PPO agent with specific hyperparameters
        try:
            agent = PPOAgent(
                state_dim=8,
                action_dim=4,
                lr=config.lr,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_eps=config.clip_eps,
                entropy_coef=config.entropy_coef,
                value_coef=config.value_coef,
                rollout_steps=config.rollout_steps,
                ppo_epochs=config.ppo_epochs,
                batch_size=config.batch_size,
                hidden_dim=config.hidden_dim,
                verbose=False
            )
        except Exception as e:
            print(f"⚠️  Error creating PPO agent: {e}")
            print("Trying with default parameters...")
            
            # Fallback: create agent with defaults and manually set key parameters
            agent = PPOAgent(state_dim=8, action_dim=4, verbose=False)
            
            # Manually update key hyperparameters that we can change
            agent.gamma = config.gamma
            agent.lmbda = config.gae_lambda
            agent.clip_eps = config.clip_eps
            agent.entropy_coef = config.entropy_coef
            agent.value_coef = config.value_coef
            agent.rollout_steps = config.rollout_steps
            agent.ppo_epochs = config.ppo_epochs
            agent.batch_size = config.batch_size
            
            # Update optimizer learning rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = config.lr
        
        # Training loop
        episode_rewards = []
        success_count = 0
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)
                agent.update(state, reward, action, next_state, terminated)
                
                episode_reward += reward
                state = next_state
                
                if terminated:
                    success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = success_count / episodes
        final_reward = np.mean(episode_rewards[-10:])  # Last 10 episodes
        
        # Calculate convergence (when reward stabilizes)
        convergence_episode = self._find_convergence(episode_rewards)
        
        # Calculate stability (lower variance = more stable)
        stability_score = 1.0 / (1.0 + std_reward)
        
        # Calculate sample efficiency (how quickly it reaches good performance)
        target_reward = mean_reward * 0.8  # 80% of final performance
        sample_efficiency = episodes  # Default if never reached
        for i, reward in enumerate(episode_rewards):
            if reward >= target_reward:
                sample_efficiency = i + 1
                break
        
        return ExperimentResult(
            config=config,
            grid_name=grid_name,
            mean_reward=mean_reward,
            std_reward=std_reward,
            success_rate=success_rate,
            convergence_episode=convergence_episode,
            final_reward=final_reward,
            stability_score=stability_score,
            sample_efficiency=sample_efficiency,
            episode_rewards=episode_rewards
        )
    
    def _find_convergence(self, rewards: List[float], window: int = 20) -> int:
        """Find when the reward curve converges (stabilizes)."""
        if len(rewards) < 2 * window:
            return len(rewards)
        
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        
        for i in range(window, len(moving_avg) - window):
            # Check if variance in window is low (converged)
            window_std = moving_avg[i:i+window].std()
            if window_std < (np.std(rewards) * 0.1):  # 10% of total variance
                return i
        
        return len(rewards)
    
    def optimize_grid(self, grid_name: str, search_type: str = 'comprehensive', 
                     max_configs: int = 30, episodes: int = 80) -> List[ExperimentResult]:
        """Optimize hyperparameters for a specific grid."""
        
        print(f"\n=== OPTIMIZING HYPERPARAMETERS FOR {grid_name.upper()} ===")
        print(f"Grid difficulty: {self.grids[grid_name]['difficulty']}")
        print(f"Description: {self.grids[grid_name]['description']}")
        
        # Check if grid exists
        grid_path = Path(self.grids[grid_name]['path'])
        if not grid_path.exists():
            print(f"⚠️  Grid {grid_name} not found at {grid_path}")
            return []
        
        # Generate configurations
        search_space = self.define_search_spaces(search_type)
        
        # Adjust search strategy based on grid difficulty
        if self.grids[grid_name]['difficulty'] in ['easy', 'medium']:
            strategy = 'smart'  # Focus on good known configurations
        else:
            strategy = 'random'  # More exploration for hard grids
        
        configurations = self.generate_configurations(search_space, max_configs, strategy)
        
        print(f"Testing {len(configurations)} configurations...")
        
        # Run experiments
        grid_results = []
        
        for i, config in enumerate(tqdm(configurations, desc=f"Testing {grid_name}")):
            try:
                result = self.run_single_experiment(config, grid_name, episodes)
                grid_results.append(result)
                
                # Progress update
                if (i + 1) % 5 == 0:
                    best_so_far = max(grid_results, key=lambda x: x.mean_reward)
                    print(f"  Progress: {i+1}/{len(configurations)} - "
                          f"Best reward so far: {best_so_far.mean_reward:.1f}")
                    
            except Exception as e:
                print(f"  ⚠️  Config {i+1} failed: {e}")
                continue
        
        # Sort results by performance
        grid_results.sort(key=lambda x: x.mean_reward, reverse=True)
        
        # Print top 3 results
        print(f"\n📊 Top 3 configurations for {grid_name}:")
        for i, result in enumerate(grid_results[:3]):
            print(f"{i+1}. Reward: {result.mean_reward:.1f} ± {result.std_reward:.1f}, "
                  f"Success: {result.success_rate:.1%}")
            print(f"   LR: {result.config.lr}, Entropy: {result.config.entropy_coef}, "
                  f"Rollout: {result.config.rollout_steps}")
        
        return grid_results
    
    def optimize_all_grids(self, search_type: str = 'comprehensive', 
                          max_configs: int = 30, episodes: int = 80, 
                          selected_grids: List[str] = None) -> Dict[str, List[ExperimentResult]]:
        """Optimize hyperparameters for all grids."""
        
        print("🚀 PPO HYPERPARAMETER OPTIMIZATION FOR ASSIGNMENT 2")
        print("=" * 60)
        
        # Analyze current implementation first
        self.analyze_current_implementation()
        
        # Determine which grids to test
        grids_to_test = selected_grids if selected_grids else list(self.grids.keys())
        
        # Filter existing grids
        available_grids = []
        for grid_name in grids_to_test:
            if Path(self.grids[grid_name]['path']).exists():
                available_grids.append(grid_name)
            else:
                print(f"⚠️  Skipping {grid_name} - file not found")
        
        print(f"Testing {len(available_grids)} grids: {', '.join(available_grids)}")
        print(f"Search type: {search_type}")
        print(f"Max configurations per grid: {max_configs}")
        print(f"Episodes per experiment: {episodes}")
        print()
        
        # Optimize each grid
        all_results = {}
        
        for grid_name in available_grids:
            grid_results = self.optimize_grid(grid_name, search_type, max_configs, episodes)
            all_results[grid_name] = grid_results
            self.results.extend(grid_results)
        
        # Create comprehensive analysis
        self._create_optimization_analysis(all_results)
        
        return all_results
    
    def _create_optimization_analysis(self, all_results: Dict[str, List[ExperimentResult]]):
        """Create comprehensive analysis of optimization results."""
        
        print("\n=== CREATING COMPREHENSIVE ANALYSIS ===")
        
        # 1. Save raw results
        self._save_raw_results(all_results)
        
        # 2. Create performance summary
        self._create_performance_summary(all_results)
        
        # 3. Create hyperparameter analysis
        self._create_hyperparameter_analysis()
        
        # 4. Create visualizations
        self._create_optimization_plots(all_results)
        
        # 5. Generate recommendations
        self._generate_recommendations(all_results)
        
        print(f"📁 All results saved to: {self.output_dir}")
    
    def _save_raw_results(self, all_results: Dict[str, List[ExperimentResult]]):
        """Save raw experimental results with proper JSON serialization."""
        
        # Helper function to convert numpy types to Python types
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to JSON serializable types."""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif obj is None:
                return None
            elif isinstance(obj, str):
                return obj
            else:
                # For any other type, try to convert to string as fallback
                try:
                    return str(obj)
                except:
                    return None
        
        # Convert to JSON-serializable format
        serializable_results = {}
        
        for grid_name, results in all_results.items():
            serializable_results[grid_name] = []
            
            for result in results:
                result_dict = {
                    'config': {
                        'lr': float(result.config.lr),
                        'gamma': float(result.config.gamma),
                        'gae_lambda': float(result.config.gae_lambda),
                        'clip_eps': float(result.config.clip_eps),
                        'entropy_coef': float(result.config.entropy_coef),
                        'value_coef': float(result.config.value_coef),
                        'rollout_steps': int(result.config.rollout_steps),
                        'ppo_epochs': int(result.config.ppo_epochs),
                        'batch_size': int(result.config.batch_size),
                        'hidden_dim': int(result.config.hidden_dim)
                    },
                    'metrics': {
                        'mean_reward': float(result.mean_reward),
                        'std_reward': float(result.std_reward),
                        'success_rate': float(result.success_rate),
                        'convergence_episode': int(result.convergence_episode),
                        'final_reward': float(result.final_reward),
                        'stability_score': float(result.stability_score),
                        'sample_efficiency': int(result.sample_efficiency)
                    },
                    'episode_rewards': [float(r) for r in result.episode_rewards]
                }
                serializable_results[grid_name].append(result_dict)
        
        # Apply comprehensive conversion as backup
        serializable_results = convert_to_json_serializable(serializable_results)
        
        # Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f"ppo_optimization_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✓ Raw results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️  Error saving raw results: {e}")
            # Try saving a simplified version
            simplified_results = {}
            for grid_name, results in all_results.items():
                simplified_results[grid_name] = {
                    'count': len(results),
                    'best_reward': float(max(r.mean_reward for r in results)) if results else 0,
                    'worst_reward': float(min(r.mean_reward for r in results)) if results else 0
                }
            
            simplified_file = self.output_dir / f"ppo_optimization_summary_{timestamp}.json"
            with open(simplified_file, 'w') as f:
                json.dump(simplified_results, f, indent=2)
            print(f"✓ Simplified results saved to: {simplified_file}")
    
    def _create_performance_summary(self, all_results: Dict[str, List[ExperimentResult]]):
        """Create performance summary table."""
        
        summary_data = []
        
        for grid_name, results in all_results.items():
            if not results:
                continue
                
            best_result = max(results, key=lambda x: x.mean_reward)
            worst_result = min(results, key=lambda x: x.mean_reward)
            
            summary_data.append({
                'Grid': grid_name,
                'Difficulty': self.grids[grid_name]['difficulty'],
                'Best_Reward': best_result.mean_reward,
                'Best_Success_Rate': best_result.success_rate,
                'Best_Config_LR': best_result.config.lr,
                'Best_Config_Entropy': best_result.config.entropy_coef,
                'Best_Config_Rollout': best_result.config.rollout_steps,
                'Worst_Reward': worst_result.mean_reward,
                'Performance_Range': best_result.mean_reward - worst_result.mean_reward,
                'Configs_Tested': len(results)
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Best_Reward', ascending=False)
        
        # Save CSV
        df.to_csv(self.output_dir / "performance_summary.csv", index=False)
        
        # Print summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
    
    def _create_hyperparameter_analysis(self):
        """Analyze hyperparameter importance across all experiments."""
        
        if not self.results:
            return
        
        # Create DataFrame for analysis
        data = []
        for result in self.results:
            data.append({
                'grid': result.grid_name,
                'mean_reward': result.mean_reward,
                'lr': result.config.lr,
                'gamma': result.config.gamma,
                'gae_lambda': result.config.gae_lambda,
                'clip_eps': result.config.clip_eps,
                'entropy_coef': result.config.entropy_coef,
                'value_coef': result.config.value_coef,
                'rollout_steps': result.config.rollout_steps,
                'ppo_epochs': result.config.ppo_epochs,
                'batch_size': result.config.batch_size,
                'hidden_dim': result.config.hidden_dim,
                'success_rate': result.success_rate,
                'stability_score': result.stability_score
            })
        
        df = pd.DataFrame(data)
        
        # Hyperparameter importance analysis
        hyperparams = ['lr', 'entropy_coef', 'rollout_steps', 'ppo_epochs', 
                      'batch_size', 'clip_eps', 'value_coef']
        
        importance_analysis = {}
        
        for param in hyperparams:
            # Group by parameter value and calculate mean performance
            grouped = df.groupby(param)['mean_reward'].agg(['mean', 'std', 'count'])
            
            # Calculate variance across different values (importance indicator)
            importance_score = grouped['mean'].var()
            
            importance_analysis[param] = {
                'importance_score': importance_score,
                'value_impact': grouped.to_dict()
            }
        
        # Sort by importance
        sorted_importance = sorted(importance_analysis.items(), 
                                 key=lambda x: x[1]['importance_score'], 
                                 reverse=True)
        
        print(f"\n🔍 HYPERPARAMETER IMPORTANCE ANALYSIS")
        print("=" * 50)
        for param, analysis in sorted_importance:
            print(f"{param}: {analysis['importance_score']:.2f}")
        
        # Save analysis
        with open(self.output_dir / "hyperparameter_importance.json", 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_analysis = {}
            for param, analysis in importance_analysis.items():
                serializable_analysis[param] = {
                    'importance_score': float(analysis['importance_score']),
                    'value_impact': {str(k): {str(k2): float(v2) if isinstance(v2, (int, float)) else v2 
                                            for k2, v2 in v.items()} 
                                   for k, v in analysis['value_impact'].items()}
                }
            json.dump(serializable_analysis, f, indent=2)
    
    def _create_optimization_plots(self, all_results: Dict[str, List[ExperimentResult]]):
        """Create comprehensive visualization plots."""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create multiple plots
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Performance by Grid
        ax1 = plt.subplot(3, 3, 1)
        grid_names = []
        best_rewards = []
        worst_rewards = []
        
        for grid_name, results in all_results.items():
            if results:
                grid_names.append(grid_name)
                best_rewards.append(max(r.mean_reward for r in results))
                worst_rewards.append(min(r.mean_reward for r in results))
        
        x = range(len(grid_names))
        ax1.bar(x, best_rewards, alpha=0.7, label='Best Config')
        ax1.bar(x, worst_rewards, alpha=0.7, label='Worst Config')
        ax1.set_xticks(x)
        ax1.set_xticklabels(grid_names, rotation=45, ha='right')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Performance Range by Grid')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate Impact
        ax2 = plt.subplot(3, 3, 2)
        if self.results:
            lr_data = {}
            for result in self.results:
                lr = result.config.lr
                if lr not in lr_data:
                    lr_data[lr] = []
                lr_data[lr].append(result.mean_reward)
            
            lrs = sorted(lr_data.keys())
            means = [np.mean(lr_data[lr]) for lr in lrs]
            stds = [np.std(lr_data[lr]) for lr in lrs]
            
            ax2.errorbar(lrs, means, yerr=stds, marker='o', capsize=5)
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('Mean Reward')
            ax2.set_title('Learning Rate Impact')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Entropy Coefficient Impact
        ax3 = plt.subplot(3, 3, 3)
        if self.results:
            entropy_data = {}
            for result in self.results:
                entropy = result.config.entropy_coef
                if entropy not in entropy_data:
                    entropy_data[entropy] = []
                entropy_data[entropy].append(result.mean_reward)
            
            entropies = sorted(entropy_data.keys())
            means = [np.mean(entropy_data[e]) for e in entropies]
            stds = [np.std(entropy_data[e]) for e in entropies]
            
            ax3.errorbar(entropies, means, yerr=stds, marker='o', capsize=5)
            ax3.set_xlabel('Entropy Coefficient')
            ax3.set_ylabel('Mean Reward')
            ax3.set_title('Entropy Coefficient Impact')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rollout Steps Impact
        ax4 = plt.subplot(3, 3, 4)
        if self.results:
            rollout_data = {}
            for result in self.results:
                rollout = result.config.rollout_steps
                if rollout not in rollout_data:
                    rollout_data[rollout] = []
                rollout_data[rollout].append(result.mean_reward)
            
            rollouts = sorted(rollout_data.keys())
            means = [np.mean(rollout_data[r]) for r in rollouts]
            
            ax4.bar(range(len(rollouts)), means, alpha=0.7)
            ax4.set_xticks(range(len(rollouts)))
            ax4.set_xticklabels(rollouts)
            ax4.set_xlabel('Rollout Steps')
            ax4.set_ylabel('Mean Reward')
            ax4.set_title('Rollout Steps Impact')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Success Rate vs Reward
        ax5 = plt.subplot(3, 3, 5)
        if self.results:
            rewards = [r.mean_reward for r in self.results]
            success_rates = [r.success_rate for r in self.results]
            
            ax5.scatter(rewards, success_rates, alpha=0.6)
            ax5.set_xlabel('Mean Reward')
            ax5.set_ylabel('Success Rate')
            ax5.set_title('Success Rate vs Reward')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Stability vs Performance
        ax6 = plt.subplot(3, 3, 6)
        if self.results:
            rewards = [r.mean_reward for r in self.results]
            stability = [r.stability_score for r in self.results]
            
            ax6.scatter(rewards, stability, alpha=0.6)
            ax6.set_xlabel('Mean Reward')
            ax6.set_ylabel('Stability Score')
            ax6.set_title('Stability vs Performance')
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Sample Efficiency Analysis
        ax7 = plt.subplot(3, 3, 7)
        if self.results:
            grid_efficiency = {}
            for result in self.results:
                grid = result.grid_name
                if grid not in grid_efficiency:
                    grid_efficiency[grid] = []
                grid_efficiency[grid].append(result.sample_efficiency)
            
            grids = list(grid_efficiency.keys())
            efficiencies = [np.mean(grid_efficiency[g]) for g in grids]
            
            ax7.bar(range(len(grids)), efficiencies, alpha=0.7)
            ax7.set_xticks(range(len(grids)))
            ax7.set_xticklabels(grids, rotation=45, ha='right')
            ax7.set_ylabel('Episodes to Converge')
            ax7.set_title('Sample Efficiency by Grid')
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Hyperparameter Correlation Heatmap
        ax8 = plt.subplot(3, 3, 8)
        if self.results:
            # Create correlation matrix for hyperparameters
            hyperparam_data = []
            for result in self.results:
                hyperparam_data.append([
                    result.config.lr,
                    result.config.entropy_coef,
                    result.config.rollout_steps,
                    result.config.ppo_epochs,
                    result.config.batch_size,
                    result.mean_reward
                ])
            
            df_corr = pd.DataFrame(hyperparam_data, columns=[
                'LR', 'Entropy', 'Rollout', 'Epochs', 'Batch', 'Reward'
            ])
            
            correlation_matrix = df_corr.corr()
            im = ax8.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            
            # Add labels
            ax8.set_xticks(range(len(correlation_matrix.columns)))
            ax8.set_yticks(range(len(correlation_matrix.columns)))
            ax8.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax8.set_yticklabels(correlation_matrix.columns)
            ax8.set_title('Hyperparameter Correlation')
            
            # Add correlation values
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    text = ax8.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black")
        
        # Plot 9: Best Configuration per Grid
        ax9 = plt.subplot(3, 3, 9)
        best_configs_text = "BEST CONFIGURATIONS:\n\n"
        
        for grid_name, results in all_results.items():
            if results:
                best_result = max(results, key=lambda x: x.mean_reward)
                best_configs_text += f"{grid_name}:\n"
                best_configs_text += f"  Reward: {best_result.mean_reward:.1f}\n"
                best_configs_text += f"  LR: {best_result.config.lr}\n"
                best_configs_text += f"  Entropy: {best_result.config.entropy_coef}\n"
                best_configs_text += f"  Rollout: {best_result.config.rollout_steps}\n\n"
        
        ax9.text(0.05, 0.95, best_configs_text, transform=ax9.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=8)
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        ax9.set_title('Best Configurations Summary')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.output_dir / f"ppo_optimization_analysis_{timestamp}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Optimization plots saved")
    
    def _generate_recommendations(self, all_results: Dict[str, List[ExperimentResult]]):
        """Generate specific recommendations for each grid."""
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'grid_specific': {},
            'general_insights': []
        }
        
        # Overall summary
        if self.results:
            best_overall = max(self.results, key=lambda x: x.mean_reward)
            recommendations['summary'] = {
                'best_overall_grid': best_overall.grid_name,
                'best_overall_reward': best_overall.mean_reward,
                'total_experiments': len(self.results),
                'grids_tested': len(all_results)
            }
        
        # Grid-specific recommendations
        for grid_name, results in all_results.items():
            if not results:
                continue
                
            best_result = max(results, key=lambda x: x.mean_reward)
            grid_difficulty = self.grids[grid_name]['difficulty']
            
            # Determine optimal configuration
            optimal_config = {
                'lr': best_result.config.lr,
                'gamma': best_result.config.gamma,
                'gae_lambda': best_result.config.gae_lambda,
                'clip_eps': best_result.config.clip_eps,
                'entropy_coef': best_result.config.entropy_coef,
                'value_coef': best_result.config.value_coef,
                'rollout_steps': best_result.config.rollout_steps,
                'ppo_epochs': best_result.config.ppo_epochs,
                'batch_size': best_result.config.batch_size,
                'hidden_dim': best_result.config.hidden_dim
            }
            
            # Generate specific recommendations
            grid_recommendations = []
            
            # Learning rate recommendations
            if best_result.config.lr >= 0.003:
                grid_recommendations.append("High learning rate works well - agent can learn quickly")
            elif best_result.config.lr <= 0.001:
                grid_recommendations.append("Conservative learning rate needed - stable learning")
            
            # Entropy recommendations
            if best_result.config.entropy_coef >= 0.05:
                grid_recommendations.append("High exploration needed - complex navigation required")
            elif best_result.config.entropy_coef <= 0.01:
                grid_recommendations.append("Low exploration sufficient - straightforward environment")
            
            # Rollout recommendations
            if best_result.config.rollout_steps >= 128:
                grid_recommendations.append("Long rollouts beneficial - complex planning required")
            elif best_result.config.rollout_steps <= 32:
                grid_recommendations.append("Short rollouts sufficient - quick decisions work well")
            
            recommendations['grid_specific'][grid_name] = {
                'difficulty': grid_difficulty,
                'optimal_config': optimal_config,
                'expected_performance': {
                    'mean_reward': best_result.mean_reward,
                    'success_rate': best_result.success_rate,
                    'convergence_episodes': best_result.convergence_episode
                },
                'recommendations': grid_recommendations,
                'performance_range': max(r.mean_reward for r in results) - min(r.mean_reward for r in results)
            }
        
        # General insights
        if self.results:
            # Learning rate analysis
            lr_performance = {}
            for result in self.results:
                lr = result.config.lr
                if lr not in lr_performance:
                    lr_performance[lr] = []
                lr_performance[lr].append(result.mean_reward)
            
            best_lr = max(lr_performance.items(), key=lambda x: np.mean(x[1]))
            recommendations['general_insights'].append(
                f"Best learning rate overall: {best_lr[0]} (avg reward: {np.mean(best_lr[1]):.1f})"
            )
            
            # Entropy analysis
            entropy_performance = {}
            for result in self.results:
                entropy = result.config.entropy_coef
                if entropy not in entropy_performance:
                    entropy_performance[entropy] = []
                entropy_performance[entropy].append(result.mean_reward)
            
            best_entropy = max(entropy_performance.items(), key=lambda x: np.mean(x[1]))
            recommendations['general_insights'].append(
                f"Best entropy coefficient overall: {best_entropy[0]} (avg reward: {np.mean(best_entropy[1]):.1f})"
            )
            
            # Grid difficulty insights
            difficulty_performance = {}
            for result in self.results:
                difficulty = self.grids[result.grid_name]['difficulty']
                if difficulty not in difficulty_performance:
                    difficulty_performance[difficulty] = []
                difficulty_performance[difficulty].append(result.mean_reward)
            
            for difficulty, rewards in difficulty_performance.items():
                avg_reward = np.mean(rewards)
                recommendations['general_insights'].append(
                    f"{difficulty.title()} grids: average reward {avg_reward:.1f}"
                )
        
        # Save recommendations
        with open(self.output_dir / "optimization_recommendations.json", 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print recommendations
        print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)
        
        for grid_name, grid_rec in recommendations['grid_specific'].items():
            print(f"\n{grid_name.upper()} ({grid_rec['difficulty']}):")
            print(f"  Expected Performance: {grid_rec['expected_performance']['mean_reward']:.1f} reward")
            print(f"  Optimal LR: {grid_rec['optimal_config']['lr']}")
            print(f"  Optimal Entropy: {grid_rec['optimal_config']['entropy_coef']}")
            print(f"  Optimal Rollout: {grid_rec['optimal_config']['rollout_steps']}")
            
            for rec in grid_rec['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nGeneral Insights:")
        for insight in recommendations['general_insights']:
            print(f"  • {insight}")
        
        return recommendations
    
    def create_grid_grids_if_missing(self):
        """Create Assignment 2 grids if they don't exist."""
        try:
            # Try multiple import paths for grid creators
            try:
                from world.create_restaurant_grids import (
                    create_open_space, create_simple_restaurant, 
                    create_corridor_test, create_maze_challenge, create_assignment_grid
                )
            except ImportError:
                import sys
                sys.path.append('world')
                from create_restaurant_grids import (
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
            for grid_name, grid_info in self.grids.items():
                grid_path = Path(grid_info['path'])
                if not grid_path.exists() and grid_name in grid_creators:
                    print(f"Creating missing grid: {grid_name}")
                    try:
                        grid_creators[grid_name]()
                        missing_grids.append(grid_name)
                    except Exception as e:
                        print(f"  ⚠️  Failed to create {grid_name}: {e}")
                        continue
                    
            if missing_grids:
                print(f"✓ Created {len(missing_grids)} missing grids: {', '.join(missing_grids)}")
            else:
                print("✓ All grids already exist or checked")
                
        except ImportError as e:
            print(f"⚠️  Could not import grid creators: {e}")
            print("Will check for existing grid files instead...")
            
            # Check which grids exist
            existing_grids = []
            missing_grids = []
            
            for grid_name, grid_info in self.grids.items():
                grid_path = Path(grid_info['path'])
                if grid_path.exists():
                    existing_grids.append(grid_name)
                else:
                    missing_grids.append(grid_name)
            
            print(f"Existing grids: {', '.join(existing_grids) if existing_grids else 'None'}")
            if missing_grids:
                print(f"Missing grids: {', '.join(missing_grids)}")
                print("Please create these grids manually or ensure the grid creator script works.")
        except Exception as e:
            print(f"❌ Error during grid creation: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run PPO hyperparameter optimization."""
    
    parser = argparse.ArgumentParser(
        description='PPO Hyperparameter Optimization for Assignment 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ppo_hyperparameter_optimizer.py                           # Full optimization
  python ppo_hyperparameter_optimizer.py --quick                   # Quick test
  python ppo_hyperparameter_optimizer.py --grids A1_grid,open_space # Specific grids
  python ppo_hyperparameter_optimizer.py --trials 20 --episodes 60  # Custom settings
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick optimization mode (fewer configs and episodes)')
    parser.add_argument('--grids', type=str, default=None,
                       help='Comma-separated list of grids to optimize (default: all)')
    parser.add_argument('--trials', type=int, default=30,
                       help='Number of hyperparameter configurations to test per grid')
    parser.add_argument('--episodes', type=int, default=80,
                       help='Number of episodes per experiment')
    parser.add_argument('--search', type=str, default='comprehensive',
                       choices=['quick', 'comprehensive', 'grid_specific'],
                       help='Search strategy for hyperparameters')
    parser.add_argument('--output', type=str, default='ppo_optimization',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick mode
    if args.quick:
        trials = 10
        episodes = 40
        search_type = 'quick'
        print("🚀 Quick optimization mode enabled")
    else:
        trials = args.trials
        episodes = args.episodes
        search_type = args.search
    
    # Parse grid list
    selected_grids = None
    if args.grids:
        selected_grids = [g.strip() for g in args.grids.split(',')]
    
    print("🎯 PPO HYPERPARAMETER OPTIMIZATION FOR ASSIGNMENT 2")
    print("=" * 60)
    print(f"Search strategy: {search_type}")
    print(f"Trials per grid: {trials}")
    print(f"Episodes per trial: {episodes}")
    print(f"Output directory: {args.output}")
    if selected_grids:
        print(f"Selected grids: {', '.join(selected_grids)}")
    else:
        print("Grids: All available")
    print()
    
    # Create optimizer
    optimizer = PPOHyperparameterOptimizer(args.output)
    
    # Create grids if missing
    print("📁 Checking for Assignment 2 grids...")
    optimizer.create_grid_grids_if_missing()
    print()
    
    # Run optimization
    try:
        results = optimizer.optimize_all_grids(
            search_type=search_type,
            max_configs=trials,
            episodes=episodes,
            selected_grids=selected_grids
        )
        
        print("\n✅ OPTIMIZATION COMPLETE!")
        print(f"📁 Results saved to: {optimizer.output_dir}")
        
        # Print final summary
        if results:
            total_experiments = sum(len(grid_results) for grid_results in results.values())
            best_overall = None
            best_reward = float('-inf')
            
            for grid_name, grid_results in results.items():
                if grid_results:
                    best_grid = max(grid_results, key=lambda x: x.mean_reward)
                    if best_grid.mean_reward > best_reward:
                        best_reward = best_grid.mean_reward
                        best_overall = (grid_name, best_grid)
            
            print(f"\n📊 FINAL SUMMARY:")
            print(f"Total experiments: {total_experiments}")
            print(f"Grids optimized: {len(results)}")
            if best_overall:
                grid_name, best_result = best_overall
                print(f"Best performing grid: {grid_name}")
                print(f"Best reward achieved: {best_result.mean_reward:.1f}")
                print(f"Best configuration LR: {best_result.config.lr}")
                print(f"Best configuration Entropy: {best_result.config.entropy_coef}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()