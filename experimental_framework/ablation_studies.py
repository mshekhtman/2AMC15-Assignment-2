"""
Ablation Studies Framework for Assignment 2.
Studies the impact of different components on DQN performance.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

try:
    from agents.DQN_agent import DQNAgent
    from agents.DQN_nn import DQNetwork
    from world import Environment
    from logger import Logger
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from agents.DQN_agent import DQNAgent
    from agents.DQN_nn import DQNetwork
    from world import Environment
    from logger import Logger


class AblationStudy:
    """Framework for conducting ablation studies on DQN components."""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}
        self.experiment_dir = Path(f"experiments/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def study_state_representation(self, episodes=100):
        """Ablation study on state representation components."""
        print("=== State Representation Ablation Study ===")
        
        # Test different state representations
        state_variants = {
            'Full_8D': 'full',           # All 8 features
            'No_Position': 'no_pos',     # Remove position info
            'No_Clearance': 'no_clear',  # Remove clearance info  
            'No_Mission': 'no_mission',  # Remove mission info
            'Position_Only': 'pos_only', # Only position
            'Local_Only': 'local_only'   # Only clearance + mission
        }
        
        results = {}
        for variant_name, variant_type in state_variants.items():
            print(f"\nTesting state variant: {variant_name}")
            
            # Create custom environment with modified state representation
            env = self._create_modified_environment(variant_type)
            
            # Determine state dimension for this variant
            state_dim = self._get_state_dim(variant_type)
            
            # Create agent with appropriate state dimension
            agent = DQNAgent(state_dim=state_dim, action_dim=4)
            
            # Run experiment
            variant_results = self._run_ablation_experiment(env, agent, episodes, variant_name)
            results[variant_name] = variant_results
        
        self.results['state_representation'] = results
        self._save_ablation_results('state_representation', results)
        return results
    
    def study_network_architecture(self, episodes=100):
        """Ablation study on network architecture choices."""
        print("=== Network Architecture Ablation Study ===")
        
        architectures = {
            'Small_Net': {'hidden_dim': 64, 'num_layers': 2},
            'Medium_Net': {'hidden_dim': 128, 'num_layers': 3}, 
            'Large_Net': {'hidden_dim': 256, 'num_layers': 3},
            'Deep_Net': {'hidden_dim': 128, 'num_layers': 4},
            'Wide_Net': {'hidden_dim': 512, 'num_layers': 2}
        }
        
        results = {}
        for arch_name, arch_config in architectures.items():
            print(f"\nTesting architecture: {arch_name}")
            
            # Create custom agent with specific architecture
            agent = self._create_custom_architecture_agent(arch_config)
            
            # Standard environment
            env = Environment(
                grid_fp=self.base_config['grid_path'],
                no_gui=True,
                sigma=self.base_config['sigma'],
                agent_start_pos=self.base_config['agent_start_pos'],
                random_seed=self.base_config['random_seed'],
                state_representation='continuous_vector'
            )
            
            # Run experiment
            arch_results = self._run_ablation_experiment(env, agent, episodes, arch_name)
            results[arch_name] = arch_results
        
        self.results['network_architecture'] = results
        self._save_ablation_results('network_architecture', results)
        return results
    
    def study_training_components(self, episodes=100):
        """Ablation study on training components."""
        print("=== Training Components Ablation Study ===")
        
        training_variants = {
            'Standard_DQN': {'target_network': True, 'replay_buffer': True, 'epsilon_decay': True},
            'No_Target_Network': {'target_network': False, 'replay_buffer': True, 'epsilon_decay': True},
            'No_Replay_Buffer': {'target_network': True, 'replay_buffer': False, 'epsilon_decay': True},
            'No_Epsilon_Decay': {'target_network': True, 'replay_buffer': True, 'epsilon_decay': False},
            'Minimal_DQN': {'target_network': False, 'replay_buffer': False, 'epsilon_decay': False}
        }
        
        results = {}
        for variant_name, variant_config in training_variants.items():
            print(f"\nTesting training variant: {variant_name}")
            
            # Create agent with specific training configuration
            agent = self._create_training_variant_agent(variant_config)
            
            # Standard environment
            env = Environment(
                grid_fp=self.base_config['grid_path'],
                no_gui=True,
                sigma=self.base_config['sigma'],
                agent_start_pos=self.base_config['agent_start_pos'],
                random_seed=self.base_config['random_seed'],
                state_representation='continuous_vector'
            )
            
            # Run experiment
            variant_results = self._run_ablation_experiment(env, agent, episodes, variant_name)
            results[variant_name] = variant_results
        
        self.results['training_components'] = results
        self._save_ablation_results('training_components', results)
        return results
    
    def study_exploration_strategies(self, episodes=100):
        """Ablation study on exploration strategies."""
        print("=== Exploration Strategies Ablation Study ===")
        
        exploration_strategies = {
            'Epsilon_Greedy': {'type': 'epsilon_greedy', 'epsilon_start': 1.0, 'epsilon_min': 0.01, 'epsilon_decay': 0.995},
            'High_Exploration': {'type': 'epsilon_greedy', 'epsilon_start': 1.0, 'epsilon_min': 0.1, 'epsilon_decay': 0.99},
            'Low_Exploration': {'type': 'epsilon_greedy', 'epsilon_start': 0.5, 'epsilon_min': 0.01, 'epsilon_decay': 0.999},
            'Constant_Exploration': {'type': 'epsilon_greedy', 'epsilon_start': 0.1, 'epsilon_min': 0.1, 'epsilon_decay': 1.0},
            'No_Exploration': {'type': 'epsilon_greedy', 'epsilon_start': 0.0, 'epsilon_min': 0.0, 'epsilon_decay': 1.0}
        }
        
        results = {}
        for strategy_name, strategy_config in exploration_strategies.items():
            print(f"\nTesting exploration strategy: {strategy_name}")
            
            # Create agent with specific exploration strategy
            agent = DQNAgent(
                state_dim=8, 
                action_dim=4,
                epsilon_start=strategy_config['epsilon_start'],
                epsilon_min=strategy_config['epsilon_min'],
                epsilon_decay=strategy_config['epsilon_decay']
            )
            
            # Standard environment
            env = Environment(
                grid_fp=self.base_config['grid_path'],
                no_gui=True,
                sigma=self.base_config['sigma'],
                agent_start_pos=self.base_config['agent_start_pos'],
                random_seed=self.base_config['random_seed'],
                state_representation='continuous_vector'
            )
            
            # Run experiment
            strategy_results = self._run_ablation_experiment(env, agent, episodes, strategy_name)
            results[strategy_name] = strategy_results
        
        self.results['exploration_strategies'] = results
        self._save_ablation_results('exploration_strategies', results)
        return results
    
    def _create_modified_environment(self, variant_type):
        """Create environment with modified state representation."""
        # This would require modifying the Environment class to support different state variants
        # For now, we'll use the standard environment and modify states in the agent
        return Environment(
            grid_fp=self.base_config['grid_path'],
            no_gui=True,
            sigma=self.base_config['sigma'],
            agent_start_pos=self.base_config['agent_start_pos'],
            random_seed=self.base_config['random_seed'],
            state_representation='continuous_vector'
        )
    
    def _get_state_dim(self, variant_type):
        """Get state dimension for different variants."""
        state_dims = {
            'full': 8,
            'no_pos': 6,      # Remove position (2 features)
            'no_clear': 4,    # Remove clearance (4 features)
            'no_mission': 6,  # Remove mission (2 features)
            'pos_only': 2,    # Only position
            'local_only': 6   # Only clearance + mission
        }
        return state_dims.get(variant_type, 8)
    
    def _create_custom_architecture_agent(self, arch_config):
        """Create agent with custom network architecture."""
        # Create standard DQN agent
        agent = DQNAgent(state_dim=8, action_dim=4)
        
        # Replace networks with custom architecture
        class CustomDQNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
                super().__init__()
                
                layers = []
                input_dim = state_dim
                
                for i in range(num_layers):
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    if i < num_layers - 1:  # Add dropout except for last layer
                        layers.append(nn.Dropout(0.1))
                    input_dim = hidden_dim
                
                layers.append(nn.Linear(hidden_dim, action_dim))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # Replace agent's networks
        agent.q_net = CustomDQNetwork(8, 4, arch_config['hidden_dim'], arch_config['num_layers']).to(agent.device)
        agent.target_q_net = CustomDQNetwork(8, 4, arch_config['hidden_dim'], arch_config['num_layers']).to(agent.device)
        agent.target_q_net.load_state_dict(agent.q_net.state_dict())
        
        # Update optimizer
        agent.optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=agent.lr)
        
        return agent
    
    def _create_training_variant_agent(self, variant_config):
        """Create agent with specific training components."""
        agent = DQNAgent(state_dim=8, action_dim=4)
        
        # Modify agent based on variant configuration
        if not variant_config['target_network']:
            # Use main network as target (no separate target network)
            agent.target_q_net = agent.q_net
        
        if not variant_config['replay_buffer']:
            # Implement online learning (no replay buffer)
            agent.min_replay_size = 1  # Train immediately
            agent.batch_size = 1       # Single sample training
        
        if not variant_config['epsilon_decay']:
            # Constant exploration
            agent.epsilon_decay = 1.0
        
        return agent
    
    def _run_ablation_experiment(self, env, agent, episodes, experiment_name):
        """Run single ablation experiment."""
        episode_rewards = []
        episode_lengths = []
        evaluation_rewards = []
        success_count = 0
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(1000):
                action = agent.take_training_action(state, training=True)
                next_state, reward, terminated, info = env.step(action)
                agent.update(state, reward, action, next_state, terminated)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if terminated:
                    success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Periodic evaluation
            if episode % 20 == 0 and episode > 0:
                eval_reward = self._evaluate_agent(env, agent)
                evaluation_rewards.append(eval_reward)
        
        # Final evaluation
        final_eval = self._evaluate_agent(env, agent, eval_episodes=10)
        
        return {
            'experiment_name': experiment_name,
            'episode_rewards': episode_rewards,
            'evaluation_rewards': evaluation_rewards,
            'final_evaluation': final_eval,
            'success_rate': success_count / episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'convergence_episode': self._find_convergence(episode_rewards)
        }
    
    def _evaluate_agent(self, env, agent, eval_episodes=5):
        """Evaluate agent with greedy policy."""
        total_rewards = []
        
        for _ in range(eval_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = agent.take_action(state)
                state, reward, terminated, info = env.step(action)
                episode_reward += reward
                
                if terminated:
                    break
                    
            total_rewards.append(episode_reward)
            
        return np.mean(total_rewards)
    
    def _find_convergence(self, rewards, window=20, threshold=100):
        """Find convergence point in training."""
        if len(rewards) < 2 * window:
            return len(rewards)
        
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        for i in range(window, len(moving_avg) - window):
            if moving_avg[i:i+window].std() < threshold:
                return i
        return len(rewards)
    
    def _save_ablation_results(self, study_name, results):
        """Save ablation study results."""
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for variant_name, variant_results in results.items():
            serializable_result = {}
            for key, value in variant_results.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
                    serializable_result[key] = [float(x) for x in value]
                else:
                    serializable_result[key] = value
            serializable_results[variant_name] = serializable_result
        
        # Save to JSON
        with open(self.experiment_dir / f"{study_name}_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create visualization
        self._create_ablation_plots(study_name, results)
    
    def _create_ablation_plots(self, study_name, results):
        """Create visualization for ablation study."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Ablation Study: {study_name.replace("_", " ").title()}', fontsize=16)
        
        variant_names = list(results.keys())
        
        # Plot 1: Mean rewards
        mean_rewards = [results[name]['mean_reward'] for name in variant_names]
        std_rewards = [results[name]['std_reward'] for name in variant_names]
        
        axes[0, 0].bar(range(len(variant_names)), mean_rewards, yerr=std_rewards, capsize=5)
        axes[0, 0].set_xticks(range(len(variant_names)))
        axes[0, 0].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Mean Episode Reward')
        axes[0, 0].set_title('Mean Performance Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Success rates
        success_rates = [results[name]['success_rate'] for name in variant_names]
        
        axes[0, 1].bar(range(len(variant_names)), success_rates)
        axes[0, 1].set_xticks(range(len(variant_names)))
        axes[0, 1].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_title('Success Rate Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Convergence speed
        convergence_episodes = [results[name]['convergence_episode'] for name in variant_names]
        
        axes[1, 0].bar(range(len(variant_names)), convergence_episodes)
        axes[1, 0].set_xticks(range(len(variant_names)))
        axes[1, 0].set_xticklabels(variant_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Episodes to Convergence')
        axes[1, 0].set_title('Convergence Speed')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning curves
        for name in variant_names:
            rewards = results[name]['episode_rewards']
            # Smooth with moving average
            smoothed = pd.Series(rewards).rolling(window=10).mean()
            axes[1, 1].plot(smoothed, label=name, alpha=0.7)
        
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward (Smoothed)')
        axes[1, 1].set_title('Learning Curves')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / f"{study_name}_ablation.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_all_ablation_studies(self, episodes=100):
        """Run complete ablation study suite."""
        print("=== RUNNING COMPLETE ABLATION STUDY SUITE ===")
        
        # Run all ablation studies
        self.study_state_representation(episodes)
        self.study_network_architecture(episodes)
        self.study_training_components(episodes)
        self.study_exploration_strategies(episodes)
        
        # Create comprehensive summary
        self._create_comprehensive_summary()
        
        print(f"\n=== ALL ABLATION STUDIES COMPLETE ===")
        print(f"Results saved to: {self.experiment_dir}")
        
        return self.results
    
    def _create_comprehensive_summary(self):
        """Create comprehensive summary of all ablation studies."""
        summary = {}
        
        for study_name, study_results in self.results.items():
            # Find best and worst variants
            best_variant = max(study_results.items(), key=lambda x: x[1]['mean_reward'])
            worst_variant = min(study_results.items(), key=lambda x: x[1]['mean_reward'])
            
            summary[study_name] = {
                'best_variant': {
                    'name': best_variant[0],
                    'mean_reward': best_variant[1]['mean_reward'],
                    'success_rate': best_variant[1]['success_rate']
                },
                'worst_variant': {
                    'name': worst_variant[0],
                    'mean_reward': worst_variant[1]['mean_reward'],
                    'success_rate': worst_variant[1]['success_rate']
                },
                'performance_gap': best_variant[1]['mean_reward'] - worst_variant[1]['mean_reward']
            }
        
        # Save comprehensive summary
        with open(self.experiment_dir / "comprehensive_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create summary visualization
        self._create_summary_plot(summary)
        
        return summary
    
    def _create_summary_plot(self, summary):
        """Create summary plot across all ablation studies."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Ablation Studies Summary', fontsize=16)
        
        study_names = list(summary.keys())
        best_rewards = [summary[name]['best_variant']['mean_reward'] for name in study_names]
        worst_rewards = [summary[name]['worst_variant']['mean_reward'] for name in study_names]
        performance_gaps = [summary[name]['performance_gap'] for name in study_names]
        
        # Plot 1: Best vs Worst performance
        x = range(len(study_names))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], best_rewards, width, label='Best Variant', alpha=0.8)
        axes[0].bar([i + width/2 for i in x], worst_rewards, width, label='Worst Variant', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([name.replace('_', ' ').title() for name in study_names], rotation=45, ha='right')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Best vs Worst Variants')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Performance gaps
        axes[1].bar(x, performance_gaps)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([name.replace('_', ' ').title() for name in study_names], rotation=45, ha='right')
        axes[1].set_ylabel('Performance Gap')
        axes[1].set_title('Component Importance (Performance Gap)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "ablation_summary.png", dpi=150, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    base_config = {
        'grid_path': Path('grid_configs/A1_grid.npy'),
        'sigma': 0.1,
        'agent_start_pos': (3, 11),
        'random_seed': 42
    }
    
    ablation = AblationStudy(base_config)
    results = ablation.run_all_ablation_studies(episodes=50)