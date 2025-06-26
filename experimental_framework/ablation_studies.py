"""
Simplified Ablation Studies Framework for Assignment 2.
FOCUSES ON: Network architecture, training components, and exploration strategies
AVOIDS: State representation changes (which require base agent modifications)
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
    from agents.PPO_agent import PPOAgent
    from world import Environment
    from logger import Logger
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from agents.DQN_agent import DQNAgent
    from agents.PPO_agent import PPOAgent
    from world import Environment
    from logger import Logger


class FlexibleDQNetwork(nn.Module):
    """Flexible DQN that can handle different architectures."""
    
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128, num_layers=3):
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)


class SimplifiedAblationStudy:
    """Simplified framework focusing on learnable components."""
    
    def __init__(self, base_config, agent_types=['dqn', 'ppo']):
        self.base_config = base_config
        self.agent_types = agent_types
        self.results = {}
        self.experiment_dir = Path(f"experiments/simplified_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.experiment_dir}")
    
    def study_network_architecture(self, episodes=50):
        """Ablation study on network architecture choices."""
        print("=== Network Architecture Ablation Study ===")
        
        architectures = {
            'Tiny_Net': {'hidden_dim': 32, 'num_layers': 2},
            'Small_Net': {'hidden_dim': 64, 'num_layers': 2},
            'Medium_Net': {'hidden_dim': 128, 'num_layers': 3},  # Standard
            'Large_Net': {'hidden_dim': 256, 'num_layers': 3},
            'Deep_Net': {'hidden_dim': 128, 'num_layers': 4},
        }
        
        results = {}
        
        for agent_type in self.agent_types:
            print(f"\n--- Testing {agent_type.upper()} agent ---")
            agent_results = {}
            
            for arch_name, arch_config in architectures.items():
                print(f"Testing architecture: {arch_name}")
                
                # Create environment
                env = self._create_standard_environment()
                
                # Create agent with custom architecture
                agent = self._create_custom_architecture_agent(agent_type, arch_config)
                
                # Run experiment
                arch_results = self._run_experiment(env, agent, episodes, f"{agent_type}_{arch_name}")
                agent_results[arch_name] = arch_results
            
            results[agent_type] = agent_results
        
        self.results['network_architecture'] = results
        self._save_results('network_architecture', results)
        return results
    
    def study_training_components(self, episodes=50):
        """Ablation study on training components."""
        print("=== Training Components Ablation Study ===")
        
        # Define training variants
        training_variants = {
            'dqn': {
                'Standard_DQN': {'target_update_freq': 100, 'batch_size': 32, 'replay_size': 10000},
                'Fast_Updates': {'target_update_freq': 50, 'batch_size': 32, 'replay_size': 10000},
                'Slow_Updates': {'target_update_freq': 200, 'batch_size': 32, 'replay_size': 10000},
                'Large_Batch': {'target_update_freq': 100, 'batch_size': 64, 'replay_size': 10000},
                'Small_Replay': {'target_update_freq': 100, 'batch_size': 32, 'replay_size': 5000},
            },
            'ppo': {
                'Standard_PPO': {'ppo_epochs': 4, 'batch_size': 32, 'rollout_steps': 64},
                'More_Epochs': {'ppo_epochs': 8, 'batch_size': 32, 'rollout_steps': 64},
                'Fewer_Epochs': {'ppo_epochs': 2, 'batch_size': 32, 'rollout_steps': 64},
                'Large_Batch': {'ppo_epochs': 4, 'batch_size': 64, 'rollout_steps': 64},
                'Long_Rollouts': {'ppo_epochs': 4, 'batch_size': 32, 'rollout_steps': 128},
            }
        }
        
        results = {}
        
        for agent_type in self.agent_types:
            print(f"\n--- Testing {agent_type.upper()} agent ---")
            agent_results = {}
            
            if agent_type in training_variants:
                for variant_name, variant_config in training_variants[agent_type].items():
                    print(f"Testing training variant: {variant_name}")
                    
                    # Create environment
                    env = self._create_standard_environment()
                    
                    # Create agent with training configuration
                    agent = self._create_training_variant_agent(agent_type, variant_config)
                    
                    # Run experiment
                    variant_results = self._run_experiment(env, agent, episodes, f"{agent_type}_{variant_name}")
                    agent_results[variant_name] = variant_results
            
            results[agent_type] = agent_results
        
        self.results['training_components'] = results
        self._save_results('training_components', results)
        return results
    
    def study_exploration_strategies(self, episodes=50):
        """Ablation study on exploration strategies."""
        print("=== Exploration Strategies Ablation Study ===")
        
        exploration_strategies = {
            'dqn': {
                'Standard_Epsilon': {'epsilon_start': 1.0, 'epsilon_min': 0.01, 'epsilon_decay': 0.995},
                'High_Exploration': {'epsilon_start': 1.0, 'epsilon_min': 0.1, 'epsilon_decay': 0.99},
                'Low_Exploration': {'epsilon_start': 0.5, 'epsilon_min': 0.01, 'epsilon_decay': 0.999},
                'Constant_High': {'epsilon_start': 0.3, 'epsilon_min': 0.3, 'epsilon_decay': 1.0},
                'No_Exploration': {'epsilon_start': 0.0, 'epsilon_min': 0.0, 'epsilon_decay': 1.0},
            },
            'ppo': {
                'Standard_Entropy': {'entropy_coef': 0.03},
                'High_Entropy': {'entropy_coef': 0.1},
                'Medium_Entropy': {'entropy_coef': 0.05},
                'Low_Entropy': {'entropy_coef': 0.01},
                'No_Entropy': {'entropy_coef': 0.0},
            }
        }
        
        results = {}
        
        for agent_type in self.agent_types:
            print(f"\n--- Testing {agent_type.upper()} agent ---")
            agent_results = {}
            
            if agent_type in exploration_strategies:
                for strategy_name, strategy_config in exploration_strategies[agent_type].items():
                    print(f"Testing exploration strategy: {strategy_name}")
                    
                    # Create environment
                    env = self._create_standard_environment()
                    
                    # Create agent with exploration strategy
                    agent = self._create_exploration_variant_agent(agent_type, strategy_config)
                    
                    # Run experiment
                    strategy_results = self._run_experiment(env, agent, episodes, f"{agent_type}_{strategy_name}")
                    agent_results[strategy_name] = strategy_results
            
            results[agent_type] = agent_results
        
        self.results['exploration_strategies'] = results
        self._save_results('exploration_strategies', results)
        return results
    
    def study_hyperparameter_sensitivity(self, episodes=50):
        """Study sensitivity to key hyperparameters."""
        print("=== Hyperparameter Sensitivity Study ===")
        
        hyperparameters = {
            'dqn': {
                'High_LR': {'lr': 0.001},
                'Standard_LR': {'lr': 0.0005},
                'Low_LR': {'lr': 0.0001},
                'High_Gamma': {'gamma': 0.99},
                'Low_Gamma': {'gamma': 0.95},
            },
            'ppo': {
                'High_LR': {'lr': 0.003},
                'Standard_LR': {'lr': 0.001},
                'Low_LR': {'lr': 0.0003},
                'High_Clip': {'clip_eps': 0.3},
                'Low_Clip': {'clip_eps': 0.1},
            }
        }
        
        results = {}
        
        for agent_type in self.agent_types:
            print(f"\n--- Testing {agent_type.upper()} agent ---")
            agent_results = {}
            
            if agent_type in hyperparameters:
                for param_name, param_config in hyperparameters[agent_type].items():
                    print(f"Testing hyperparameter: {param_name}")
                    
                    # Create environment
                    env = self._create_standard_environment()
                    
                    # Create agent with hyperparameter
                    agent = self._create_hyperparameter_variant_agent(agent_type, param_config)
                    
                    # Run experiment
                    param_results = self._run_experiment(env, agent, episodes, f"{agent_type}_{param_name}")
                    agent_results[param_name] = param_results
            
            results[agent_type] = agent_results
        
        self.results['hyperparameter_sensitivity'] = results
        self._save_results('hyperparameter_sensitivity', results)
        return results
    
    def _create_standard_environment(self):
        """Create standard environment with 8D state space."""
        return Environment(
            grid_fp=self.base_config['grid_path'],
            no_gui=True,
            sigma=self.base_config['sigma'],
            agent_start_pos=self.base_config['agent_start_pos'],
            random_seed=self.base_config['random_seed'],
            state_representation='continuous_vector'
        )
    
    def _create_custom_architecture_agent(self, agent_type, arch_config):
        """Create agent with custom architecture."""
        if agent_type == 'dqn':
            agent = DQNAgent(state_dim=8, action_dim=4, verbose=False)
            
            # Replace with custom network
            agent.q_net = FlexibleDQNetwork(
                8, 4, 
                arch_config['hidden_dim'], 
                arch_config['num_layers']
            ).to(agent.device)
            
            agent.target_q_net = FlexibleDQNetwork(
                8, 4, 
                arch_config['hidden_dim'], 
                arch_config['num_layers']
            ).to(agent.device)
            
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())
            agent.optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=agent.lr)
            
            return agent
        
        elif agent_type == 'ppo':
            # Create PPO with custom hidden dim (if supported)
            try:
                return PPOAgent(
                    state_dim=8, 
                    action_dim=4,
                    hidden_dim=arch_config['hidden_dim'],
                    verbose=False
                )
            except:
                # Fallback to standard PPO
                print(f"  Warning: Using standard PPO (custom hidden_dim not supported)")
                return PPOAgent(state_dim=8, action_dim=4, verbose=False)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _create_training_variant_agent(self, agent_type, variant_config):
        """Create agent with training configuration."""
        if agent_type == 'dqn':
            agent = DQNAgent(state_dim=8, action_dim=4, verbose=False)
            
            # Apply training configuration
            if 'target_update_freq' in variant_config:
                agent.target_update_freq = variant_config['target_update_freq']
            if 'batch_size' in variant_config:
                agent.batch_size = variant_config['batch_size']
            if 'replay_size' in variant_config:
                # Create new replay buffer with desired size
                from collections import deque
                new_size = variant_config['replay_size']
                # Keep existing experiences if buffer is smaller than new size
                existing_experiences = list(agent.replay_buffer)
                if len(existing_experiences) > new_size:
                    existing_experiences = existing_experiences[-new_size:]
                agent.replay_buffer = deque(existing_experiences, maxlen=new_size)
            
            return agent
        
        elif agent_type == 'ppo':
            # Create PPO with training configuration
            try:
                return PPOAgent(
                    state_dim=8, 
                    action_dim=4,
                    ppo_epochs=variant_config.get('ppo_epochs', 4),
                    batch_size=variant_config.get('batch_size', 32),
                    rollout_steps=variant_config.get('rollout_steps', 64),
                    verbose=False
                )
            except TypeError as e:
                # Fallback if PPO doesn't accept these parameters
                print(f"    Warning: PPO doesn't support custom parameters ({e})")
                agent = PPOAgent(state_dim=8, action_dim=4, verbose=False)
                
                # Try to set attributes directly
                if hasattr(agent, 'ppo_epochs'):
                    agent.ppo_epochs = variant_config.get('ppo_epochs', agent.ppo_epochs)
                if hasattr(agent, 'batch_size'):
                    agent.batch_size = variant_config.get('batch_size', agent.batch_size)
                if hasattr(agent, 'rollout_steps'):
                    agent.rollout_steps = variant_config.get('rollout_steps', agent.rollout_steps)
                
                return agent
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _create_exploration_variant_agent(self, agent_type, strategy_config):
        """Create agent with exploration strategy."""
        if agent_type == 'dqn':
            try:
                return DQNAgent(
                    state_dim=8, 
                    action_dim=4,
                    epsilon_start=strategy_config.get('epsilon_start', 1.0),
                    epsilon_min=strategy_config.get('epsilon_min', 0.01),
                    epsilon_decay=strategy_config.get('epsilon_decay', 0.995),
                    verbose=False
                )
            except TypeError:
                # Fallback if DQN doesn't accept these parameters
                agent = DQNAgent(state_dim=8, action_dim=4, verbose=False)
                
                # Set attributes directly
                if hasattr(agent, 'epsilon'):
                    agent.epsilon = strategy_config.get('epsilon_start', 1.0)
                if hasattr(agent, 'epsilon_min'):
                    agent.epsilon_min = strategy_config.get('epsilon_min', 0.01)
                if hasattr(agent, 'epsilon_decay'):
                    agent.epsilon_decay = strategy_config.get('epsilon_decay', 0.995)
                
                return agent
        
        elif agent_type == 'ppo':
            try:
                return PPOAgent(
                    state_dim=8, 
                    action_dim=4,
                    entropy_coef=strategy_config.get('entropy_coef', 0.03),
                    verbose=False
                )
            except TypeError:
                # Fallback if PPO doesn't accept entropy_coef in constructor
                agent = PPOAgent(state_dim=8, action_dim=4, verbose=False)
                if hasattr(agent, 'entropy_coef'):
                    agent.entropy_coef = strategy_config.get('entropy_coef', 0.03)
                return agent
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _create_hyperparameter_variant_agent(self, agent_type, param_config):
        """Create agent with hyperparameter configuration."""
        if agent_type == 'dqn':
            try:
                return DQNAgent(
                    state_dim=8, 
                    action_dim=4,
                    lr=param_config.get('lr', 0.0005),
                    gamma=param_config.get('gamma', 0.99),
                    verbose=False
                )
            except TypeError:
                # Fallback if DQN doesn't accept these parameters
                agent = DQNAgent(state_dim=8, action_dim=4, verbose=False)
                
                # Update learning rate directly in optimizer
                if 'lr' in param_config:
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = param_config['lr']
                
                # Set gamma if available
                if 'gamma' in param_config and hasattr(agent, 'gamma'):
                    agent.gamma = param_config['gamma']
                
                return agent
        
        elif agent_type == 'ppo':
            try:
                return PPOAgent(
                    state_dim=8, 
                    action_dim=4,
                    lr=param_config.get('lr', 0.001),
                    clip_eps=param_config.get('clip_eps', 0.2),
                    verbose=False
                )
            except TypeError:
                # Fallback if PPO doesn't accept these parameters
                agent = PPOAgent(state_dim=8, action_dim=4, verbose=False)
                
                # Update learning rate directly in optimizer
                if 'lr' in param_config:
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = param_config['lr']
                
                # Set clip_eps if available
                if 'clip_eps' in param_config and hasattr(agent, 'clip_eps'):
                    agent.clip_eps = param_config['clip_eps']
                
                return agent
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _run_experiment(self, env, agent, episodes, experiment_name):
        """Run single experiment."""
        print(f"    Running {episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Reset episode for PPO agents
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            
            for step in range(1000):
                # Take action
                if hasattr(agent, 'take_training_action'):
                    action = agent.take_training_action(state, training=True)
                else:
                    action = agent.take_action(state)
                
                # Execute action
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
            
            # Progress indicator
            if (episode + 1) % 10 == 0:
                recent_avg = np.mean(episode_rewards[-10:])
                print(f"      Episode {episode + 1}/{episodes}, Recent Avg: {recent_avg:.1f}")
        
        # Calculate metrics
        return {
            'experiment_name': experiment_name,
            'episode_rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / episodes,
            'convergence_episode': self._find_convergence(episode_rewards),
            'final_10_avg': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        }
    
    def _find_convergence(self, rewards, window=10, threshold=50):
        """Find convergence point in training."""
        if len(rewards) < 2 * window:
            return len(rewards)
        
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        for i in range(window, len(moving_avg) - window):
            if moving_avg[i:i+window].std() < threshold:
                return i
        return len(rewards)
    
    def _save_results(self, study_name, results):
        """Save results and create visualizations."""
        # Convert to JSON-serializable format
        serializable_results = {}
        for agent_type, agent_results in results.items():
            serializable_agent_results = {}
            for variant_name, variant_results in agent_results.items():
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
                serializable_agent_results[variant_name] = serializable_result
            serializable_results[agent_type] = serializable_agent_results
        
        # Save JSON
        with open(self.experiment_dir / f"{study_name}_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create plots
        self._create_plots(study_name, results)
        
        print(f"✅ {study_name} results saved to {self.experiment_dir}")
    
    def _create_plots(self, study_name, results):
        """Create visualizations."""
        if not results:
            return
        
        n_agents = len(results)
        fig, axes = plt.subplots(2, n_agents, figsize=(15, 10))
        if n_agents == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Ablation Study: {study_name.replace("_", " ").title()}', fontsize=16)
        
        for col, (agent_type, agent_results) in enumerate(results.items()):
            variant_names = list(agent_results.keys())
            
            # Plot 1: Mean rewards
            mean_rewards = [agent_results[name]['mean_reward'] for name in variant_names]
            std_rewards = [agent_results[name]['std_reward'] for name in variant_names]
            
            axes[0, col].bar(range(len(variant_names)), mean_rewards, yerr=std_rewards, capsize=5)
            axes[0, col].set_xticks(range(len(variant_names)))
            axes[0, col].set_xticklabels(variant_names, rotation=45, ha='right')
            axes[0, col].set_ylabel('Mean Reward')
            axes[0, col].set_title(f'{agent_type.upper()}: Mean Performance')
            axes[0, col].grid(True, alpha=0.3)
            
            # Plot 2: Success rates
            success_rates = [agent_results[name]['success_rate'] for name in variant_names]
            
            axes[1, col].bar(range(len(variant_names)), success_rates)
            axes[1, col].set_xticks(range(len(variant_names)))
            axes[1, col].set_xticklabels(variant_names, rotation=45, ha='right')
            axes[1, col].set_ylabel('Success Rate')
            axes[1, col].set_title(f'{agent_type.upper()}: Success Rate')
            axes[1, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / f"{study_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_all_studies(self, episodes=50):
        """Run all ablation studies."""
        print("=== RUNNING SIMPLIFIED ABLATION STUDY SUITE ===")
        print(f"Testing agent types: {', '.join([t.upper() for t in self.agent_types])}")
        print(f"Episodes per experiment: {episodes}")
        
        # Run studies
        self.study_network_architecture(episodes)
        self.study_training_components(episodes)
        self.study_exploration_strategies(episodes)
        self.study_hyperparameter_sensitivity(episodes)
        
        # Create summary
        self._create_summary()
        
        print(f"\n=== ALL STUDIES COMPLETE ===")
        print(f"Results saved to: {self.experiment_dir}")
        
        return self.results
    
    def _create_summary(self):
        """Create comprehensive summary."""
        summary = {}
        
        for study_name, study_results in self.results.items():
            study_summary = {}
            
            for agent_type, agent_results in study_results.items():
                if not agent_results:
                    continue
                
                # Find best and worst variants
                best_variant = max(agent_results.items(), key=lambda x: x[1]['mean_reward'])
                worst_variant = min(agent_results.items(), key=lambda x: x[1]['mean_reward'])
                
                study_summary[agent_type] = {
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
            
            summary[study_name] = study_summary
        
        # Save summary
        with open(self.experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary):
        """Print summary to console."""
        print(f"\n{'='*80}")
        print("SIMPLIFIED ABLATION STUDY SUMMARY")
        print(f"{'='*80}")
        
        for study_name, study_results in summary.items():
            print(f"\n📊 {study_name.replace('_', ' ').title()}:")
            print("-" * 60)
            
            for agent_type, agent_summary in study_results.items():
                print(f"\n{agent_type.upper()} Results:")
                print(f"  🥇 Best: {agent_summary['best_variant']['name']}")
                print(f"     Reward: {agent_summary['best_variant']['mean_reward']:.1f}")
                print(f"     Success: {agent_summary['best_variant']['success_rate']:.1%}")
                
                print(f"  🥉 Worst: {agent_summary['worst_variant']['name']}")
                print(f"     Reward: {agent_summary['worst_variant']['mean_reward']:.1f}")
                print(f"     Success: {agent_summary['worst_variant']['success_rate']:.1%}")
                
                print(f"  📈 Gap: {agent_summary['performance_gap']:.1f}")
        
        print(f"\n{'='*80}")


# Example usage
if __name__ == "__main__":
    # Configuration
    test_config = {
        'grid_path': Path('grid_configs/assignment2_main.npy'),
        'sigma': 0.1,
        'agent_start_pos': (3, 9),
        'random_seed': 42
    }
    
    print("🔬 Running Simplified Ablation Studies")
    
    # Run studies
    ablation = SimplifiedAblationStudy(test_config, agent_types=['dqn', 'ppo'])
    results = ablation.run_all_studies(episodes=25)  # Faster testing
    
    print("✅ Ablation studies completed!")