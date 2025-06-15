"""
Algorithm comparison framework for Assignment 2.
Compares DQN baseline with advanced RL algorithms.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

try:
    from agents.base_agent import BaseAgent
    from agents.DQN_nn import DQNetwork, DuelingDQNetwork
    from world import Environment
    from logger import Logger
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from agents.base_agent import BaseAgent
    from agents.DQN_nn import DQNetwork, DuelingDQNetwork
    from world import Environment
    from logger import Logger


class DoubleDQNAgent(BaseAgent):
    """Double DQN implementation to reduce overestimation bias."""
    
    def __init__(self, state_dim=8, action_dim=4, **kwargs):
        super().__init__(state_dim, action_dim, 'continuous_vector')
        
        # Convert NumPy types to native Python types
        def convert_param(value):
            if isinstance(value, (np.integer, np.floating)):
                return float(value) if isinstance(value, np.floating) else int(value)
            return value
        
        # Use converted parameters
        self.gamma = convert_param(kwargs.get('gamma', 0.99))
        self.lr = convert_param(kwargs.get('lr', 1e-3))
        self.batch_size = convert_param(kwargs.get('batch_size', 64))
        self.buffer_size = convert_param(kwargs.get('buffer_size', 50000))
        self.min_replay_size = convert_param(kwargs.get('min_replay_size', 1000))
        self.target_update_freq = convert_param(kwargs.get('target_update_freq', 500))
        self.epsilon = convert_param(kwargs.get('epsilon_start', 1.0))
        self.epsilon_min = convert_param(kwargs.get('epsilon_min', 0.01))
        self.epsilon_decay = convert_param(kwargs.get('epsilon_decay', 0.995))
        
        # Rest of initialization...
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        self.training_steps = 0
        self.episode_count = 0
        self.losses = []
        
        print(f"Double DQN Agent initialized")
    
    def take_training_action(self, state, training=True):
        state = self.preprocess_state(state)
        
        if training and (np.random.rand() < self.epsilon):
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            return action
    
    def take_action(self, state):
        return self.take_training_action(state, training=False)
    
    def update(self, state, reward, action, next_state=None, done=False):
        if next_state is not None:
            state = self.preprocess_state(state)
            next_state = self.preprocess_state(next_state)
            self.replay_buffer.append((state, action, reward, next_state, done))
        
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        self._train_step()
        
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        if done:
            self.episode_count += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def _train_step(self):
        """Double DQN training step - key difference from regular DQN."""
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        current_q_values = self.q_net(states).gather(1, actions)
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_q_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())


class DuelingDQNAgent(BaseAgent):
    """Dueling DQN implementation for better value estimation."""
    
    def __init__(self, state_dim=8, action_dim=4, **kwargs):
        super().__init__(state_dim, action_dim, 'continuous_vector')
        
        # Convert NumPy types to native Python types
        def convert_param(value):
            if isinstance(value, (np.integer, np.floating)):
                return float(value) if isinstance(value, np.floating) else int(value)
            return value
        
        # Use converted parameters
        self.gamma = convert_param(kwargs.get('gamma', 0.99))
        self.lr = convert_param(kwargs.get('lr', 1e-3))
        self.batch_size = convert_param(kwargs.get('batch_size', 64))
        self.buffer_size = convert_param(kwargs.get('buffer_size', 50000))
        self.min_replay_size = convert_param(kwargs.get('min_replay_size', 1000))
        self.target_update_freq = convert_param(kwargs.get('target_update_freq', 500))
        self.epsilon = convert_param(kwargs.get('epsilon_start', 1.0))
        self.epsilon_min = convert_param(kwargs.get('epsilon_min', 0.01))
        self.epsilon_decay = convert_param(kwargs.get('epsilon_decay', 0.995))
        
        # Rest of initialization...
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingDQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = DuelingDQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        self.training_steps = 0
        self.episode_count = 0
        self.losses = []
        
        print(f"Dueling DQN Agent initialized")
    
    def take_training_action(self, state, training=True):
        state = self.preprocess_state(state)
        
        if training and (np.random.rand() < self.epsilon):
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            return action
    
    def take_action(self, state):
        return self.take_training_action(state, training=False)
    
    def update(self, state, reward, action, next_state=None, done=False):
        if next_state is not None:
            state = self.preprocess_state(state)
            next_state = self.preprocess_state(next_state)
            self.replay_buffer.append((state, action, reward, next_state, done))
        
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        self._train_step()
        
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        if done:
            self.episode_count += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def _train_step(self):
        """Standard DQN training but with dueling architecture."""
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        current_q_values = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())


class AlgorithmComparison:
    """Framework for comparing different RL algorithms."""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}
        self.experiment_dir = Path(f"experiments/algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def get_algorithms(self):
        """Define algorithms to compare."""
        from agents.DQN_agent import DQNAgent
        from agents.heuristic_agent import HeuristicAgent
        from agents.random_agent import RandomAgent
        # TODO: Import PPO agent when implemented
        # from agents.PPO_agent import PPOAgent
        
        algorithms = {
            'Random': RandomAgent,
            'Heuristic': HeuristicAgent, 
            'DQN': DQNAgent,
            'Double DQN': DoubleDQNAgent,
            'Dueling DQN': DuelingDQNAgent
        }
        
        # TODO: Add PPO when available
        # algorithms['PPO'] = PPOAgent
        
        return algorithms
    
    def run_comparison(self, episodes=100, num_runs=3):
        """Run comparison across multiple algorithms and seeds."""
        algorithms = self.get_algorithms()
        
        for alg_name, alg_class in algorithms.items():
            print(f"\n=== Testing {alg_name} ===")
            
            alg_results = []
            
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}")
                
                # Create fresh environment for each run
                env = Environment(
                    grid_fp=self.base_config['grid_path'],
                    no_gui=True,
                    sigma=self.base_config['sigma'],
                    agent_start_pos=self.base_config['agent_start_pos'],
                    random_seed=self.base_config['random_seed'] + run,
                    state_representation='continuous_vector'
                )
                
                # Initialize agent
                if alg_name in ['Random', 'Heuristic']:
                    agent = alg_class()
                # TODO: Add PPO initialization when implemented
                # elif alg_name == 'PPO':
                #     agent = alg_class(
                #         state_dim=8, 
                #         action_dim=4,
                #         lr=3e-4,  # Typical PPO learning rate
                #         gamma=0.99,
                #         clip_epsilon=0.2,  # PPO clipping parameter
                #         value_coef=0.5,    # Value function coefficient
                #         entropy_coef=0.01  # Entropy regularization
                #     )
                else:
                    agent = alg_class(state_dim=8, action_dim=4)
                
                # Run training
                run_results = self._run_single_experiment(env, agent, episodes, alg_name)
                run_results['run'] = run
                alg_results.append(run_results)
            
            self.results[alg_name] = alg_results
        
        # Analyze and save results
        self._analyze_algorithm_comparison()
        return self.results
    
    def _run_single_experiment(self, env, agent, episodes, alg_name):
        """Run single training experiment."""
        episode_rewards = []
        episode_lengths = []
        evaluation_rewards = []
        success_count = 0
        
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
                    success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Periodic evaluation for learning algorithms
            if episode % 20 == 0 and hasattr(agent, 'take_action') and alg_name not in ['Random', 'Heuristic']:
                eval_reward = self._evaluate_agent(env, agent)
                evaluation_rewards.append(eval_reward)
        
        # Final evaluation
        final_eval = self._evaluate_agent(env, agent, eval_episodes=10)
        
        return {
            'algorithm': alg_name,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'evaluation_rewards': evaluation_rewards,
            'final_evaluation': final_eval,
            'success_rate': success_count / episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'convergence_episode': self._find_convergence(episode_rewards)
        }
    
    def _find_convergence(self, rewards, window=20, threshold=100):
        """Find episode where performance converged."""
        if len(rewards) < 2 * window:
            return len(rewards)
        
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        for i in range(window, len(moving_avg) - window):
            if moving_avg[i:i+window].std() < threshold:
                return i
        return len(rewards)
    
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
    
    def _analyze_algorithm_comparison(self):
        """Analyze algorithm comparison results."""
        # Create summary statistics
        summary = {}
        
        for alg_name, runs in self.results.items():
            metrics = {
                'mean_reward_avg': np.mean([run['mean_reward'] for run in runs]),
                'mean_reward_std': np.std([run['mean_reward'] for run in runs]),
                'success_rate_avg': np.mean([run['success_rate'] for run in runs]),
                'success_rate_std': np.std([run['success_rate'] for run in runs]),
                'final_eval_avg': np.mean([run['final_evaluation'] for run in runs]),
                'final_eval_std': np.std([run['final_evaluation'] for run in runs]),
                'convergence_avg': np.mean([run['convergence_episode'] for run in runs])
            }
            summary[alg_name] = metrics
        
        # Save summary
        import json
        with open(self.experiment_dir / "algorithm_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create comparison plots
        self._create_comparison_plots()
        
        print(f"\n=== ALGORITHM COMPARISON COMPLETE ===")
        print(f"Results saved to: {self.experiment_dir}")
        
        # Print summary table
        print(f"\n{'Algorithm':<15} {'Mean Reward':<12} {'Success Rate':<12} {'Final Eval':<12}")
        print("-" * 60)
        for alg_name, metrics in summary.items():
            print(f"{alg_name:<15} {metrics['mean_reward_avg']:<12.1f} "
                  f"{metrics['success_rate_avg']:<12.1%} {metrics['final_eval_avg']:<12.1f}")
    
    def _create_comparison_plots(self):
        """Create visualization plots comparing algorithms."""
        # Learning curves comparison
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Learning curves
        plt.subplot(2, 3, 1)
        for alg_name, runs in self.results.items():
            all_rewards = []
            max_episodes = max(len(run['episode_rewards']) for run in runs)
            
            for episode in range(max_episodes):
                episode_rewards = []
                for run in runs:
                    if episode < len(run['episode_rewards']):
                        episode_rewards.append(run['episode_rewards'][episode])
                if episode_rewards:
                    all_rewards.append(np.mean(episode_rewards))
                else:
                    all_rewards.append(np.nan)
            
            plt.plot(all_rewards, label=alg_name, alpha=0.8)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Learning Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Success rates
        plt.subplot(2, 3, 2)
        alg_names = list(self.results.keys())
        success_rates = [np.mean([run['success_rate'] for run in runs]) 
                        for runs in self.results.values()]
        success_stds = [np.std([run['success_rate'] for run in runs]) 
                       for runs in self.results.values()]
        
        plt.bar(alg_names, success_rates, yerr=success_stds, capsize=5)
        plt.xlabel('Algorithm')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Mean rewards
        plt.subplot(2, 3, 3)
        mean_rewards = [np.mean([run['mean_reward'] for run in runs]) 
                       for runs in self.results.values()]
        mean_stds = [np.std([run['mean_reward'] for run in runs]) 
                    for runs in self.results.values()]
        
        plt.bar(alg_names, mean_rewards, yerr=mean_stds, capsize=5)
        plt.xlabel('Algorithm')
        plt.ylabel('Mean Episode Reward')
        plt.title('Mean Reward Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Final evaluation
        plt.subplot(2, 3, 4)
        final_evals = [np.mean([run['final_evaluation'] for run in runs]) 
                      for runs in self.results.values()]
        final_stds = [np.std([run['final_evaluation'] for run in runs]) 
                     for runs in self.results.values()]
        
        plt.bar(alg_names, final_evals, yerr=final_stds, capsize=5)
        plt.xlabel('Algorithm')
        plt.ylabel('Final Evaluation Score')
        plt.title('Final Performance Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Convergence speed
        plt.subplot(2, 3, 5)
        convergence_episodes = [np.mean([run['convergence_episode'] for run in runs]) 
                               for runs in self.results.values()]
        
        plt.bar(alg_names, convergence_episodes)
        plt.xlabel('Algorithm')
        plt.ylabel('Episodes to Convergence')
        plt.title('Convergence Speed Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Stability (variance)
        plt.subplot(2, 3, 6)
        reward_variances = [np.mean([run['std_reward'] for run in runs]) 
                           for runs in self.results.values()]
        
        plt.bar(alg_names, reward_variances)
        plt.xlabel('Algorithm')
        plt.ylabel('Reward Standard Deviation')
        plt.title('Training Stability Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "algorithm_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    base_config = {
        'grid_path': Path('grid_configs/A1_grid.npy'),
        'sigma': 0.1,
        'agent_start_pos': (3, 11),
        'random_seed': 42
    }
    
    comparison = AlgorithmComparison(base_config)
    results = comparison.run_comparison(episodes=100, num_runs=3)