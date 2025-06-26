"""
Example usage of optimized PPO and DQN agents with grid-specific hyperparameters.
This script demonstrates how to use the optimized agents for section 4.5 analysis.
"""
from pathlib import Path
import numpy as np

# Import optimized agents
from agents.optimized_dqn_agent import OptimizedDQNAgent
from agents.optimized_ppo_agent import OptimizedPPOAgent
from world import Environment


def train_optimized_agent(agent_type, grid_path, episodes=100, verbose=True):
    """Train an optimized agent on a specific grid."""
    
    # Grid-specific start positions
    start_positions = {
        'A1_grid.npy': (3, 11),
        'open_space.npy': (1, 1),
        'simple_restaurant.npy': (2, 2),
        'maze_challenge.npy': (1, 1),
        'assignment2_main.npy': (3, 11),
        'corridor_test.npy': (1, 1)
    }
    
    grid_name = Path(grid_path).name
    start_pos = start_positions.get(grid_name, (1, 1))
    
    print(f"\n=== Training {agent_type} on {grid_name} ===")
    print(f"Using optimized hyperparameters for this grid")
    
    # Create environment
    env = Environment(
        grid_fp=grid_path,
        no_gui=True,
        sigma=0.1,
        agent_start_pos=start_pos,
        random_seed=42,
        state_representation='continuous_vector'
    )
    
    # Create optimized agent - automatically selects best hyperparameters
    if agent_type.upper() == 'PPO':
        agent = OptimizedPPOAgent(
            state_dim=8,
            action_dim=4,
            grid_path=grid_path,  # This triggers optimal config selection
            verbose=verbose
        )
    elif agent_type.upper() == 'DQN':
        agent = OptimizedDQNAgent(
            state_dim=8,
            action_dim=4,
            grid_path=grid_path,  # This triggers optimal config selection
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Training loop
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(1000):  # Max steps per episode
            action = agent.take_training_action(state, training=True)
            next_state, reward, terminated, info = env.step(action)
            
            agent.update(state, reward, action, next_state, terminated)
            
            episode_reward += reward
            state = next_state
            
            if terminated:
                break
        
        episode_rewards.append(episode_reward)
        
        # Progress reporting
        if (episode + 1) % 25 == 0:
            recent_avg = np.mean(episode_rewards[-25:])
            print(f"Episode {episode + 1}/{episodes}, Recent avg reward: {recent_avg:.1f}")
    
    # Final performance metrics
    final_performance = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'final_10_avg': np.mean(episode_rewards[-10:]),
        'best_reward': np.max(episode_rewards),
        'episode_rewards': episode_rewards
    }
    
    print(f"Final Performance:")
    print(f"  Mean Reward: {final_performance['mean_reward']:.2f} ± {final_performance['std_reward']:.2f}")
    print(f"  Final 10 Episodes: {final_performance['final_10_avg']:.2f}")
    print(f"  Best Episode: {final_performance['best_reward']:.2f}")
    
    return agent, final_performance


def compare_optimized_agents_single_grid(grid_path, episodes=100):
    """Compare PPO vs DQN on a single grid using optimized hyperparameters."""
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZED AGENT COMPARISON: {Path(grid_path).name}")
    print(f"{'='*60}")
    
    results = {}
    
    # Train PPO with optimized hyperparameters
    ppo_agent, ppo_results = train_optimized_agent('PPO', grid_path, episodes, verbose=False)
    results['PPO'] = ppo_results
    
    # Train DQN with optimized hyperparameters  
    dqn_agent, dqn_results = train_optimized_agent('DQN', grid_path, episodes, verbose=False)
    results['DQN'] = dqn_results
    
    # Comparison summary
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"{'Metric':<20} {'PPO':<15} {'DQN':<15} {'Winner'}")
    print("-" * 55)
    
    metrics = ['mean_reward', 'std_reward', 'final_10_avg', 'best_reward']
    winners = {'PPO': 0, 'DQN': 0}
    
    for metric in metrics:
        ppo_val = ppo_results[metric]
        dqn_val = dqn_results[metric]
        
        if metric == 'std_reward':
            # Lower is better for standard deviation
            winner = 'PPO' if ppo_val < dqn_val else 'DQN'
        else:
            # Higher is better for other metrics
            winner = 'PPO' if ppo_val > dqn_val else 'DQN'
        
        winners[winner] += 1
        
        print(f"{metric:<20} {ppo_val:<15.2f} {dqn_val:<15.2f} {winner}")
    
    overall_winner = 'PPO' if winners['PPO'] > winners['DQN'] else 'DQN'
    print("-" * 55)
    print(f"{'OVERALL WINNER':<20} {'':<15} {'':<15} {overall_winner}")
    
    return results


def run_comprehensive_optimized_comparison():
    """Run comprehensive comparison across all grids with optimized hyperparameters."""
    
    # Test grids with their paths
    test_grids = {
        'Open Space': 'grid_configs/A2/open_space.npy',
        'Simple Restaurant': 'grid_configs/A2/simple_restaurant.npy', 
        'A1 Grid': 'grid_configs/A1_grid.npy',
        'Maze Challenge': 'grid_configs/A2/maze_challenge.npy'
    }
    
    all_results = {}
    
    print("COMPREHENSIVE OPTIMIZED AGENT COMPARISON")
    print("Using grid-specific optimized hyperparameters")
    print("="*70)
    
    for grid_name, grid_path in test_grids.items():
        try:
            grid_results = compare_optimized_agents_single_grid(grid_path, episodes=75)
            all_results[grid_name] = grid_results
        except Exception as e:
            print(f"Error testing {grid_name}: {e}")
            continue
    
    # Overall summary across all grids
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY ACROSS ALL GRIDS")
    print(f"{'='*70}")
    
    ppo_wins = 0
    dqn_wins = 0
    
    for grid_name, grid_results in all_results.items():
        ppo_mean = grid_results['PPO']['mean_reward']
        dqn_mean = grid_results['DQN']['mean_reward']
        winner = 'PPO' if ppo_mean > dqn_mean else 'DQN'
        
        if winner == 'PPO':
            ppo_wins += 1
        else:
            dqn_wins += 1
        
        improvement = abs(ppo_mean - dqn_mean) / max(abs(ppo_mean), abs(dqn_mean)) * 100
        print(f"{grid_name:<20}: {winner} wins (+{improvement:.1f}% better)")
    
    print(f"\nFINAL TALLY:")
    print(f"PPO wins: {ppo_wins}/{len(all_results)} grids")
    print(f"DQN wins: {dqn_wins}/{len(all_results)} grids")
    
    overall_champion = 'PPO' if ppo_wins > dqn_wins else 'DQN' if dqn_wins > ppo_wins else 'TIE'
    print(f"OVERALL CHAMPION: {overall_champion}")
    
    return all_results


if __name__ == "__main__":
    # Example 1: Train a single optimized agent
    print("Example 1: Training optimized PPO on A1_grid")
    agent, results = train_optimized_agent('PPO', 'grid_configs/A1_grid.npy', episodes=50)
    
    # Example 2: Compare agents on one grid
    print("\nExample 2: Comparing optimized agents on open_space")
    comparison = compare_optimized_agents_single_grid('grid_configs/A2/open_space.npy', episodes=50)
    
    # Example 3: Full comprehensive comparison (uncomment to run)
    # print("\nExample 3: Full comprehensive comparison")
    # full_results = run_comprehensive_optimized_comparison()
    
    print("\nOptimized agent examples complete!")
    print("Key benefits of optimized agents:")
    print("• Automatically select best hyperparameters for each grid")
    print("• Grid-specific learning rates, batch sizes, and architectures")
    print("• Improved sample efficiency and final performance")
    print("• Ready for section 4.5 comprehensive performance analysis")