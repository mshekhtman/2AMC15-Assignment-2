"""Updated train.py for realistic 8D continuous state testing with PPO support and Logger functionality."""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

# Handle module imports like original template
try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.DQN_agent import DQNAgent
    from agents.PPO_agent import PPOAgent  # Added PPO support
    from logger import Logger
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.DQN_agent import DQNAgent
    from agents.PPO_agent import PPOAgent
    from logger import Logger


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer - Assignment 2 (8D State Space).")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--agent_type", type=str, default="heuristic", 
                   choices=["random", "heuristic", "dqn", "ppo"],  # Added PPO choice
                   help="Type of agent to train/test.")
    p.add_argument("--state_representation", type=str, default="continuous_vector",
                   choices=["continuous_vector", "discrete"],
                   help="State representation to use.")
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of episodes to run.")
    p.add_argument("--agent_start_pos", type=int, nargs=2, default=None,
                   help="Starting position for agent as 'x y'")
    p.add_argument("--save_agent", type=str, default=None,
                   help="Path to save trained agent")
    p.add_argument("--load_agent", type=str, default=None,
                   help="Path to load pre-trained agent")
    return p.parse_args()


def evaluate_agent_greedy(grid_fp: Path,
                         agent,
                         max_steps: int,
                         sigma: float = 0.,
                         agent_start_pos: tuple[int, int] = None,
                         random_seed: int = 0,
                         state_representation: str = 'continuous_vector'):
    """Evaluate agent with greedy policy (no exploration) for logger."""
    
    env = Environment(grid_fp=grid_fp,
                      no_gui=True,
                      sigma=sigma,
                      agent_start_pos=agent_start_pos,
                      target_fps=-1,
                      random_seed=random_seed,
                      state_representation=state_representation)
    
    state = env.reset()
    
    for _ in range(max_steps):
        # Use greedy action (no exploration)
        action = agent.take_action(state)
        state, _, terminated, _ = env.step(action)
        
        if terminated:
            break
    
    return env.world_stats


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_type: str, state_representation: str, 
         episodes: int, agent_start_pos: tuple = None, save_agent: str = None, load_agent: str = None):
    """Main training loop for realistic 8D state space with PPO and Logger functionality."""

    for grid in grid_paths:
        print(f"Training on grid: {grid}")
        print(f"Agent type: {agent_type}")
        print(f"State representation: {state_representation}")
        
        # Convert agent_start_pos to tuple if provided
        start_pos = tuple(agent_start_pos) if agent_start_pos else None
        
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed, state_representation=state_representation,
                          agent_start_pos=start_pos)

        # Initialize agent based on type
        if agent_type == "random":
            agent = RandomAgent()
        elif agent_type == "heuristic":
            agent = HeuristicAgent()
        elif agent_type == "dqn":
            agent = DQNAgent(state_dim=8, action_dim=4)  # Updated for 8D state space
            print("DQN agent initialized for 8D realistic continuous state space")
            
            # Load pre-trained agent if specified
            if load_agent:
                agent.load_agent(load_agent)
                
        elif agent_type == "ppo":  # Added PPO agent initialization
            agent = PPOAgent(state_dim=8, action_dim=4)  # 8D state space
            print("PPO agent initialized for 8D realistic continuous state space")
            
            # Load pre-trained agent if specified
            if load_agent:
                agent.load_agent(load_agent)
                
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Create a logger to keep track of training parameters and performance
        # Use agent hyperparameters if it's a learning agent, otherwise use defaults
        if agent_type == "dqn":
            logger = Logger(grid, sigma, agent.gamma, agent.lr, agent.batch_size, 
                          agent.buffer_size, agent.min_replay_size,
                          agent.target_update_freq, agent.epsilon, 
                          agent.epsilon_min, agent.epsilon_decay)
        elif agent_type == "ppo":  # Added PPO logger support
            logger = Logger(grid, sigma, agent.gamma, agent.optimizer.param_groups[0]['lr'], 
                          agent.batch_size, 0, 0, 0, 0, 0, 0)  # PPO doesn't use DQN-specific params
        else:
            logger = Logger(grid, sigma)

        # Training statistics
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # Training loop
        for episode in trange(episodes, desc="Training Episodes"):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(iters):
                # Agent takes an action
                if agent_type == "dqn":
                    # For DQN agent, use the training action method during training (with exploration)
                    action = agent.take_training_action(state, training=True)
                elif agent_type == "ppo":
                    # For PPO agent, use take_action (which handles exploration internally)
                    action = agent.take_action(state)
                else:
                    # For other agents, use their respective action methods
                    action = agent.take_action(state)

                # Execute action in environment
                next_state, reward, terminated, info = env.step(action)
                
                # Update agent (for learning agents)
                agent.update(state, reward, action, next_state, terminated)
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                state = next_state

                # Check if episode ended
                if terminated:
                    success_count += 1
                    break

            # Log target rewards (training with exploration)
            logger.log_target_rewards(episode_reward)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print progress and evaluate learning agents every 20 episodes
            if episode % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_length = np.mean(episode_lengths[-20:])
                success_rate = (success_count / (episode + 1)) * 100
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {avg_length:.1f}, Success Rate = {success_rate:.1f}%")
                
                # Log learning agent rewards with greedy evaluation
                if agent_type == "dqn" and len(agent.replay_buffer) > agent.min_replay_size:
                    # Evaluate DQN with greedy policy (no exploration)
                    stats = evaluate_agent_greedy(grid, agent, iters, sigma, 
                                                agent_start_pos=start_pos,
                                                random_seed=random_seed, 
                                                state_representation=state_representation)
                    logger.log_DQN_rewards(episode, stats['cumulative_reward'])
                    
                    # Show DQN training stats
                    training_stats = agent.get_training_stats()
                    print(f"  DQN Stats: epsilon={training_stats['epsilon']:.3f}, "
                          f"buffer_size={training_stats['buffer_size']}, "
                          f"avg_loss={training_stats['avg_loss_last_100']:.4f}")
                
                elif agent_type == "ppo":
                    # Evaluate PPO with current policy (already deterministic in evaluation)
                    stats = evaluate_agent_greedy(grid, agent, iters, sigma, 
                                                agent_start_pos=start_pos,
                                                random_seed=random_seed, 
                                                state_representation=state_representation)
                    logger.log_DQN_rewards(episode, stats['cumulative_reward'])
                    
                    # Show PPO training stats
                    training_stats = agent.get_training_stats()
                    print(f"  PPO Stats: episodes={training_stats['episode_count']}, "
                          f"updates={training_stats['update_count']}, "
                          f"policy_loss={training_stats['avg_policy_loss']:.4f}, "
                          f"entropy={training_stats['avg_entropy']:.4f}")

        # Save trained agent if specified
        if save_agent and agent_type in ["dqn", "ppo"]:
            agent.save_agent(save_agent)

        # After all episodes, plot the rewards
        logger.plot_target_rewards()
        logger.plot_DQN_rewards()  # Works for both DQN and PPO evaluation rewards
        logger.print_summary()

        # Final statistics (Assignment 1 style)
        print(f"\n=== TRAINING COMPLETED ===")
        print(f"Grid: {grid.name}")
        print(f"Agent: {agent_type}")
        print(f"Episodes: {episodes}")
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"Success rate: {(success_count / episodes) * 100:.1f}%")
        print(f"Best episode reward: {max(episode_rewards):.2f}")
        
        # Only show shortest successful episode if there were any successes
        successful_lengths = [l for i, l in enumerate(episode_lengths) if episode_rewards[i] > 0]
        if successful_lengths:
            print(f"Shortest successful episode: {min(successful_lengths):.0f} steps")
        else:
            print("No successful episodes completed")
        
        # Show final training stats for learning agents
        if agent_type in ["dqn", "ppo"]:
            final_stats = agent.get_training_stats()
            print(f"\nFinal Training Stats:")
            for key, value in final_stats.items():
                print(f"  {key}: {value}")
        
        # Evaluate the trained agent (Assignment 1 style) - THIS CREATES THE PATH IMAGE
        print(f"\n=== EVALUATING AGENT ===")
        Environment.evaluate_agent(grid, agent, iters, sigma, 
                                 agent_start_pos=start_pos,
                                 random_seed=random_seed, 
                                 state_representation=state_representation,
                                 show_images=not no_gui)


if __name__ == '__main__':
    args = parse_args()
    
    # Convert agent_start_pos from list to tuple
    start_pos = args.agent_start_pos
    
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, 
         args.random_seed, args.agent_type, args.state_representation, 
         args.episodes, start_pos, args.save_agent, args.load_agent)