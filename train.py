"""Updated train.py for simplified 10D continuous state testing."""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

# Handle module imports like original template
try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    # from agents.simple_dqn_agent import SimpleDQNAgent  # Uncomment when implementing DQN
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


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer - Assignment 2.")
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
                   choices=["random", "heuristic", "dqn"],
                   help="Type of agent to train/test.")
    p.add_argument("--state_representation", type=str, default="continuous_vector",
                   choices=["continuous_vector", "discrete"],
                   help="State representation to use.")
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of episodes to run.")
    p.add_argument("--agent_start_pos", type=int, nargs=2, default=None,
                   help="Starting position for agent as 'x y'")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_type: str, state_representation: str, 
         episodes: int, agent_start_pos: tuple = None):
    """Main training loop consistent with Assignment 1 style."""

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
            # Uncomment when DQN is implemented
            # agent = SimpleDQNAgent()
            print("DQN agent not yet implemented. Using heuristic instead.")
            agent = HeuristicAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
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
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print progress every 20 episodes
            if episode % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_length = np.mean(episode_lengths[-20:])
                success_rate = (success_count / (episode + 1)) * 100
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {avg_length:.1f}, Success Rate = {success_rate:.1f}%")

        # Final statistics (Assignment 1 style)
        print(f"\n=== TRAINING COMPLETED ===")
        print(f"Grid: {grid.name}")
        print(f"Agent: {agent_type}")
        print(f"Episodes: {episodes}")
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"Success rate: {(success_count / episodes) * 100:.1f}%")
        print(f"Best episode reward: {max(episode_rewards):.2f}")
        print(f"Shortest successful episode: {min([l for i, l in enumerate(episode_lengths) if episode_rewards[i] > 0]):.0f} steps")
        
        # Evaluate the trained agent (Assignment 1 style)
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
         args.episodes, start_pos)