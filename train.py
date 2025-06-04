"""Updated train.py for continuous state space training."""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent  # NEW
    from agents.DQN.DQN_agent import DQNAgent  # Uncomment when implementing DQN
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer - Continuous States.")
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
                   help="Type of agent to train/test.")  # NEW
    p.add_argument("--state_representation", type=str, default="continuous_vector",
                   choices=["continuous_vector", "discrete"],
                   help="State representation to use.")  # NEW
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of episodes to run.")  # NEW
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_type: str, state_representation: str, episodes: int):
    """Main training loop with continuous states."""

    for grid in grid_paths:
        print(f"Training on grid: {grid}")
        print(f"Agent type: {agent_type}")
        print(f"State representation: {state_representation}")
        
        # Set up the environment with continuous states
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed, state_representation=state_representation)
        
        # Initialize agent based on type
        if agent_type == "random":
            agent = RandomAgent()
        elif agent_type == "heuristic":
            agent = HeuristicAgent()
        elif agent_type == "dqn":
            agent = DQNAgent()  # Uncomment when implementing
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Training statistics
        episode_rewards = []
        episode_lengths = []
        
        # Training loop
        for episode in trange(episodes, desc="Training Episodes"):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(iters):
                # Agent takes an action based on continuous state
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
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}")

        # Final statistics
        print(f"\nTraining completed!")
        print(f"Average reward over last 10 episodes: {np.mean(episode_rewards[-10:]):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        
        # Evaluate the trained agent
        print(f"\nEvaluating agent...")
        Environment.evaluate_agent(grid, agent, iters, sigma, 
                                 random_seed=random_seed, 
                                 state_representation=state_representation)


def test_state_representation():
    """Test function to verify continuous state representation works."""
    print("Testing continuous state representation...")
    
    # Create a simple test environment
    grid_path = Path("grid_configs/example_grid.npy")
    if not grid_path.exists():
        print(f"Warning: {grid_path} not found. Using default grid.")
        return
    
    env = Environment(grid_path, no_gui=True, state_representation='continuous_vector')
    state = env.reset()
    
    print(f"State shape: {state.shape}")
    print(f"State values: {state}")
    print(f"State features:")
    print(f"  Position: ({state[0]:.3f}, {state[1]:.3f})")
    print(f"  Target distance: {state[2]:.3f}")
    print(f"  Target direction: ({state[3]:.3f}, {state[4]:.3f})")
    print(f"  Remaining targets: {state[5]:.3f}")
    print(f"  Obstacle density: {state[6]:.3f}")
    print(f"  Clear directions: Front={state[7]:.0f}, Left={state[8]:.0f}, Right={state[9]:.0f}, Back={state[10]:.0f}")
    print(f"  Velocity: ({state[11]:.3f}, {state[12]:.3f})")
    print(f"  Speed: {state[13]:.3f}")
    print(f"  Progress: {state[14]:.3f}")
    
    # Test a few steps
    for i in range(3):
        action = np.random.randint(0, 4)
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")
        print(f"  New position: ({next_state[0]:.3f}, {next_state[1]:.3f})")


if __name__ == '__main__':
    args = parse_args()
    
    # Test the state representation first
    test_state_representation()
    print("\n" + "="*50 + "\n")
    
    # Run main training
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, 
         args.random_seed, args.agent_type, args.state_representation, args.episodes)