"""
Complete world/environment.py with continuous state space implementation.
"""
import random
import datetime
import numpy as np
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from world.helpers import save_results, action_to_direction

try:
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path

class Environment:
    def __init__(self,
                 grid_fp: Path,
                 no_gui: bool = False,
                 sigma: float = 0.,
                 agent_start_pos: tuple[int, int] = None,
                 reward_fn: callable = None,
                 target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = 0,
                 state_representation: str = 'continuous_vector'):
        
        """Creates the Grid Environment for the Reinforcement Learning robot
        from the provided file with continuous state space support.

        This environment follows the general principles of reinforcement
        learning. It can be thought of as a function E : action -> observation
        where E is the environment represented as a function.

        Args:
            grid_fp: Path to the grid file to use.
            no_gui: True if no GUI is desired.
            sigma: The stochasticity of the environment. The probability that
                the agent makes the move that it has provided as an action is
                calculated as 1-sigma.
            agent_start_pos: Tuple where each agent should start.
                If None is provided, then a random start position is used.
            reward_fn: Custom reward function to use. 
            target_fps: How fast the simulation should run if it is being shown
                in a GUI. If in no_gui mode, then the simulation will run as fast as
                possible. We may set a low FPS so we can actually see what's
                happening. Set to 0 or less to unlock FPS.
            random_seed: The random seed to use for this environment. If None
                is provided, then the seed will be set to 0.
            state_representation: Type of state representation to use.
                'continuous_vector' for 15D vector, 'discrete' for backward compatibility.
        """
        random.seed(random_seed)

        # Initialize Grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        else:
            self.grid_fp = grid_fp

        # Initialize other variables
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma
        
        # NEW: State representation configuration
        self.state_representation = state_representation
        self.movement_history = []  # Track recent positions for velocity calculation
        self.initial_target_count = 0  # Track initial number of targets for progress calculation
              
        # Set up reward function
        if reward_fn is None:
            if state_representation == 'continuous_vector':
                warn("No reward function provided. Using restaurant delivery reward.")
                self.reward_fn = self._restaurant_delivery_reward
            else:
                warn("No reward function provided. Using default reward.")
                self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        # GUI specific code: Set up the environment as a blank state.
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None

    def _reset_info(self) -> dict:
        """Resets the info dictionary.

        info is a dict with information of the most recent step
        consisting of whether the target was reached or the agent
        moved and the updated agent position.
        """
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None}
    
    @staticmethod
    def _reset_world_stats() -> dict:
        """Resets the world stats dictionary.

        world_stats is a dict with information about the 
        environment since last env.reset(). Basically, it
        accumulates information.
        """
        return {"cumulative_reward": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_failed_moves": 0,
                "total_targets_reached": 0,
                }

    def _initialize_agent_pos(self):
        """Initializes agent position from the given location or
        randomly chooses one if None was given.
        """

        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1])
            if self.grid[pos] == 0:
                # Cell is empty. We can place the agent there.
                self.agent_pos = pos
            else:
                raise ValueError(
                    "Attempted to place agent on top of obstacle, delivery"
                    "location or charger")
        else:
            # No positions were given. We place agents randomly.
            warn("No initial agent positions given. Randomly placing agents "
                 "on the grid.")
            # Find all empty locations and choose one at random
            zeros = np.where(self.grid == 0)
            idx = random.randint(0, len(zeros[0]) - 1)
            self.agent_pos = (zeros[0][idx], zeros[1][idx])

    def reset(self, **kwargs) -> np.ndarray:
        """Reset the environment to an initial state and return continuous state.

        You can fit it keyword arguments which will overwrite the 
        initial arguments provided when initializing the environment.

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        Returns:
             initial state as numpy array (continuous vector or discrete position).
        """
        for k, v in kwargs.items():
            # Go through each possible keyword argument.
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 1. / v if v > 0 else 0.
                case "state_representation":
                    self.state_representation = v
                case _:
                    raise ValueError(f"{k} is not one of the possible "
                                     f"keyword arguments.")
        
        # Reset variables
        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()

        # NEW: Initialize continuous state tracking
        self.movement_history = [self.agent_pos]
        self.initial_target_count = np.sum(self.grid == 3)

        # GUI specific code
        if not self.no_gui:
            self.gui = GUI(self.grid.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()

        # NEW: Return appropriate state representation
        if self.state_representation == 'continuous_vector':
            return self.get_restaurant_delivery_state()
        else:
            return np.array(self.agent_pos, dtype=np.float32)

    def _move_agent(self, new_pos: tuple[int, int]):
        """Moves the agent, if possible and updates the 
        corresponding stats.

        Args:
            new_pos: The new position of the agent.
        """

        match self.grid[new_pos]:
            case 0:  # Moved to an empty tile
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case 1 | 2:  # Moved to a wall or obstacle
                self.world_stats["total_failed_moves"] += 1
                self.info["agent_moved"] = False
                pass
            case 3:  # Moved to a target tile
                self.agent_pos = new_pos
                self.grid[new_pos] = 0
                if np.sum(self.grid == 3) == 0:
                    self.terminal_state = True
                self.info["target_reached"] = True
                self.world_stats["total_targets_reached"] += 1
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
                # Otherwise, the agent can't move and nothing happens
            case _:
                raise ValueError(f"Grid is badly formed. It has a value of "
                                 f"{self.grid[new_pos]} at position "
                                 f"{new_pos}.")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """This function makes the agent take a step on the grid.

        Action is provided as integer and values are:
            - 0: Move down
            - 1: Move up
            - 2: Move left
            - 3: Move right
        Args:
            action: Integer representing the action the agent should
                take. 

        Returns:
            0) Current state as numpy array,
            1) The reward for the agent,
            2) If the terminal state has been reached,
            3) Info dictionary
        """
        
        self.world_stats["total_steps"] += 1
        
        # GUI specific code
        is_single_step = False
        if not self.no_gui:
            start_time = time()
            while self.gui.paused:
                # If the GUI is paused but asking to step, then we step
                if self.gui.step:
                    is_single_step = True
                    self.gui.step = False
                    break
                # Otherwise, we render the current state only
                paused_info = self._reset_info()
                paused_info["agent_moved"] = True
                self.gui.render(self.grid, self.agent_pos, paused_info,
                                0, is_single_step)    

        # Add stochasticity into the agent action
        val = random.random()
        if val > self.sigma:
            actual_action = action
        else:
            actual_action = random.randint(0, 3)
        
        # NEW: Store previous position for enhanced reward calculation
        prev_pos = self.agent_pos

        # Make the move
        self.info["actual_action"] = actual_action
        direction = action_to_direction(actual_action)    
        new_pos = (self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1])

        # Calculate the reward for the agent
        reward = self.reward_fn(self.grid, new_pos, prev_pos)

        self._move_agent(new_pos)
        
        # NEW: Update movement history after successful move
        if self.info["agent_moved"]:
            self.movement_history.append(self.agent_pos)
            if len(self.movement_history) > 5:  # Keep last 5 positions for velocity calculation
                self.movement_history.pop(0)
        
        self.world_stats["cumulative_reward"] += reward

        # NEW: Return appropriate state representation
        if self.state_representation == 'continuous_vector':
            next_state = self.get_restaurant_delivery_state()
        else:
            next_state = np.array(self.agent_pos, dtype=np.float32)

        # GUI specific code
        if not self.no_gui:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.gui.render(self.grid, self.agent_pos, self.info,
                            reward, is_single_step)

        return next_state, reward, self.terminal_state, self.info

    def get_restaurant_delivery_state(self) -> np.ndarray:
        """NEW METHOD: Generate continuous state vector for restaurant delivery robot.
        
        Returns:
            15-dimensional state vector containing:
            [0-1]: Normalized position (x, y)
            [2]: Distance to nearest target (normalized)
            [3-4]: Direction to nearest target (unit vector)
            [5]: Remaining targets (normalized)
            [6]: Local obstacle density
            [7-10]: Clear directions (front, left, right, back)
            [11-12]: Velocity vector
            [13]: Speed
            [14]: Mission progress
        """
        
        # 1. Continuous position (normalized to [0,1])
        norm_x = self.agent_pos[0] / self.grid.shape[0]
        norm_y = self.agent_pos[1] / self.grid.shape[1]
        
        # 2. Target information
        target_positions = np.where(self.grid == 3)
        if len(target_positions[0]) > 0:
            targets = list(zip(target_positions[0], target_positions[1]))
            
            # Distance to nearest target (normalized)
            distances = [np.sqrt((self.agent_pos[0] - tx)**2 + (self.agent_pos[1] - ty)**2) 
                        for tx, ty in targets]
            max_distance = np.sqrt(self.grid.shape[0]**2 + self.grid.shape[1]**2)
            min_target_dist = min(distances) / max_distance
            
            # Direction vector to nearest target (unit vector)
            nearest_target = targets[np.argmin(distances)]
            dx = nearest_target[0] - self.agent_pos[0]
            dy = nearest_target[1] - self.agent_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                target_dir_x, target_dir_y = dx/dist, dy/dist
            else:
                target_dir_x, target_dir_y = 0, 0
            
            # Number of remaining targets (normalized by assumed max of 10)
            remaining_targets = len(targets) / 10.0
        else:
            min_target_dist = 0
            target_dir_x, target_dir_y = 0, 0
            remaining_targets = 0
        
        # 3. Local environment awareness (3x3 grid around agent)
        local_view = self._get_local_view(radius=1)
        
        # Convert to interpretable features
        obstacle_count = np.sum((local_view == 1) | (local_view == 2))
        obstacle_density = obstacle_count / 9.0  # Normalize by 3x3 grid size
        
        # Check specific directions for obstacles (more interpretable than raw grid)
        front_clear = float(local_view[0, 1] == 0)  # Front is clear (up direction)
        left_clear = float(local_view[1, 0] == 0)   # Left is clear  
        right_clear = float(local_view[1, 2] == 0)  # Right is clear
        back_clear = float(local_view[2, 1] == 0)   # Back is clear (down direction)
        
        # 4. Movement dynamics (for smooth navigation)
        velocity = self._get_velocity()
        speed = np.linalg.norm(velocity)
        
        # 5. Mission progress
        if self.initial_target_count > 0:
            current_targets = len(target_positions[0]) if len(target_positions[0]) > 0 else 0
            progress = 1.0 - (current_targets / self.initial_target_count)
        else:
            progress = 1.0
        
        # Combine into state vector (15 features total)
        state_vector = np.array([
            # Position (2 features)
            norm_x, norm_y,
            
            # Target information (4 features)
            min_target_dist, target_dir_x, target_dir_y, remaining_targets,
            
            # Local environment (5 features)
            obstacle_density, front_clear, left_clear, right_clear, back_clear,
            
            # Movement (3 features)
            velocity[0], velocity[1], speed,
            
            # Mission status (1 feature)
            progress
        ], dtype=np.float32)
        
        return state_vector

    def _get_local_view(self, radius: int = 1) -> np.ndarray:
        """NEW METHOD: Get local grid view around agent position.
        
        Args:
            radius: Radius of the local view (1 = 3x3, 2 = 5x5, etc.)
            
        Returns:
            Local grid view as numpy array
        """
        x, y = self.agent_pos
        view_size = 2 * radius + 1
        local_view = np.zeros((view_size, view_size))
        
        for i in range(view_size):
            for j in range(view_size):
                grid_x = x + i - radius
                grid_y = y + j - radius
                
                if (0 <= grid_x < self.grid.shape[0] and 
                    0 <= grid_y < self.grid.shape[1]):
                    local_view[i, j] = self.grid[grid_x, grid_y]
                else:
                    local_view[i, j] = 1  # Treat out-of-bounds as boundary
        
        return local_view

    def _get_velocity(self) -> np.ndarray:
        """NEW METHOD: Calculate velocity based on recent movement history.
        
        Returns:
            2D velocity vector normalized by grid size
        """
        if len(self.movement_history) < 2:
            return np.array([0.0, 0.0])
        
        # Calculate velocity as change in position
        recent_pos = np.array(self.movement_history[-2:])
        velocity = recent_pos[-1] - recent_pos[-2]
        
        # Normalize by grid size for consistent scale
        velocity = velocity.astype(np.float32) / np.array([self.grid.shape[0], self.grid.shape[1]])
        
        return velocity

    def _restaurant_delivery_reward(self, grid: np.ndarray, agent_pos: tuple[int, int], 
                                  prev_pos: tuple[int, int] = None) -> float:
        """NEW METHOD: Enhanced reward function for restaurant delivery robot.
        
        Optimized for:
        - Delivery efficiency (time penalty)
        - Safety (collision avoidance)
        - Customer comfort (smooth movement)
        - Goal achievement (target reaching)
        
        Args:
            grid: Current grid state
            agent_pos: Position agent is moving to
            prev_pos: Previous agent position (for movement analysis)
            
        Returns:
            Reward value
        """
        reward = 0
        
        # Basic movement rewards
        if grid[agent_pos] == 0:  # Empty space
            reward -= 1  # Time penalty for efficiency
        elif grid[agent_pos] == 1 or grid[agent_pos] == 2:  # Collision with wall/obstacle
            reward -= 10  # Higher penalty for safety
            return reward  # Return immediately, don't process further
        elif grid[agent_pos] == 3:  # Target reached
            reward += 20  # Higher reward for successful delivery
        
        # Smoothness reward (encourage smooth movement for customer comfort)
        if prev_pos is not None and len(self.movement_history) >= 3:
            # Calculate acceleration magnitude (change in velocity)
            positions = np.array(self.movement_history[-3:])
            if len(positions) >= 3:
                v1 = positions[-2] - positions[-3]  # Previous velocity
                v2 = positions[-1] - positions[-2]  # Current velocity
                acceleration = np.linalg.norm(v2 - v1)
                reward -= 0.5 * acceleration  # Penalty for jerky movement
        
        # Efficiency bonus: reward for moving towards targets
        if prev_pos is not None:
            target_positions = np.where(grid == 3)
            if len(target_positions[0]) > 0:
                targets = list(zip(target_positions[0], target_positions[1]))
                
                # Distance before and after move
                prev_distances = [np.sqrt((prev_pos[0] - tx)**2 + (prev_pos[1] - ty)**2) 
                                for tx, ty in targets]
                curr_distances = [np.sqrt((agent_pos[0] - tx)**2 + (agent_pos[1] - ty)**2) 
                                for tx, ty in targets]
                
                # Small reward for getting closer to nearest target
                if min(curr_distances) < min(prev_distances):
                    reward += 0.5
                # Small penalty for moving away
                elif min(curr_distances) > min(prev_distances):
                    reward -= 0.2
        
        return reward

    @staticmethod
    def _default_reward_function(grid, agent_pos, prev_pos=None) -> float:
        """This is the original simple reward function for backward compatibility.
        Any custom reward function must also follow the same signature, meaning
        it must be written like `reward_name(grid, agent_pos, prev_pos)`.

        Args:
            grid: The grid the agent is moving on, in case that is needed by
                the reward function.
            agent_pos: The position the agent is moving to.
            prev_pos: Previous position (ignored in this function)

        Returns:
            A single floating point value representing the reward for a given
            action.
        """

        match grid[agent_pos]:
            case 0:  # Moved to an empty tile
                reward = -1
            case 1 | 2:  # Moved to a wall or obstacle
                reward = -5
                pass
            case 3:  # Moved to a target tile
                reward = 10
                # "Illegal move"
            case _:
                raise ValueError(f"Grid cell should not have value: {grid[agent_pos]}.",
                                 f"at position {agent_pos}")
        return reward

    @staticmethod
    def evaluate_agent(grid_fp: Path,
                       agent: BaseAgent,
                       max_steps: int,
                       sigma: float = 0.,
                       agent_start_pos: tuple[int, int] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       show_images: bool = False,
                       state_representation: str = 'continuous_vector'):
        """Evaluates a single trained agent's performance with continuous states.

        What this does is it creates a completely new environment from the
        provided grid and does a number of steps _without_ processing rewards
        for the agent. This means that the agent doesn't learn here and simply
        provides actions for any provided observation.

        For each evaluation run, this produces a statistics file in the out
        directory which is a txt. This txt contains the values:
        [ 'total_steps`, `total_failed_moves`]

        Args:
            grid_fp: Path to the grid file to use.
            agent: Trained agent to evaluate.
            max_steps: Max number of steps to take.
            sigma: same as above.
            agent_start_pos: same as above.
            random_seed: same as above.
            show_images: Whether to show the images at the end of the
                evaluation. If False, only saves the images.
            state_representation: State representation to use for evaluation.
        """

        env = Environment(grid_fp=grid_fp,
                          no_gui=True,
                          sigma=sigma,
                          agent_start_pos=agent_start_pos,
                          target_fps=-1,
                          random_seed=random_seed,
                          state_representation=state_representation)
        
        state = env.reset()
        initial_grid = np.copy(env.grid)

        # Add initial agent position to the path
        agent_path = [env.agent_pos]

        for _ in trange(max_steps, desc="Evaluating agent"):
            
            action = agent.take_action(state)
            state, _, terminated, _ = env.step(action)

            agent_path.append(env.agent_pos)  # Use actual position for visualization

            if terminated:
                break

        env.world_stats["targets_remaining"] = np.sum(env.grid == 3)

        path_image = visualize_path(initial_grid, agent_path)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        save_results(file_name, env.world_stats, path_image, show_images)