"""
Simplified world/environment.py with realistic 8D state space and IMPROVED REWARD FUNCTION
Key changes:
- Enhanced reward function with progress tracking and dense rewards
- Better learning signals for DQN training
- Maintains all existing functionality
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
        
        """Creates the Grid Environment with realistic 8D continuous state space.

        Args:
            grid_fp: Path to the grid file to use.
            no_gui: True if no GUI is desired.
            sigma: The stochasticity of the environment.
            agent_start_pos: Tuple where agent should start.
            reward_fn: Custom reward function to use.
            target_fps: How fast the simulation should run in GUI.
            random_seed: Random seed for environment.
            state_representation: 'continuous_vector' for 8D vector, 'discrete' for backward compatibility.
        """
        random.seed(random_seed)

        # Initialize Grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        else:
            self.grid_fp = grid_fp

        # Initialize variables
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma
        
        # State representation configuration
        self.state_representation = state_representation
        self.last_position = None  # Simple movement tracking
        self.initial_target_count = 0
        
        # ENHANCED REWARD TRACKING
        self.previous_distance_to_nearest = None
        self.visited_positions = set()
        self.steps_without_progress = 0
              
        # Set up reward function
        if reward_fn is None:
            warn("No reward function provided. Using enhanced restaurant reward.")
            self.reward_fn = self._enhanced_restaurant_reward
        else:
            self.reward_fn = reward_fn

        # GUI setup
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None

    def _reset_info(self) -> dict:
        """Resets the info dictionary."""
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None}
    
    @staticmethod
    def _reset_world_stats() -> dict:
        """Resets the world stats dictionary."""
        return {"cumulative_reward": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_failed_moves": 0,
                "total_targets_reached": 0,}

    def _initialize_agent_pos(self):
        """Initializes agent position."""
        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1])
            if self.grid[pos] == 0:
                self.agent_pos = pos
            else:
                raise ValueError("Attempted to place agent on obstacle or target")
        else:
            warn("No initial agent positions given. Randomly placing agent.")
            zeros = np.where(self.grid == 0)
            idx = random.randint(0, len(zeros[0]) - 1)
            self.agent_pos = (zeros[0][idx], zeros[1][idx])

    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return initial state."""
        for k, v in kwargs.items():
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
                    raise ValueError(f"{k} is not a valid keyword argument.")
        
        # Reset variables
        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()

        # Initialize simplified tracking
        self.last_position = None
        self.initial_target_count = np.sum(self.grid == 3)
        
        # RESET ENHANCED REWARD TRACKING
        self.previous_distance_to_nearest = None
        self.visited_positions.clear()
        self.steps_without_progress = 0

        # GUI setup
        if not self.no_gui:
            self.gui = GUI(self.grid.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()

        # Return appropriate state representation
        if self.state_representation == 'continuous_vector':
            return self.get_realistic_delivery_state()
        else:
            return np.array(self.agent_pos, dtype=np.float32)

    def _move_agent(self, new_pos: tuple[int, int]):
        """Moves the agent and updates stats."""
        match self.grid[new_pos]:
            case 0:  # Empty tile
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case 1 | 2:  # Wall or obstacle
                self.world_stats["total_failed_moves"] += 1
                self.info["agent_moved"] = False
            case 3:  # Target tile
                self.agent_pos = new_pos
                self.grid[new_pos] = 0
                if np.sum(self.grid == 3) == 0:
                    self.terminal_state = True
                self.info["target_reached"] = True
                self.world_stats["total_targets_reached"] += 1
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case _:
                raise ValueError(f"Invalid grid value {self.grid[new_pos]} at {new_pos}")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment."""
        self.world_stats["total_steps"] += 1
        
        # GUI handling
        is_single_step = False
        if not self.no_gui:
            start_time = time()
            while self.gui.paused:
                if self.gui.step:
                    is_single_step = True
                    self.gui.step = False
                    break
                paused_info = self._reset_info()
                paused_info["agent_moved"] = True
                self.gui.render(self.grid, self.agent_pos, paused_info, 0, is_single_step)    

        # Add stochasticity
        val = random.random()
        if val > self.sigma:
            actual_action = action
        else:
            actual_action = random.randint(0, 3)
        
        # Store previous position for reward calculation
        prev_pos = self.agent_pos

        # Execute action
        self.info["actual_action"] = actual_action
        direction = action_to_direction(actual_action)    
        new_pos = (self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1])

        # Calculate reward
        reward = self.reward_fn(self.grid, new_pos, prev_pos)

        # Move agent
        self._move_agent(new_pos)
        
        # Update position tracking
        if self.info["agent_moved"]:
            self.last_position = prev_pos
        
        self.world_stats["cumulative_reward"] += reward

        # Return appropriate state representation
        if self.state_representation == 'continuous_vector':
            next_state = self.get_realistic_delivery_state()
        else:
            next_state = np.array(self.agent_pos, dtype=np.float32)

        # GUI rendering
        if not self.no_gui:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.gui.render(self.grid, self.agent_pos, self.info, reward, is_single_step)

        return next_state, reward, self.terminal_state, self.info

    def get_realistic_delivery_state(self) -> np.ndarray:
        """Generate realistic 8D continuous state vector.
        
        This represents what a real delivery robot could actually sense:
        - Its own position (from GPS/odometry)
        - Local obstacle detection (from sensors like lidar/cameras)
        - Mission status (from internal tracking)
        
        Returns:
            8-dimensional state vector:
            [0-1]: Normalized position (x, y)
            [2-5]: Clear directions (down, up, left, right) - obstacle detection
            [6]: Remaining targets (normalized) - mission tracking
            [7]: Mission progress
        """
        
        # 1. Normalized position (realistic - robot knows its position)
        norm_x = self.agent_pos[0] / self.grid.shape[0]
        norm_y = self.agent_pos[1] / self.grid.shape[1]
        
        # 2. Local environment sensing (realistic - robot has sensors)
        local_view = self._get_local_view_simple()
        
        # FIXED: Check specific directions for obstacles with correct indexing
        # local_view is 3x3 with agent at [1,1]
        # [0,1] = up, [2,1] = down, [1,0] = left, [1,2] = right
        down_clear = float(local_view[2, 1] == 0)   # Down direction (action 0)
        up_clear = float(local_view[0, 1] == 0)     # Up direction (action 1)  
        left_clear = float(local_view[1, 0] == 0)   # Left direction (action 2)
        right_clear = float(local_view[1, 2] == 0)  # Right direction (action 3)
        
        # 3. Mission tracking (realistic - robot tracks its assigned deliveries)
        target_positions = np.where(self.grid == 3)
        if self.initial_target_count > 0:
            current_targets = len(target_positions[0]) if len(target_positions[0]) > 0 else 0
            remaining_targets_norm = current_targets / self.initial_target_count
            progress = 1.0 - remaining_targets_norm
        else:
            remaining_targets_norm = 0
            progress = 1.0
        
        # Realistic 8D state vector with FIXED clearance mapping
        state_vector = np.array([
            # Position (2 features) - GPS/odometry
            norm_x, norm_y,
            
            # Local obstacle detection (4 features) - sensor data
            # ORDERED TO MATCH ACTION INDICES: [down, up, left, right]
            down_clear, up_clear, left_clear, right_clear,
            
            # Mission status (2 features) - internal tracking
            remaining_targets_norm, progress
        ], dtype=np.float32)
        
        return state_vector

    def _get_local_view_simple(self) -> np.ndarray:
        """Get simplified 3x3 local view around agent (simulates sensors)."""
        x, y = self.agent_pos
        local_view = np.ones((3, 3))  # Default to obstacles (conservative sensing)
        
        for i in range(3):
            for j in range(3):
                grid_x = x + i - 1
                grid_y = y + j - 1
                
                if (0 <= grid_x < self.grid.shape[0] and 
                    0 <= grid_y < self.grid.shape[1]):
                    local_view[i, j] = self.grid[grid_x, grid_y]
        
        return local_view

    def _enhanced_restaurant_reward(self, grid: np.ndarray, agent_pos: tuple[int, int], 
                                  prev_pos: tuple[int, int] = None) -> float:
        """Enhanced reward function for better DQN learning.
        
        Key improvements:
        - Less harsh time penalty (-0.5 instead of -1)
        - Reduced collision penalty (-5 instead of -10) 
        - Stronger target reward (30 instead of 15)
        - Progress-based rewards for dense feedback
        - Exploration bonus for visiting new areas
        """
        reward = 0.0
        
        # 1. Base rewards based on current cell
        if grid[agent_pos] == 0:  # Empty space
            reward += -0.5  # Reduced time penalty (was -1)
        elif grid[agent_pos] == 1 or grid[agent_pos] == 2:  # Collision
            reward += -5.0  # Reduced collision penalty (was -10)
            return reward  # Early return for collisions
        elif grid[agent_pos] == 3:  # Target reached
            reward += 30.0  # Increased target reward (was 15)
            return reward  # Early return for successful target
        else:
            reward += -0.5  # Default case
        
        # 2. PROGRESS-BASED REWARDS (only for valid movements)
        if prev_pos is not None and self.info["agent_moved"]:
            progress_reward = self._calculate_progress_reward(agent_pos, prev_pos)
            reward += progress_reward
        
        # 3. EXPLORATION BONUS
        if agent_pos not in self.visited_positions:
            reward += 0.5  # Small bonus for visiting new areas
            self.visited_positions.add(agent_pos)
        
        return reward

    def _calculate_progress_reward(self, current_pos: tuple[int, int], 
                                 prev_pos: tuple[int, int]) -> float:
        """Calculate reward based on progress toward nearest target."""
        # Find all current target positions
        target_positions = list(zip(*np.where(self.grid == 3)))
        
        if not target_positions:
            return 0.0  # No targets left
        
        # Calculate distance to nearest target
        current_distance = min(self._manhattan_distance(current_pos, target) 
                             for target in target_positions)
        
        # Initialize previous distance if needed
        if self.previous_distance_to_nearest is None:
            self.previous_distance_to_nearest = min(self._manhattan_distance(prev_pos, target) 
                                                   for target in target_positions)
        
        # Calculate progress
        progress = self.previous_distance_to_nearest - current_distance
        
        # Update tracking
        self.previous_distance_to_nearest = current_distance
        
        # Scale progress reward
        progress_scale = 5.0  # Scale factor for progress rewards
        max_grid_distance = max(self.grid.shape)
        
        if progress > 0:
            # Made progress toward target
            self.steps_without_progress = 0
            progress_reward = progress_scale * progress / max_grid_distance
        elif progress < 0:
            # Moved away from target  
            self.steps_without_progress += 1
            progress_reward = progress_scale * progress / max_grid_distance * 0.5  # Smaller penalty
        else:
            # No progress
            self.steps_without_progress += 1
            progress_reward = 0.0
            
            # Small penalty for staying in place too long
            if self.steps_without_progress > 15:
                progress_reward = -0.1
        
        return progress_reward
    
    def _manhattan_distance(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _simplified_restaurant_reward(self, grid: np.ndarray, agent_pos: tuple[int, int], 
                                    prev_pos: tuple[int, int] = None) -> float:
        """Original simplified reward function for restaurant delivery.
        
        Focus on core behaviors:
        - Time efficiency
        - Safety (collision avoidance)
        - Goal achievement
        """
        
        # Basic movement rewards
        if grid[agent_pos] == 0:  # Empty space
            reward = -1  # Time penalty for efficiency
        elif grid[agent_pos] == 1 or grid[agent_pos] == 2:  # Collision
            reward = -10  # Safety penalty
            return reward
        elif grid[agent_pos] == 3:  # Target reached
            reward = 15  # Goal achievement reward
        else:
            reward = -1  # Default case
        
        return reward

    @staticmethod
    def _default_reward_function(grid, agent_pos, prev_pos=None) -> float:
        """Original simple reward function for backward compatibility."""
        match grid[agent_pos]:
            case 0:  # Empty tile
                reward = -1
            case 1 | 2:  # Wall or obstacle
                reward = -5
            case 3:  # Target tile
                reward = 10
            case _:
                raise ValueError(f"Grid cell should not have value: {grid[agent_pos]}")
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
        """Evaluate agent performance."""

        env = Environment(grid_fp=grid_fp,
                          no_gui=True,
                          sigma=sigma,
                          agent_start_pos=agent_start_pos,
                          target_fps=-1,
                          random_seed=random_seed,
                          state_representation=state_representation)
        
        state = env.reset()
        initial_grid = np.copy(env.grid)
        agent_path = [env.agent_pos]

        for _ in trange(max_steps, desc="Evaluating agent"):
            action = agent.take_action(state)
            state, _, terminated, _ = env.step(action)
            agent_path.append(env.agent_pos)

            if terminated:
                break

        env.world_stats["targets_remaining"] = np.sum(env.grid == 3)

        path_image = visualize_path(initial_grid, agent_path)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        save_results(file_name, env.world_stats, path_image, show_images)