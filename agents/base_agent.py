"""Complete agents/base_agent.py for continuous state space.

We define the base class for all agents in this file with support for
continuous state representations.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    def __init__(self, state_dim: int = 15, action_dim: int = 4, state_type: str = 'continuous_vector'):
        """Base agent for continuous state space environments.

        This base class provides the interface that all RL agents must implement.
        It supports both continuous vector states and discrete position states
        for backward compatibility.

        Args:
            state_dim: Dimension of the continuous state vector. Default is 15
                      for the restaurant delivery robot state representation:
                      [norm_x, norm_y, target_dist, target_dir_x, target_dir_y,
                       remaining_targets, obstacle_density, front_clear, left_clear,
                       right_clear, back_clear, velocity_x, velocity_y, speed, progress]
            action_dim: Number of discrete actions available. Default is 4 for
                       movement actions: [down, up, left, right]
            state_type: Type of state representation this agent expects.
                       'continuous_vector' for 15D state vectors
                       'discrete' for backward compatibility with (x,y) positions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_type = state_type
        
        # Validate parameters
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if state_type not in ['continuous_vector', 'discrete']:
            raise ValueError(f"state_type must be 'continuous_vector' or 'discrete', got {state_type}")

    @abstractmethod
    def take_action(self, state) -> int:
        """Take action based on the current state.

        This method must be implemented by all agents. It should analyze the
        current state and return a discrete action.

        Args:
            state: Current state observation. Can be:
                  - np.ndarray of shape (state_dim,) for continuous vector states
                  - np.ndarray of shape (2,) for discrete position states  
                  - tuple (x, y) for discrete position states (backward compatibility)
                  
                  For restaurant delivery robot with continuous_vector:
                  state[0-1]: Normalized position (x, y) in [0, 1]
                  state[2]: Distance to nearest target, normalized
                  state[3-4]: Direction to nearest target (unit vector)
                  state[5]: Remaining targets, normalized
                  state[6]: Local obstacle density in [0, 1]
                  state[7-10]: Clear directions (binary: front, left, right, back)
                  state[11-12]: Velocity vector, normalized
                  state[13]: Speed (magnitude of velocity)
                  state[14]: Mission progress in [0, 1]

        Returns:
            action: Discrete action integer in [0, action_dim-1]
                   For movement: 0=down, 1=up, 2=left, 3=right
        """
        raise NotImplementedError("Subclasses must implement take_action method")
    
    @abstractmethod
    def update(self, state, reward: float, action: int, next_state=None, done: bool = False):
        """Update agent based on experience.

        This method is called after each action to allow the agent to learn
        from the experience. For non-learning agents (like random or heuristic),
        this can be a no-op.

        Args:
            state: Current state that action was taken from
            reward: Reward received for taking the action
            action: Action that was taken (should match what take_action returned)
            next_state: Next state after taking the action (optional, for Q-learning style updates)
            done: Whether the episode terminated after this action
        """
        raise NotImplementedError("Subclasses must implement update method")

    def preprocess_state(self, state) -> np.ndarray:
        """Preprocess state to ensure consistent format for the agent.
        
        This method handles different input formats and converts them to the
        expected state representation. It provides backward compatibility
        with discrete states while supporting new continuous states.
        
        Args:
            state: Input state in various formats:
                  - np.ndarray of shape (state_dim,): continuous vector (returned as-is)
                  - np.ndarray of shape (2,): discrete position, converted to continuous
                  - tuple (x, y): discrete position, converted to continuous
                  
        Returns:
            np.ndarray: Preprocessed state in the format expected by this agent
                       If agent expects continuous_vector: shape (state_dim,)
                       If agent expects discrete: shape (2,)
        """
        # Handle different input formats
        if isinstance(state, tuple):
            # Convert tuple (x, y) to numpy array
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, list):
            # Convert list to numpy array
            state = np.array(state, dtype=np.float32)
        elif not isinstance(state, np.ndarray):
            raise TypeError(f"State must be tuple, list, or numpy array, got {type(state)}")
        
        # Ensure float32 dtype
        state = state.astype(np.float32)
        
        # Handle different state representations
        if self.state_type == 'continuous_vector':
            if state.shape == (2,):
                # Convert discrete position to minimal continuous representation
                # This is for backward compatibility when using discrete positions
                # with continuous agents
                norm_x, norm_y = state / 10.0  # Assume max grid size of 10x10 for normalization
                # Create minimal state vector with position and zeros for other features
                minimal_state = np.zeros(self.state_dim, dtype=np.float32)
                minimal_state[0] = norm_x
                minimal_state[1] = norm_y
                return minimal_state
            elif state.shape == (self.state_dim,):
                # Already in correct continuous vector format
                return state
            else:
                raise ValueError(f"For continuous_vector agent, expected state shape (2,) or ({self.state_dim},), "
                               f"got {state.shape}")
        
        elif self.state_type == 'discrete':
            if state.shape == (2,):
                # Already in correct discrete format
                return state
            elif state.shape == (self.state_dim,):
                # Extract position from continuous vector (first 2 elements should be normalized position)
                # Convert back to discrete coordinates (assuming 10x10 grid)
                discrete_pos = (state[0:2] * 10.0).astype(np.int32)
                return discrete_pos.astype(np.float32)
            else:
                raise ValueError(f"For discrete agent, expected state shape (2,) or ({self.state_dim},), "
                               f"got {state.shape}")
        
        else:
            raise ValueError(f"Unknown state_type: {self.state_type}")

    def get_state_info(self, state) -> dict:
        """Extract interpretable information from state for debugging/logging.
        
        This utility method helps with debugging by extracting human-readable
        information from the state vector.
        
        Args:
            state: State vector (will be preprocessed)
            
        Returns:
            dict: Dictionary with interpretable state information
        """
        state = self.preprocess_state(state)
        
        if self.state_type == 'continuous_vector' and len(state) >= 15:
            return {
                'position': (state[0], state[1]),
                'position_grid': (state[0] * 10, state[1] * 10),  # Assuming 10x10 grid
                'target_distance': state[2],
                'target_direction': (state[3], state[4]),
                'remaining_targets': state[5],
                'obstacle_density': state[6],
                'clear_directions': {
                    'front': bool(state[7]),
                    'left': bool(state[8]),
                    'right': bool(state[9]),
                    'back': bool(state[10])
                },
                'velocity': (state[11], state[12]),
                'speed': state[13],
                'progress': state[14]
            }
        elif self.state_type == 'discrete':
            return {
                'position': (state[0], state[1]),
                'position_grid': (int(state[0]), int(state[1]))
            }
        else:
            return {'raw_state': state.tolist()}

    def validate_action(self, action: int) -> int:
        """Validate and clip action to valid range.
        
        Ensures the action is within the valid action space. Useful for
        preventing errors from agents that might output invalid actions.
        
        Args:
            action: Raw action from agent
            
        Returns:
            int: Validated action in range [0, action_dim-1]
        """
        if not isinstance(action, (int, np.integer)):
            # Try to convert to int
            try:
                action = int(action)
            except (ValueError, TypeError):
                raise TypeError(f"Action must be convertible to int, got {type(action)}")
        
        # Clip to valid range
        action = max(0, min(action, self.action_dim - 1))
        return action

    def get_action_name(self, action: int) -> str:
        """Get human-readable name for action.
        
        Args:
            action: Action integer
            
        Returns:
            str: Human-readable action name
        """
        action_names = {
            0: "down",
            1: "up", 
            2: "left",
            3: "right"
        }
        return action_names.get(action, f"unknown_action_{action}")

    def reset_episode(self):
        """Reset any episode-specific state in the agent.
        
        This method is called at the beginning of each episode and can be
        overridden by subclasses that maintain episode-specific state
        (like exploration parameters, episode memories, etc.).
        
        Default implementation does nothing.
        """
        pass

    def save_agent(self, filepath: str):
        """Save agent state to file.
        
        This method can be overridden by learning agents to save their
        learned parameters (neural network weights, Q-tables, etc.).
        
        Args:
            filepath: Path where to save the agent
        """
        import pickle
        
        # Save basic agent configuration
        agent_data = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'state_type': self.state_type,
            'class_name': self.__class__.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"Basic agent configuration saved to {filepath}")

    def load_agent(self, filepath: str):
        """Load agent state from file.
        
        This method can be overridden by learning agents to load their
        learned parameters.
        
        Args:
            filepath: Path to load the agent from
        """
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
            
            # Verify compatibility
            if agent_data.get('state_dim') != self.state_dim:
                print(f"Warning: Loaded state_dim {agent_data.get('state_dim')} != current {self.state_dim}")
            if agent_data.get('action_dim') != self.action_dim:
                print(f"Warning: Loaded action_dim {agent_data.get('action_dim')} != current {self.action_dim}")
                
            print(f"Basic agent configuration loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"No saved agent found at {filepath}")
        except Exception as e:
            print(f"Error loading agent: {e}")

    def __str__(self) -> str:
        """String representation of the agent."""
        return (f"{self.__class__.__name__}("
                f"state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"state_type='{self.state_type}')")

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return self.__str__()