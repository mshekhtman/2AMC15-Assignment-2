"""Updated agents/base_agent.py for realistic 8D continuous state space."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    def __init__(self, state_dim: int = 8, action_dim: int = 4, state_type: str = 'continuous_vector'):
        """Base agent for realistic continuous state space.

        Args:
            state_dim: Dimension of the continuous state vector (8 for realistic restaurant robot)
            action_dim: Number of discrete actions (4 for up/down/left/right)
            state_type: Type of state representation ('continuous_vector', 'discrete')
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

        Args:
            state: Current state observation. Can be:
                  - np.ndarray of shape (state_dim,) for continuous vector states
                  - np.ndarray of shape (2,) for discrete position states  
                  - tuple (x, y) for discrete position states (backward compatibility)
                  
                  For realistic restaurant delivery robot with continuous_vector (8D):
                  state[0-1]: Normalized position (x, y) in [0, 1]
                  state[2-5]: Clear directions (binary: front, left, right, back)
                  state[6]: Remaining targets, normalized
                  state[7]: Mission progress in [0, 1]

        Returns:
            action: Discrete action integer in [0, action_dim-1]
                   For movement: 0=down, 1=up, 2=left, 3=right
        """
        raise NotImplementedError("Subclasses must implement take_action method")
    
    @abstractmethod
    def update(self, state, reward: float, action: int, next_state=None, done: bool = False):
        """Update agent based on experience.

        Args:
            state: Current state that action was taken from
            reward: Reward received for taking the action
            action: Action that was taken
            next_state: Next state after taking the action
            done: Whether the episode terminated after this action
        """
        raise NotImplementedError("Subclasses must implement update method")

    def preprocess_state(self, state) -> np.ndarray:
        """Preprocess state to ensure consistent format for the agent."""
        # Handle different input formats
        if isinstance(state, tuple):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        elif not isinstance(state, np.ndarray):
            raise TypeError(f"State must be tuple, list, or numpy array, got {type(state)}")
        
        state = state.astype(np.float32)
        
        # Handle different state representations
        if self.state_type == 'continuous_vector':
            if state.shape == (2,):
                # Convert discrete position to minimal continuous representation
                norm_x, norm_y = state / 10.0  # Assume max grid size of 10x10
                minimal_state = np.zeros(self.state_dim, dtype=np.float32)
                minimal_state[0] = norm_x
                minimal_state[1] = norm_y
                return minimal_state
            elif state.shape == (self.state_dim,):
                return state
            else:
                raise ValueError(f"For continuous_vector agent, expected state shape (2,) or ({self.state_dim},), "
                               f"got {state.shape}")
        
        elif self.state_type == 'discrete':
            if state.shape == (2,):
                return state
            elif state.shape == (self.state_dim,):
                # Extract position from continuous vector
                discrete_pos = (state[0:2] * 10.0).astype(np.int32)
                return discrete_pos.astype(np.float32)
            else:
                raise ValueError(f"For discrete agent, expected state shape (2,) or ({self.state_dim},), "
                               f"got {state.shape}")
        
        else:
            raise ValueError(f"Unknown state_type: {self.state_type}")

    def get_state_info(self, state) -> dict:
        """Extract interpretable information from state for debugging."""
        state = self.preprocess_state(state)
        
        if self.state_type == 'continuous_vector' and len(state) >= 8:
            return {
                'position': (state[0], state[1]),
                'position_grid': (state[0] * 10, state[1] * 10),
                'clear_directions': {
                    'front': bool(state[2]),
                    'left': bool(state[3]),
                    'right': bool(state[4]),
                    'back': bool(state[5])
                },
                'remaining_targets': state[6],
                'progress': state[7]
            }
        elif self.state_type == 'discrete':
            return {
                'position': (state[0], state[1]),
                'position_grid': (int(state[0]), int(state[1]))
            }
        else:
            return {'raw_state': state.tolist()}

    def validate_action(self, action: int) -> int:
        """Validate and clip action to valid range."""
        if not isinstance(action, (int, np.integer)):
            try:
                action = int(action)
            except (ValueError, TypeError):
                raise TypeError(f"Action must be convertible to int, got {type(action)}")
        
        action = max(0, min(action, self.action_dim - 1))
        return action

    def get_action_name(self, action: int) -> str:
        """Get human-readable name for action."""
        action_names = {0: "down", 1: "up", 2: "left", 3: "right"}
        return action_names.get(action, f"unknown_action_{action}")

    def reset_episode(self):
        """Reset episode-specific state. Override in subclasses if needed."""
        pass

    def save_agent(self, filepath: str):
        """Save basic agent configuration."""
        import pickle
        
        agent_data = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'state_type': self.state_type,
            'class_name': self.__class__.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"Agent configuration saved to {filepath}")

    def load_agent(self, filepath: str):
        """Load basic agent configuration."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
            
            if agent_data.get('state_dim') != self.state_dim:
                print(f"Warning: Loaded state_dim {agent_data.get('state_dim')} != current {self.state_dim}")
            if agent_data.get('action_dim') != self.action_dim:
                print(f"Warning: Loaded action_dim {agent_data.get('action_dim')} != current {self.action_dim}")
                
            print(f"Agent configuration loaded from {filepath}")
            
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