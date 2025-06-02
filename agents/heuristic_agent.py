"""Heuristic agent that uses continuous state features intelligently."""
import numpy as np
from agents import BaseAgent


class HeuristicAgent(BaseAgent):
    """Agent that uses heuristics based on continuous state features."""
    
    def __init__(self):
        super().__init__(state_dim=15, action_dim=4)
    
    def take_action(self, state: np.ndarray) -> int:
        """Make decisions based on state features."""
        state = self.preprocess_state(state)
        
        # Extract relevant features (based on our state vector design)
        target_dir_x = state[3]  # Direction to target (x)
        target_dir_y = state[4]  # Direction to target (y)
        front_clear = state[9]   # Is front clear
        left_clear = state[10]   # Is left clear
        right_clear = state[11]  # Is right clear
        back_clear = state[12]   # Is back clear
        
        # Priority: move towards target if path is clear
        if target_dir_y > 0.5 and front_clear:  # Target is below and front is clear
            return 0  # Move down
        elif target_dir_y < -0.5 and back_clear:  # Target is above and back is clear
            return 1  # Move up
        elif target_dir_x < -0.5 and left_clear:  # Target is left and left is clear
            return 2  # Move left
        elif target_dir_x > 0.5 and right_clear:  # Target is right and right is clear
            return 3  # Move right
        
        # Fallback: move in any clear direction
        if front_clear:
            return 0
        elif right_clear:
            return 3
        elif left_clear:
            return 2
        elif back_clear:
            return 1
        
        # Last resort: random move (shouldn't happen often)
        return np.random.randint(0, 4)
    
    def update(self, state: np.ndarray, reward: float, action: int, next_state: np.ndarray = None, done: bool = False):
        """Heuristic agent doesn't learn from experience."""
        pass