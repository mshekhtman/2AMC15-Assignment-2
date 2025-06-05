"""Simplified Heuristic Agent for Restaurant Delivery Robot."""
import numpy as np
from agents import BaseAgent


class HeuristicAgent(BaseAgent):
    """Simplified agent using basic continuous state features."""
    
    def __init__(self):
        super().__init__(state_dim=10, action_dim=4, state_type='continuous_vector')
        
    def take_action(self, state) -> int:
        """Make decisions based on simplified state features.
        
        Strategy:
        1. Move towards target if path is clear
        2. Avoid obstacles
        3. Fallback to any clear direction
        """
        state = self.preprocess_state(state)
        
        # Extract features from simplified 10D state vector
        target_dir_x = state[2]  # Direction to target (x)
        target_dir_y = state[3]  # Direction to target (y)
        front_clear = state[5]   # Is front clear (up)
        left_clear = state[6]    # Is left clear
        right_clear = state[7]   # Is right clear  
        back_clear = state[8]    # Is back clear (down)
        
        # Primary strategy: move towards target if path is clear
        if target_dir_y > 0.3 and back_clear:  # Target is below and back is clear
            return 0  # Move down
        elif target_dir_y < -0.3 and front_clear:  # Target is above and front is clear
            return 1  # Move up
        elif target_dir_x < -0.3 and left_clear:  # Target is left and left is clear
            return 2  # Move left
        elif target_dir_x > 0.3 and right_clear:  # Target is right and right is clear
            return 3  # Move right
        
        # Fallback: move in any clear direction
        if front_clear:
            return 1
        elif right_clear:
            return 3
        elif left_clear:
            return 2
        elif back_clear:
            return 0
        
        # Last resort: random move
        return np.random.randint(0, 4)
    
    def update(self, state, reward: float, action: int, next_state=None, done: bool = False):
        """Heuristic agent doesn't learn from experience."""
        pass