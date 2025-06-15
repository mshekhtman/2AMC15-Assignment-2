"""Fixed Heuristic Agent for realistic 8D continuous state space."""
import numpy as np
from agents import BaseAgent


class HeuristicAgent(BaseAgent):
    """Heuristic agent using realistic 8D continuous state features."""
    
    def __init__(self):
        # Update to use 8D state space instead of 10D
        super().__init__(state_dim=8, action_dim=4, state_type='continuous_vector')
        
    def take_action(self, state) -> int:
        """Make decisions based on 8D realistic state features.
        
        Strategy:
        1. Move towards target if path is clear
        2. Avoid obstacles
        3. Fallback to any clear direction
        
        8D State vector:
        [0-1]: Normalized position (x, y)
        [2-5]: Clear directions (front, left, right, back)
        [6]: Remaining targets (normalized)
        [7]: Mission progress
        """
        state = self.preprocess_state(state)
        
        # Extract features from 8D state vector
        # Position features [0-1]
        pos_x = state[0]
        pos_y = state[1]
        
        # Clearance features [2-5]
        front_clear = state[2]   # Is front clear (up)
        left_clear = state[3]    # Is left clear
        right_clear = state[4]   # Is right clear  
        back_clear = state[5]    # Is back clear (down)
        
        # Mission features [6-7]
        remaining_targets = state[6]  # Normalized remaining targets
        progress = state[7]           # Mission progress
        
        # Simple heuristic strategy:
        # 1. Prefer directions that are clear
        # 2. If multiple directions clear, choose randomly among them
        # 3. Encourage exploration when many targets remain
        
        available_actions = []
        action_names = ["down", "up", "left", "right"]
        clearance = [back_clear, front_clear, left_clear, right_clear]
        
        # Find all clear directions
        for i, is_clear in enumerate(clearance):
            if is_clear > 0.5:  # Threshold for "clear"
                available_actions.append(i)
        
        if available_actions:
            # If multiple actions available, add some intelligent preference
            if len(available_actions) > 1:
                # Prefer exploration if many targets remain
                if remaining_targets > 0.5:
                    # Prefer moving to unexplored areas (edges)
                    if pos_x < 0.3 and right_clear > 0.5:  # Near left edge, go right
                        return 3
                    elif pos_x > 0.7 and left_clear > 0.5:  # Near right edge, go left
                        return 2
                    elif pos_y < 0.3 and back_clear > 0.5:  # Near top, go down
                        return 0
                    elif pos_y > 0.7 and front_clear > 0.5:  # Near bottom, go up
                        return 1
            
            # Default: choose random available action
            return np.random.choice(available_actions)
        
        else:
            # No clear directions - emergency random action
            return np.random.randint(0, 4)
    
    def update(self, state, reward: float, action: int, next_state=None, done: bool = False):
        """Heuristic agent doesn't learn from experience."""
        pass