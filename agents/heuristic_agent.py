"""FIXED Heuristic Agent for realistic 8D continuous state space."""
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
        [2-5]: Clear directions (down, up, left, right) - FIXED ORDER
        [6]: Remaining targets (normalized)
        [7]: Mission progress
        """
        state = self.preprocess_state(state)
        
        # Extract features from 8D state vector
        # Position features [0-1]
        pos_x = state[0]
        pos_y = state[1]
        
        # FIXED: Clearance features [2-5] with correct mapping
        # state[2] = down_clear (action 0)
        # state[3] = up_clear (action 1)
        # state[4] = left_clear (action 2)  
        # state[5] = right_clear (action 3)
        clearance_values = state[2:6]  # [down, up, left, right]
        
        # Mission features [6-7]
        remaining_targets = state[6]  # Normalized remaining targets
        progress = state[7]           # Mission progress
        
        # Find all clear directions (actions with clearance > 0.5)
        available_actions = []
        for action_id, is_clear in enumerate(clearance_values):
            if is_clear > 0.5:  # Threshold for "clear"
                available_actions.append(action_id)
        
        if available_actions:
            # ENHANCED STRATEGY: Intelligent action selection
            if len(available_actions) > 1:
                # Strategy 1: Exploration preference when many targets remain
                if remaining_targets > 0.5:
                    # Prefer moving to unexplored areas (towards edges)
                    if pos_x < 0.3 and 3 in available_actions:  # Near left edge, go right
                        return 3
                    elif pos_x > 0.7 and 2 in available_actions:  # Near right edge, go left
                        return 2
                    elif pos_y < 0.3 and 0 in available_actions:  # Near top, go down
                        return 0
                    elif pos_y > 0.7 and 1 in available_actions:  # Near bottom, go up
                        return 1
                
                # Strategy 2: Center-seeking when progress is low
                elif progress < 0.3:
                    center_x, center_y = 0.5, 0.5
                    
                    # Move towards center
                    if pos_x < center_x and 3 in available_actions:  # Go right towards center
                        return 3
                    elif pos_x > center_x and 2 in available_actions:  # Go left towards center
                        return 2
                    elif pos_y < center_y and 0 in available_actions:  # Go down towards center
                        return 0
                    elif pos_y > center_y and 1 in available_actions:  # Go up towards center
                        return 1
            
            # Default: choose random available action
            return np.random.choice(available_actions)
        
        else:
            # No clear directions - emergency random action
            # This should rarely happen if clearance detection works correctly
            print(f"WARNING: No clear directions available! State: {state}")
            return np.random.randint(0, 4)
    
    def update(self, state, reward: float, action: int, next_state=None, done: bool = False):
        """Heuristic agent doesn't learn from experience."""
        pass
    
    def get_strategy_info(self, state):
        """Debug method to understand agent's decision process."""
        state = self.preprocess_state(state)
        
        info = {
            'position': (state[0], state[1]),
            'clearance': {
                'down': state[2],
                'up': state[3], 
                'left': state[4],
                'right': state[5]
            },
            'remaining_targets': state[6],
            'progress': state[7],
            'available_actions': [i for i, clear in enumerate(state[2:6]) if clear > 0.5]
        }
        
        return info