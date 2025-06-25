"""FIXED Heuristic Agent for realistic 8D continuous state space - CLEAN VERSION."""
import numpy as np
from agents import BaseAgent


class HeuristicAgent(BaseAgent):
    """Heuristic agent using realistic 8D continuous state features."""
    
    def __init__(self):
        super().__init__(state_dim=8, action_dim=4, state_type='continuous_vector')
        
    def take_action(self, state) -> int:
        """Make decisions based on 8D realistic state features.
        
        Strategy:
        1. Safety first: Only choose clear directions
        2. Intelligent navigation based on position and mission status
        3. Robust fallback handling
        
        8D State vector:
        [0-1]: Normalized position (x, y)
        [2-5]: Clear directions (down, up, left, right)
        [6]: Remaining targets (normalized)
        [7]: Mission progress
        """
        state = self.preprocess_state(state)
        
        # Extract features
        pos_x = state[0]
        pos_y = state[1]
        down_clear = state[2]   # Action 0
        up_clear = state[3]     # Action 1  
        left_clear = state[4]   # Action 2
        right_clear = state[5]  # Action 3
        remaining_targets = state[6]
        progress = state[7]
        
        # Find safe directions with conservative threshold
        clearance_threshold = 0.8
        safe_actions = []
        clearance_values = [down_clear, up_clear, left_clear, right_clear]
        
        for action_id, clearance in enumerate(clearance_values):
            if clearance > clearance_threshold:
                safe_actions.append(action_id)
        
        # Emergency fallback with lower threshold
        if not safe_actions:
            emergency_threshold = 0.5
            for action_id, clearance in enumerate(clearance_values):
                if clearance > emergency_threshold:
                    safe_actions.append(action_id)
        
        # Final fallback: choose best available
        if not safe_actions:
            return int(np.argmax(clearance_values))
        
        # Single safe option
        if len(safe_actions) == 1:
            return safe_actions[0]
        
        # Multiple safe options - choose intelligently
        
        # Early exploration phase
        if remaining_targets > 0.7:
            if pos_x < 0.25 and 3 in safe_actions:  # Near left edge, go right
                return 3
            elif pos_x > 0.75 and 2 in safe_actions:  # Near right edge, go left
                return 2
            elif pos_y < 0.25 and 0 in safe_actions:  # Near top, go down
                return 0
            elif pos_y > 0.75 and 1 in safe_actions:  # Near bottom, go up
                return 1
        
        # Center-seeking phase
        elif remaining_targets > 0.3:
            center_x, center_y = 0.5, 0.5
            if pos_x < center_x - 0.05 and 3 in safe_actions:
                return 3
            elif pos_x > center_x + 0.05 and 2 in safe_actions:
                return 2
            elif pos_y < center_y - 0.05 and 0 in safe_actions:
                return 0
            elif pos_y > center_y + 0.05 and 1 in safe_actions:
                return 1
        
        # Cleanup phase - systematic search
        else:
            if pos_y < 0.5:  # Upper half - sweep right
                if 3 in safe_actions:
                    return 3
                elif 0 in safe_actions:
                    return 0
            else:  # Lower half - sweep left
                if 2 in safe_actions:
                    return 2
                elif 1 in safe_actions:
                    return 1
        
        # Random safe action as final fallback
        return np.random.choice(safe_actions)
    
    def update(self, state, reward: float, action: int, next_state=None, done: bool = False):
        """Heuristic agent doesn't learn from experience."""
        pass