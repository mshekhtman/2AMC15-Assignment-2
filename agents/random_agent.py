"""Random Agent that works with continuous states."""
from random import randint
import numpy as np
from agents import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that performs random actions but handles continuous states."""
    
    def __init__(self):
        super().__init__(state_dim=15, action_dim=4)
    
    def take_action(self, state: np.ndarray) -> int:
        """Take random action regardless of state."""
        return randint(0, 3)
    
    def update(self, state: np.ndarray, reward: float, action: int, next_state: np.ndarray = None, done: bool = False):
        """Random agent doesn't learn."""
        pass