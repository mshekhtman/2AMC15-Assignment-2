"""Simplified Random Agent for 10D continuous states."""
from random import randint
import numpy as np
from agents import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that performs random actions with simplified state handling."""
    
    def __init__(self):
        super().__init__(state_dim=10, action_dim=4, state_type='continuous_vector')
    
    def take_action(self, state) -> int:
        """Take random action regardless of state."""
        return randint(0, 3)
    
    def update(self, state, reward: float, action: int, next_state=None, done: bool = False):
        """Random agent doesn't learn."""
        pass
