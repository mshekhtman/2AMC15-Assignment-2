import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.ppo_policy import PPOPolicy 

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    This agent uses a policy network to select actions based on the current state.
    It maintains a memory of experiences to update the policy.
    """

    def _init_(self, state_dim, action_dim, lr=2.5e-4, gamma=0.99, clip=0.2, entropy_coef=0.01):

        """Initialize the PPO agent with the given parameters.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            clip (float): Clipping parameter for PPO.
            entropy_coef (float): Coefficient for entropy regularization.
        """
        self.policy = PPOPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip
        self.entropy_coef = entropy_coef
        self.memory = []

    def take_action(self, state):

        """Select an action based on the current state using the policy network.
        Args:
            state (np.array): Current state of the environment.
        Returns:
            int: Selected action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        self.memory.append({
            'state': state_tensor,
            'action': action,
            'log_prob': dist.log_prob(action),
            'entropy': dist.entropy()
        })

        return action.item()
    
    def update(self, state, reward, next_state, done):
        """Update the agent's memory with the latest experience.
        Args:
            state (np.array): Current state of the environment.
            reward (float): Reward received from the environment.
            next_state (np.array): Next state of the environment.
            done (bool): Whether the episode has ended.
        """
       
        self.memory[-1]['reward'] = reward
        
        self.memory[-1]['done'] = done

   

    def learn(self):
        pass

    def reset_memory(self):
        """Reset the agent's memory."""
        self.memory=[]