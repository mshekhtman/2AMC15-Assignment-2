import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from agents.base_agent import BaseAgent
from agents.DQN.DQN_nn import DQNetwork  # Assuming you named your network class DQNetwork in DQN_nn.py

class DQNAgent(BaseAgent):
    def __init__(self, 
                 state_dim=15, 
                 action_dim=4, 
                 state_type='continuous_vector',
                 gamma=0.99, 
                 lr=1e-3, 
                 batch_size=64, 
                 buffer_size=100000, 
                 min_replay_size=1000, 
                 target_update_freq=1000):
        
        super().__init__(state_dim, action_dim, state_type)
        
        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Internal step counter
        self.training_steps = 0
        
    def take_action(self, state, training = True):
        state = self.preprocess_state(state)
        if training and (np.random.rand() < self.epsilon):
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            return action
        
    def update(self, state, reward, action, next_state=None, done=False):
        # Store transition
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Don't train until minimum replay size reached
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.q_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_agent(self, filepath: str):
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"DQNAgent saved to {filepath}")

    def load_agent(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['q_net_state_dict'])  # Optional: sync target net too
        self.epsilon = checkpoint.get('epsilon', 1.0)
        print(f"DQNAgent loaded from {filepath}")
