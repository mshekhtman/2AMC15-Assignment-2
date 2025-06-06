import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class Logger:
    def __init__(self, 
                 grid,
                 sigma,
                 gamma=0.99, 
                 lr=1e-3, 
                 batch_size=64, 
                 buffer_size=50000,
                 min_replay_size=1000, 
                 target_update_freq=500,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):
    
        self.grid = grid
        self.sigma = sigma
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_rewards = []
        self.DQN_rewards = []
    
    def log_target_rewards(self, reward):
        self.target_rewards.append(reward)
    
    def plot_target_rewards(self):       
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        filename = f"{timestamp}_targetrewardsplot.png"
        plt.figure(figsize=(10, 5))
        plt.plot(self.target_rewards, label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Target ewards Over Episodes')
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.close()