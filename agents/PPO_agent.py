import numpy as np
import torch
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.PPO_policy import PPOPolicy


class PPOAgent(BaseAgent):
    """
    Clipped-surrogate Proximal Policy Optimization (PPO) agent.
    This agent uses a shared actor-critic network to select actions based on the current state.
    It maintains a memory of experiences to update the policy.
    """

    def __init__(
            self,
            state_dim: int = 10,
            action_dim: int = 4,
            state_type: str = 'continuous_vector',
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            rollout_steps: int = 256,
            ppo_epochs: int = 4,
            batch_size: int = 64,
    ):

        """Initialize the PPO agent with the given parameters.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            state_type (str): Type of the state space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float, optional): GAE lambda.
            clip_eps (float, optional): PPO clip range.
            entropy_coef (float, optional): Coefficient for entropy regularization.
            value_coef (float, optional): Critic loss weight.
            max_grad_norm (float, optional): Gradient-clip norm.
            rollout_steps (int, optional): Steps per on-policy roll-out.
            ppo_epochs (int, optional): SGD epochs per update.
            batch_size (int, optional): Minibatch size.
        """
        super().__init__(state_dim, action_dim, state_type)

        # Hyperparameters
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Entropy linear decay, removed for now
        #self.entropy_start = entropy_coef
        #self.entropy_end = 0.00
        #self.decay_updates = 400  # PPO updates over which to decay
        #self.updates_done = 0  # update counter

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # On-policy experience buffer (memory)
        self.buffer = []

    @torch.no_grad()
    def take_action(self, state):
        """Select an action based on the current state using the shared actor-critic network.
        Args:
            state (np.array): Current state of the environment.
        Returns:
            int: Selected action (for env.step).
        """
        state_np = np.asarray(state, dtype=np.float32)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        action, logp, value = self.policy.act(state_t)

        self.buffer.append(
            {
                "state": state_np,
                "action": action,
                "logp": logp.cpu().item(),
                "value": value.cpu().item(),
            }
        )
        return action

    def update(self, state, reward, action, next_state=None, done=False):
        """Update the agent's memory with the latest experience.
        Args:
            state (np.array): Current state of the environment (unused).
            reward (float): Reward received from the environment.
            action (int): Action taken (unused).
            next_state (np.array): Next state of the environment (unused).
            done (bool): Whether the episode has ended.
        """
        self.buffer[-1]["reward"] = reward
        self.buffer[-1]["done"] = done

        if done or len(self.buffer) >= self.rollout_steps:
            if len(self.buffer) >= 8: # at least 8 steps
                self._learn()
            self.buffer.clear() # reset memory

    def _learn(self):
        """Compute GAE advantages and run the PPO clipped-surrogate update."""
        # Convert buffer to tensors
        states = torch.tensor(
            np.stack([b["state"] for b in self.buffer]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [b["action"] for b in self.buffer], dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            [b["reward"] for b in self.buffer], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [b["done"] for b in self.buffer], dtype=torch.float32, device=self.device
        )
        old_logp = torch.tensor(
            [b["logp"] for b in self.buffer], dtype=torch.float32, device=self.device
        )
        values = torch.tensor(
            [b["value"] for b in self.buffer], dtype=torch.float32, device=self.device
        )

        # Bootstrap the last state value if rollout ended mid-episode
        with torch.no_grad():
            if self.buffer[-1]["done"]:
                next_v = 0.0  # episode finished
            else:
                last_state = states[-1].unsqueeze(0)
                _, next_v = self.policy.forward(last_state)
                next_v = next_v.item()

        # Generalised Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * mask - values[t]
            gae = delta + self.gamma * self.lmbda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_v = values[t]

        adv_mean = advantages.mean()
        adv_std = advantages.std()

        # If the rollout is only 1–2 steps, std can be 0; avoid division by zero
        if adv_std < 1e-8:
            advantages = advantages - adv_mean
        else:
            advantages = (advantages - adv_mean) / adv_std

        # PPO clipped-objective optimization
        effective_bs = min(self.batch_size, states.size(0))
        idxs = np.arange(states.size(0))
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)

            for start in range(0, len(idxs), effective_bs):
                mb = torch.tensor(idxs[start: start + effective_bs], device=self.device)

                new_logp, entropy, new_values = self.policy.evaluate(states[mb], actions[mb])
                ratio = torch.exp(new_logp - old_logp[mb])

                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[mb] - new_values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

        print(f"entropy = {entropy.mean().item():.4f}")

    '''def _entropy_coef(self):
        """Linearly decays entropy coefficient from start to end."""
        t = min(1.0, self.updates_done / self.decay_updates)
        return self.entropy_start * (1 - t) + self.entropy_end * t'''
