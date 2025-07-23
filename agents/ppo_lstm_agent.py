import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, policy, lr=3e-4, gamma=0.99, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state, hx):
        # state shape: (1, seq_len=1, input_dim)
        logits, value, hx = self.policy(state, hx)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value, hx

    def compute_returns(self, rewards, masks, values, next_value, gamma):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def ppo_update(self, trajectories, epochs=4, batch_size=64):
        # trajectories is a dict with states, actions, log_probs, returns, values

        states = torch.cat(trajectories['states'])
        actions = torch.tensor(trajectories['actions']).to(states.device)
        old_log_probs = torch.cat(trajectories['log_probs']).detach()
        returns = torch.tensor(trajectories['returns']).to(states.device)
        values = torch.cat(trajectories['values']).detach()

        advantages = returns - values.squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        for _ in range(epochs):
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_old_log_probs = old_log_probs[start:end]
                batch_returns = returns[start:end]
                batch_advantages = advantages[start:end]

                logits, value, _ = self.policy(batch_states)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(batch_actions)

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value.squeeze(-1), batch_returns)
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
