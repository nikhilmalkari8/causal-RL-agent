import torch
import gymnasium as gym
import numpy as np
from envs.causal_gridworld import CausalDoorEnv
from models.lstm_policy import LSTMPolicy
from agents.ppo_lstm_agent import PPOAgent

def preprocess_obs(obs):
    # obs shape: (5,5) integer grid
    # One-hot encode each cell (6 classes: 0-5)
    one_hot = np.eye(6)[obs]  # shape (5,5,6)
    return torch.tensor(one_hot.flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,150)

def train():
    env = CausalDoorEnv()
    obs, _ = env.reset()

    input_dim = 5*5*6  # one-hot flattened grid
    hidden_dim = 128
    action_dim = env.action_space.n

    policy = LSTMPolicy(input_dim, hidden_dim, action_dim)
    agent = PPOAgent(policy)

    max_episodes = 500
    max_steps = 30
    gamma = 0.99

    for episode in range(max_episodes):
        obs, _ = env.reset()
        hx = None

        states = []
        actions = []
        log_probs = []
        rewards = []
        masks = []
        values = []

        episode_reward = 0

        for step in range(max_steps):
            state_tensor = preprocess_obs(obs)
            action, log_prob, value, hx = agent.select_action(state_tensor, hx)

            next_obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            masks.append(1 - done)
            values.append(value)

            obs = next_obs
            if done or truncated:
                break

        next_state_tensor = preprocess_obs(obs)
        with torch.no_grad():
            _, next_value, _ = policy(next_state_tensor, hx)

        returns = agent.compute_returns(rewards, masks, values, next_value, gamma)

        trajectories = {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'returns': returns,
            'values': values
        }

        agent.ppo_update(trajectories)

        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    train()
