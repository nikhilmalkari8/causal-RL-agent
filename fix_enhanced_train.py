import torch
import gymnasium as gym
import numpy as np
import random
from collections import deque
from envs.causal_gridworld import CausalDoorEnv
from models.lstm_policy import EnhancedLSTMPolicy, EnhancedPPOAgent, CuriosityDrivenRewardShaper

def fixed_enhanced_train():
    env = CausalDoorEnv()
    obs, _ = env.reset()

    input_dim = 5*5*6
    hidden_dim = 256
    action_dim = env.action_space.n

    policy = EnhancedLSTMPolicy(input_dim, hidden_dim, action_dim, dropout=0.1)
    agent = EnhancedPPOAgent(policy, lr=1e-4, clip_epsilon=0.15, entropy_coef=0.08)
    
    curiosity = CuriosityDrivenRewardShaper()
    max_episodes = 500  # Reduced for testing
    max_steps = 50
    gamma = 0.95
    
    recent_rewards = deque(maxlen=100)
    
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
            state_tensor = torch.tensor(np.eye(6)[obs].flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            action, log_prob, value, hx = agent.select_action(state_tensor, hx)
            next_obs, env_reward, done, truncated, _ = env.step(action)
            
            # Enhanced reward shaping
            shaped_reward = float(env_reward)  # Ensure float type
            if env_reward > 0:
                shaped_reward += 1.0
            
            episode_reward += env_reward

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(shaped_reward)
            masks.append(float(1 - done))  # Ensure float type
            values.append(value)

            obs = next_obs
            if done or truncated:
                break

        # Fixed returns computation
        next_state_tensor = torch.tensor(np.eye(6)[obs].flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, next_value, _ = policy(next_state_tensor, hx)

        returns = agent.compute_gae(rewards, masks, [v.squeeze().item() for v in values], 
                                  next_value.squeeze().item(), gamma, lambda_=0.95)

        trajectories = {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'returns': [float(r) for r in returns],  # Ensure float type
            'values': values
        }

        agent.ppo_update(trajectories, epochs=3, batch_size=16)

        recent_rewards.append(episode_reward)

        if len(recent_rewards) >= 50:
            success_rate = sum(1 for r in list(recent_rewards)[-50:] if r > 0) / 50
            
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode+1}: Success Rate: {success_rate:.3f}")
            
            if success_rate > 0.8:
                print(f"Training successful! Success rate: {success_rate:.3f}")
                torch.save(policy.state_dict(), f'enhanced_policy_{success_rate:.3f}.pth')
                break

    return policy

if __name__ == "__main__":
    policy = fixed_enhanced_train()
