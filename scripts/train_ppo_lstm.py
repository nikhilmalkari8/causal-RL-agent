import torch
import gymnasium as gym
import numpy as np
import random
from envs.causal_gridworld import CausalDoorEnv
from models.lstm_policy import LSTMPolicy
from agents.ppo_lstm_agent import PPOAgent

def preprocess_obs(obs):
    one_hot = np.eye(6)[obs]
    return torch.tensor(one_hot.flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def train():
    env = CausalDoorEnv()
    obs, _ = env.reset()

    input_dim = 5*5*6
    hidden_dim = 256  # Increased capacity
    action_dim = env.action_space.n

    policy = LSTMPolicy(input_dim, hidden_dim, action_dim)
    
    # Reduced learning rate for stability
    agent = PPOAgent(policy, lr=1e-4, clip_epsilon=0.15, entropy_coef=0.05)

    max_episodes = 2000
    max_steps = 50  # More steps to find solution
    gamma = 0.95
    
    # Experience replay buffer to prevent forgetting
    replay_buffer = []
    buffer_size = 500
    
    # Success tracking
    recent_rewards = []
    best_success_rate = 0
    
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
        trajectory_states = []
        trajectory_actions = []

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
            
            # Track trajectory for replay
            trajectory_states.append(obs.copy())
            trajectory_actions.append(action)

            obs = next_obs
            if done or truncated:
                break

        # Enhanced reward shaping
        if episode_reward > 0:
            # Bonus for successful episodes
            for i in range(len(rewards)):
                if rewards[i] == 0:
                    rewards[i] = 0.01  # Small positive reward for each step in successful episode
        else:
            # Check if agent at least visited switch
            visited_switch = any(state[0, 4] == 1 for state in trajectory_states)  # Check if agent was at switch position
            if visited_switch:
                rewards[-1] = 0.1  # Small reward for visiting switch even if didn't reach goal

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

        # Store successful episodes in replay buffer
        if episode_reward > 0:
            replay_buffer.append(trajectories)
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)

        # Update with current episode
        agent.ppo_update(trajectories, epochs=3, batch_size=32)

        # Replay successful episodes periodically
        if len(replay_buffer) > 5 and episode % 5 == 0:
            # Sample a successful episode to replay
            replay_traj = random.choice(replay_buffer)
            agent.ppo_update(replay_traj, epochs=2, batch_size=16)

        # Track performance
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Calculate success rate
        if len(recent_rewards) >= 50:
            success_rate = sum(1 for r in recent_rewards[-50:] if r > 0) / 50
            avg_reward = np.mean(recent_rewards[-50:])
            
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode+1}: Success Rate: {success_rate:.3f}, "
                      f"Avg Reward: {avg_reward:.3f}, Buffer Size: {len(replay_buffer)}")
            
            # Save best model
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save(policy.state_dict(), f'best_policy_{success_rate:.3f}.pth')
            
            # Early stopping if consistently good
            if success_rate > 0.8:
                print(f"Training successful! Success rate: {success_rate:.3f}")
                break
        else:
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode+1}: Avg Reward: {avg_reward:.3f}, Buffer Size: {len(replay_buffer)}")

    print(f"Training completed. Best success rate: {best_success_rate:.3f}")

def test_trained_model(model_path):
    """Test the trained model"""
    env = CausalDoorEnv()
    
    input_dim = 5*5*6
    hidden_dim = 256
    action_dim = env.action_space.n
    
    policy = LSTMPolicy(input_dim, hidden_dim, action_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    
    print("\n=== Testing Trained Model ===")
    
    for test_episode in range(5):
        obs, _ = env.reset()
        hx = None
        total_reward = 0
        
        print(f"\nTest Episode {test_episode + 1}:")
        env.render()
        
        for step in range(50):
            state_tensor = preprocess_obs(obs)
            
            with torch.no_grad():
                logits, _, hx = policy(state_tensor, hx)
                action = torch.argmax(logits, dim=-1).item()
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            env.render()
            
            if done or truncated:
                break
        
        print(f"Test Episode {test_episode + 1} Total Reward: {total_reward}")

if __name__ == "__main__":
    train()
    
    # Test the best model if it exists
    import os
    best_models = [f for f in os.listdir('.') if f.startswith('best_policy_')]
    if best_models:
        best_model = max(best_models, key=lambda x: float(x.split('_')[-1].split('.pth')[0]))
        print(f"\nTesting best model: {best_model}")
        test_trained_model(best_model)