import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

class EnhancedLSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, dropout=0.1):
        super(EnhancedLSTMPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Add layer normalization and dropout for better training
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Separate networks for actor and critic with residual connections
        self.fc_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.fc_critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, hx=None):
        # Normalize input
        x = self.input_norm(x)
        
        if hx is None:
            hx = (torch.zeros(1, x.size(0), self.hidden_dim).to(x.device),
                  torch.zeros(1, x.size(0), self.hidden_dim).to(x.device))
        
        lstm_out, hx = self.lstm(x, hx)
        lstm_out = self.output_norm(lstm_out[:, -1, :])
        
        action_logits = self.fc_actor(lstm_out)
        state_value = self.fc_critic(lstm_out)
        
        return action_logits, state_value, hx

class EnhancedPPOAgent:
    def __init__(self, policy, lr=3e-4, gamma=0.99, clip_epsilon=0.2, 
                 value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def select_action(self, state, hx, deterministic=False):
        logits, value, hx = self.policy(state, hx)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob, value, hx

    def compute_gae(self, rewards, masks, values, next_value, gamma=0.99, lambda_=0.95):
        """Generalized Advantage Estimation for better variance reduction"""
        values = values + [next_value]
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lambda_ * masks[step] * gae
            returns.insert(0, gae + values[step])
        
        return returns

    def ppo_update(self, trajectories, epochs=4, batch_size=64):
        states = torch.cat(trajectories['states'])
        actions = torch.tensor(trajectories['actions']).to(states.device)
        old_log_probs = torch.cat(trajectories['log_probs']).detach()
        returns = torch.tensor(trajectories['returns']).to(states.device)
        values = torch.cat(trajectories['values']).detach()

        advantages = returns - values.squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

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
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.scheduler.step()

class CuriosityDrivenRewardShaper:
    """Adds intrinsic motivation based on state visitation counts"""
    def __init__(self, decay=0.99):
        self.state_counts = {}
        self.decay = decay
        
    def get_intrinsic_reward(self, state):
        state_key = tuple(state.flatten())
        self.state_counts[state_key] = self.state_counts.get(state_key, 0) + 1
        # Inverse frequency bonus for exploration
        return 1.0 / np.sqrt(self.state_counts[state_key])
    
    def decay_counts(self):
        for key in self.state_counts:
            self.state_counts[key] *= self.decay

def enhanced_train():
    """Enhanced training with curiosity-driven exploration and better reward shaping"""
    from envs.causal_gridworld import CausalDoorEnv
    
    env = CausalDoorEnv()
    obs, _ = env.reset()

    input_dim = 5*5*6
    hidden_dim = 256
    action_dim = env.action_space.n

    policy = EnhancedLSTMPolicy(input_dim, hidden_dim, action_dim, dropout=0.1)
    agent = EnhancedPPOAgent(policy, lr=1e-4, clip_epsilon=0.15, entropy_coef=0.08)
    
    # Curiosity module
    curiosity = CuriosityDrivenRewardShaper()
    
    max_episodes = 2000
    max_steps = 50
    gamma = 0.95
    
    # Enhanced replay buffer with priority sampling
    replay_buffer = deque(maxlen=500)
    
    # Metrics tracking
    recent_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_success_rate = 0
    
    # Curriculum learning parameters
    curriculum_threshold = 0.5  # When to increase difficulty
    
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
        step_count = 0
        visited_switch = False
        door_attempts = 0

        for step in range(max_steps):
            state_tensor = torch.tensor(np.eye(6)[obs].flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Use epsilon-greedy with decaying exploration
            epsilon = max(0.05, 0.5 * (0.995 ** episode))
            deterministic = random.random() > epsilon
            
            action, log_prob, value, hx = agent.select_action(state_tensor, hx, deterministic=deterministic)

            next_obs, env_reward, done, truncated, _ = env.step(action)
            
            # Enhanced reward shaping
            shaped_reward = env_reward
            
            # Check if agent visited switch
            if list(obs) == [0, 4]:  # Agent position matches switch position
                if not visited_switch:
                    shaped_reward += 0.3  # First time bonus
                    visited_switch = True
                else:
                    shaped_reward += 0.1  # Smaller repeated bonus
            
            # Penalty for trying door without switch
            if list(obs) == [2, 2] and not env.door_open:  # At door position but not open
                door_attempts += 1
                shaped_reward -= 0.1  # Small penalty
            
            # Success bonus
            if env_reward > 0:
                shaped_reward += 1.0 + max(0, (max_steps - step) * 0.02)  # Time bonus
            
            # Curiosity reward (smaller weight)
            intrinsic_reward = curiosity.get_intrinsic_reward(obs)
            shaped_reward += 0.05 * intrinsic_reward
            
            episode_reward += env_reward
            step_count += 1

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(shaped_reward)
            masks.append(1 - done)
            values.append(value)

            obs = next_obs
            if done or truncated:
                break

        # GAE returns computation
        next_state_tensor = torch.tensor(np.eye(6)[obs].flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, next_value, _ = policy(next_state_tensor, hx)

        returns = agent.compute_gae(rewards, masks, [v.squeeze().item() for v in values], 
                                  next_value.squeeze().item(), gamma, lambda_=0.95)

        trajectories = {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'returns': returns,
            'values': values
        }

        # Store successful episodes with priority
        if episode_reward > 0:
            replay_buffer.append((trajectories, episode_reward, step_count))

        # Update with current episode
        agent.ppo_update(trajectories, epochs=4, batch_size=32)

        # Experience replay with priority sampling
        if len(replay_buffer) > 10 and episode % 3 == 0:
            # Sample based on reward (higher reward = higher probability)
            weights = [exp[1] for exp in replay_buffer]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            replay_traj = np.random.choice(list(replay_buffer), p=probs)[0]
            agent.ppo_update(replay_traj, epochs=2, batch_size=16)

        # Decay curiosity counts periodically
        if episode % 100 == 0:
            curiosity.decay_counts()

        # Tracking
        recent_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        # Performance evaluation
        if len(recent_rewards) >= 50:
            success_rate = sum(1 for r in list(recent_rewards)[-50:] if r > 0) / 50
            avg_reward = np.mean(list(recent_rewards)[-50:])
            avg_length = np.mean(list(episode_lengths)[-50:])
            
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode+1}: Success: {success_rate:.3f}, "
                      f"Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.1f}, "
                      f"Buffer: {len(replay_buffer)}, LR: {agent.scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save({
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'success_rate': success_rate,
                    'episode': episode
                }, f'enhanced_policy_{success_rate:.3f}.pth')
            
            # Early stopping
            if success_rate > 0.9:
                print(f"Training successful! Success rate: {success_rate:.3f}")
                break

    print(f"Training completed. Best success rate: {best_success_rate:.3f}")
    return policy, best_success_rate

if __name__ == "__main__":
    enhanced_train()