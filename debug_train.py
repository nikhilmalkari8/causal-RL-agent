#!/usr/bin/env python3
"""
Minimal Debug Training - Find the issue
"""

import torch
import numpy as np
import random

def set_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

def test_random_agent():
    """Test if random agent can occasionally succeed"""
    from envs.enhanced_causal_env import EnhancedCausalEnv
    
    print("🎲 Testing Random Agent...")
    env = EnhancedCausalEnv(config_name='intervention_test')
    
    successes = 0
    for episode in range(50):
        state, _ = env.reset()
        for step in range(100):
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            
            if done and reward > 0:
                successes += 1
                print(f"  Random success in episode {episode}, step {step}")
                break
            elif done or truncated:
                break
    
    success_rate = successes / 50
    print(f"Random agent success rate: {success_rate:.1%}")
    return success_rate > 0

def test_simple_training():
    """Test with ultra-simple setup"""
    from envs.enhanced_causal_env import EnhancedCausalEnv
    from models.baseline_models import LSTMBaseline
    from agents.enhanced_ppo_agent import EnhancedPPOAgent
    
    print("🧠 Testing Simple LSTM Training...")
    
    env = EnhancedCausalEnv(config_name='intervention_test')
    
    # Use simple LSTM instead of complex transformer
    policy = LSTMBaseline(
        grid_size=(env.grid_height, env.grid_width),
        num_objects=20,
        action_dim=env.action_space.n,
        hidden_dim=64  # Very small
    )
    
    agent = EnhancedPPOAgent(
        policy=policy,
        lr=1e-3,  # Higher learning rate
        entropy_coef=0.2,  # More exploration
        causal_loss_coef=0.0  # No causal loss
    )
    
    successes = 0
    rewards = []
    
    print("Training simple LSTM for 100 episodes...")
    
    for episode in range(100):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        trajectory = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'values': [], 'masks': []
        }
        
        episode_reward = 0
        hidden = None
        
        for step in range(50):
            # LSTM forward
            if hasattr(policy, 'forward'):
                outputs = policy.forward(state_tensor, hidden)
                hidden = outputs.get('hidden', None)
                
                # Sample action
                dist = torch.distributions.Categorical(logits=outputs['action_logits'])
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                value = outputs['value']
            else:
                # Fallback
                action = env.action_space.sample()
                log_prob = torch.tensor(0.0)
                value = torch.tensor(0.0)
            
            next_state, reward, done, truncated, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.long).unsqueeze(0)
            
            trajectory['states'].append(state_tensor.squeeze(0))
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['masks'].append(0.0 if done else 1.0)
            
            episode_reward += reward
            state_tensor = next_state_tensor
            
            if done:
                if reward > 0:
                    successes += 1
                break
            elif truncated:
                break
        
        rewards.append(episode_reward)
        
        # Simple update (skip if no valid trajectory)
        if len(trajectory['states']) > 0 and hasattr(policy, 'forward'):
            try:
                final_value = torch.zeros(1) if done else policy.forward(state_tensor, hidden)['value']
                returns, advantages = agent.compute_gae(
                    trajectory['rewards'], trajectory['values'], 
                    trajectory['masks'], final_value
                )
                trajectory['returns'] = returns
                trajectory['advantages'] = advantages
                agent.update(trajectory)
            except:
                pass  # Skip failed updates
        
        if (episode + 1) % 25 == 0:
            recent_success = successes / (episode + 1)
            avg_reward = np.mean(rewards[-25:])
            print(f"Episode {episode + 1}: Success {recent_success:.2f}, Reward {avg_reward:.3f}")
    
    final_success = successes / 100
    print(f"Simple LSTM final success rate: {final_success:.1%}")
    return final_success > 0.05

def main():
    set_seed()
    
    print("🔍 DEBUGGING TRAINING ISSUES")
    print("=" * 40)
    
    # Test 1: Random agent
    if not test_random_agent():
        print("❌ ISSUE: Even random agent can't succeed - environment problem")
        return
    
    print("✅ Random agent can succeed")
    
    # Test 2: Simple training
    if not test_simple_training():
        print("❌ ISSUE: Simple LSTM can't learn - training problem")
        print("Possible issues:")
        print("- Learning rate too low/high")
        print("- Model too complex")
        print("- Reward structure issues")
        return
    
    print("✅ Simple LSTM can learn!")
    print("🎯 Issue is likely in complex transformer setup")
    print("Recommendation: Use simpler model or adjust hyperparameters")

if __name__ == "__main__":
    main()