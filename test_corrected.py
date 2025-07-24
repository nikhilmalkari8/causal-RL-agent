import torch
import numpy as np
from envs.causal_gridworld import CausalDoorEnv
from models.lstm_policy import EnhancedLSTMPolicy
from agents.ppo_lstm_agent import PPOAgent

def preprocess_obs(obs):
    one_hot = np.eye(6)[obs]
    return torch.tensor(one_hot.flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def test_corrected_model():
    env = CausalDoorEnv()
    
    # Create the policy and agent (same as training)
    policy = EnhancedLSTMPolicy(150, 256, 4, dropout=0.1)
    agent = PPOAgent(policy)  # Create agent like in training
    
    # Load the trained model
    policy.load_state_dict(torch.load('best_policy_0.820.pth'))
    policy.eval()
    
    print("=== Testing with Corrected Action Selection ===")
    
    successes = 0
    for test_episode in range(5):
        obs, _ = env.reset()
        hx = None
        total_reward = 0
        
        print(f"\nTest Episode {test_episode + 1}:")
        
        for step in range(50):
            state_tensor = preprocess_obs(obs)  # Use same preprocessing as training
            
            # Use the SAME action selection method as training
            action, log_prob, value, hx = agent.select_action(state_tensor, hx)
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Show first few steps and any success
            if step < 15 or reward > 0 or done:
                env.render()
            
            if done or truncated:
                break
        
        if total_reward > 0:
            successes += 1
            print(f"✅ SUCCESS! Total Reward: {total_reward}")
        else:
            print(f"❌ Failed. Total Reward: {total_reward}")
    
    print(f"\nSuccess Rate: {successes}/5 = {successes/5:.1%}")

if __name__ == "__main__":
    test_corrected_model()
