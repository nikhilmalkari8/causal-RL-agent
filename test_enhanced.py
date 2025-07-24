import torch
import numpy as np
from envs.causal_gridworld import CausalDoorEnv
from models.lstm_policy import EnhancedLSTMPolicy

def test_enhanced_model():
    env = CausalDoorEnv()
    
    # Load the enhanced model
    policy = EnhancedLSTMPolicy(150, 256, 4, dropout=0.1)
    policy.load_state_dict(torch.load('enhanced_policy_0.820.pth'))
    policy.eval()
    
    print("=== Testing Enhanced Model ===")
    
    successes = 0
    for test_episode in range(10):  # Test more episodes
        obs, _ = env.reset()
        hx = None
        total_reward = 0
        
        print(f"\nTest Episode {test_episode + 1}:")
        
        for step in range(50):
            state_tensor = torch.tensor(np.eye(6)[obs].flatten(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                logits, _, hx = policy(state_tensor, hx)
                action = torch.argmax(logits, dim=-1).item()
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Only show key states
            if step < 10 or reward > 0 or done:
                env.render()
            
            if done or truncated:
                break
        
        if total_reward > 0:
            successes += 1
            print(f"✅ SUCCESS! Total Reward: {total_reward}")
        else:
            print(f"❌ Failed. Total Reward: {total_reward}")
    
    print(f"\nOverall Success Rate: {successes}/10 = {successes/10:.1%}")

if __name__ == "__main__":
    test_enhanced_model()
