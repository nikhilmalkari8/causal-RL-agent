#!/usr/bin/env python3
"""
Fixed Quick Test Script - Test the corrected master script
"""

import torch
import numpy as np
import sys

# Add the current directory to Python path so we can import from master script
sys.path.append('.')

def test_training_fixed():
    """Test that training loop works with fixed model signatures"""
    print("🧪 Testing Training Loop (Fixed)...")
    
    from master_train_evaluate import SimpleCausalEnv, TransformerPolicy, LSTMPolicy, PPOAgent
    
    env = SimpleCausalEnv()
    
    # Test both Transformer and LSTM
    models_to_test = {
        'Transformer': TransformerPolicy((10, 10), 7, 5, d_model=32, nhead=2, num_layers=1),
        'LSTM': LSTMPolicy((10, 10), 7, 5, hidden_dim=32)
    }
    
    for model_name, policy in models_to_test.items():
        print(f"  Testing {model_name}...")
        
        agent = PPOAgent(policy)
        
        # Run one episode
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        trajectory = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'values': []
        }
        
        hidden = None
        
        for step in range(10):  # Short episode
            # Use the fixed action selection logic
            if hasattr(policy, 'lstm'):  # LSTM model
                action, log_prob, value, hidden = agent.select_action(state_tensor, hidden)
            else:  # Transformer model
                action, log_prob, value, _ = agent.select_action(state_tensor, None)
            
            next_state, reward, done, _, _ = env.step(action)
            
            trajectory['states'].append(state_tensor.squeeze(0))
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            
            state_tensor = torch.tensor(next_state, dtype=torch.long).unsqueeze(0)
            
            if done:
                break
        
        # Test update
        trajectory['returns'] = agent.compute_returns(trajectory['rewards'])
        
        try:
            loss_dict = agent.update(trajectory)
            print(f"    ✅ {model_name} training works, loss: {loss_dict['loss']:.4f}")
        except Exception as e:
            print(f"    ❌ {model_name} training failed: {e}")
            return False
    
    return True

def test_full_integration():
    """Test a mini version of the full pipeline"""
    print("🧪 Testing Full Integration...")
    
    from master_train_evaluate import (SimpleCausalEnv, TransformerPolicy, LSTMPolicy, 
                                     PPOAgent, train_model, Config)
    
    # Mini config for testing
    mini_config = Config()
    mini_config.max_episodes = 5  # Very short training
    mini_config.eval_frequency = 2
    mini_config.quick_eval_episodes = 3
    
    env = SimpleCausalEnv()
    
    # Test training one model
    policy = TransformerPolicy((10, 10), 7, 5, d_model=32, nhead=2, num_layers=1)
    
    try:
        agent, episode_rewards, success_rates = train_model("TestTransformer", policy, env, mini_config, verbose=False)
        print(f"    ✅ Mini training works: {len(episode_rewards)} episodes, final reward: {episode_rewards[-1]:.2f}")
        return True
    except Exception as e:
        print(f"    ❌ Mini training failed: {e}")
        return False

def main():
    """Run all fixed tests"""
    print("🔍 FIXED VALIDATION TESTS")
    print("=" * 40)
    
    tests = [
        ("Training Loop (Fixed)", test_training_fixed),
        ("Full Integration", test_full_integration)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} test passed\n")
            else:
                print(f"❌ {test_name} test failed\n")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}\n")
            all_passed = False
    
    if all_passed:
        print("🎉 All tests passed! Ready to run full training.")
        print("Run: python master_train_evaluate.py")
        return 0
    else:
        print("⚠️  Some tests failed. Check the fixes above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())