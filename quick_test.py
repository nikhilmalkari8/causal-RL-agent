#!/usr/bin/env python3
"""
Quick Test Script - Validate everything works before full training
"""

import torch
import numpy as np
import sys

def test_environment():
    """Test that environment works correctly"""
    print("🧪 Testing Environment...")
    
    # Import our simplified environment from master script
    from master_train_evaluate import SimpleCausalEnv, ObjectType
    
    env = SimpleCausalEnv()
    state, _ = env.reset()
    
    print(f"✅ Environment created: {state.shape}")
    
    # Test manual solution
    total_reward = 0
    env.reset()
    
    # Go to switch and activate
    env.step(1)  # Move down to switch
    env.step(4)  # Activate switch
    
    if env.switch_activated:
        print("✅ Switch activation works")
    else:
        print("❌ Switch activation failed")
        return False
    
    # Move to goal
    for _ in range(20):  # Try to reach goal
        state, reward, done, _, _ = env.step(1)  # Move down
        total_reward += reward
        if done:
            break
    
    for _ in range(20):  # Try to reach goal  
        state, reward, done, _, _ = env.step(3)  # Move right
        total_reward += reward
        if done:
            break
    
    if total_reward > 5:
        print(f"✅ Manual solution works: {total_reward:.2f} reward")
        return True
    else:
        print(f"❌ Manual solution failed: {total_reward:.2f} reward")
        return False

def test_models():
    """Test that all models can be created and run"""
    print("🧪 Testing Models...")
    
    from master_train_evaluate import TransformerPolicy, LSTMPolicy, MLPPolicy, RandomPolicy
    
    grid_size = (10, 10)
    num_objects = 7
    action_dim = 5
    
    models = {
        'Transformer': TransformerPolicy(grid_size, num_objects, action_dim),
        'LSTM': LSTMPolicy(grid_size, num_objects, action_dim),
        'MLP': MLPPolicy(grid_size, num_objects, action_dim),
        'Random': RandomPolicy(action_dim)
    }
    
    # Test forward pass
    state = torch.randint(0, num_objects, (1, 10, 10))
    
    for name, model in models.items():
        try:
            outputs = model.forward(state)
            if 'action_logits' in outputs and 'value' in outputs:
                print(f"✅ {name} model works")
            else:
                print(f"❌ {name} model missing outputs")
                return False
        except Exception as e:
            print(f"❌ {name} model failed: {e}")
            return False
    
    return True

def test_training():
    """Test that training loop works"""
    print("🧪 Testing Training Loop...")
    
    from master_train_evaluate import SimpleCausalEnv, TransformerPolicy, PPOAgent
    
    env = SimpleCausalEnv()
    policy = TransformerPolicy((10, 10), 7, 5, d_model=32, nhead=2, num_layers=1)
    agent = PPOAgent(policy)
    
    # Run one episode
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
    
    trajectory = {
        'states': [], 'actions': [], 'log_probs': [], 
        'rewards': [], 'values': []
    }
    
    for step in range(10):  # Short episode
        action, log_prob, value, _ = agent.select_action(state_tensor)
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
        agent.update(trajectory)
        print("✅ Training loop works")
        return True
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 QUICK VALIDATION TESTS")
    print("=" * 40)
    
    tests = [
        ("Environment", test_environment),
        ("Models", test_models),
        ("Training", test_training)
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
        print("⚠️  Some tests failed. Fix issues before running full training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())