#!/usr/bin/env python3
"""
Diagnostic script to identify and fix issues in your causal RL system
"""

import sys
import os
sys.path.append('.')

def diagnose_system():
    """Run comprehensive diagnostics"""
    
    print("🔍 CAUSAL RL SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    issues_found = []
    
    # Check 1: File structure
    print("\n1️⃣ Checking file structure...")
    required_files = [
        'envs/enhanced_causal_env.py',
        'agents/enhanced_ppo_agent.py',
        'models/enhanced_causal_architecture.py',
        'language/instruction_processor.py',
        'evaluation/evaluation_framework.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            issues_found.append(f"Missing file: {file}")
    
    # Check 2: Import test
    print("\n2️⃣ Testing imports...")
    try:
        from envs.enhanced_causal_env import EnhancedCausalEnv
        print("   ✅ Environment import OK")
    except Exception as e:
        print(f"   ❌ Environment import failed: {e}")
        issues_found.append(f"Environment import error: {e}")
    
    try:
        from agents.enhanced_ppo_agent import EnhancedPPOAgent
        print("   ✅ PPO Agent import OK")
    except Exception as e:
        print(f"   ❌ PPO Agent import failed: {e}")
        issues_found.append(f"Agent import error: {e}")
    
    try:
        from models.enhanced_causal_architecture import EnhancedCausalTransformer
        print("   ✅ Model import OK")
    except Exception as e:
        print(f"   ❌ Model import failed: {e}")
        issues_found.append(f"Model import error: {e}")
    
    # Check 3: Environment functionality
    print("\n3️⃣ Testing environment...")
    try:
        from envs.enhanced_causal_env import EnhancedCausalEnv
        env = EnhancedCausalEnv(config_name='intervention_test')
        state, _ = env.reset()
        
        # Check for duplicate step methods
        import inspect
        methods = [method for method in dir(env) if method == 'step']
        if len(methods) > 1:
            print("   ⚠️  Warning: Multiple 'step' methods detected")
            issues_found.append("Environment has duplicate step methods")
        
        # Test step
        next_state, reward, done, truncated, info = env.step(0)
        print("   ✅ Environment step() works")
        
        # Check reward structure
        print(f"   📊 Initial reward: {reward}")
        
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        issues_found.append(f"Environment functionality error: {e}")
    
    # Check 4: Model architecture
    print("\n4️⃣ Testing model architecture...")
    try:
        from models.enhanced_causal_architecture import EnhancedCausalTransformer
        import torch
        
        # Check forward method signature
        import inspect
        sig = inspect.signature(EnhancedCausalTransformer.forward)
        params = list(sig.parameters.keys())
        
        print(f"   📋 Forward method parameters: {params}")
        
        if 'actions' in params:
            print("   ⚠️  Issue: forward() expects 'actions' parameter")
            print("      but PPO agent doesn't provide it!")
            issues_found.append("Model forward() method incompatible with PPO agent")
        else:
            print("   ✅ Forward method signature OK")
            
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        issues_found.append(f"Model architecture error: {e}")
    
    # Check 5: Training compatibility
    print("\n5️⃣ Testing training compatibility...")
    try:
        from envs.enhanced_causal_env import EnhancedCausalEnv
        from agents.enhanced_ppo_agent import EnhancedPPOAgent
        from models.enhanced_causal_architecture import EnhancedCausalTransformer
        from language.instruction_processor import InstructionDataset
        import torch
        
        env = EnhancedCausalEnv(config_name='intervention_test')
        dataset = InstructionDataset()
        
        # Try to create model
        model = EnhancedCausalTransformer(
            grid_size=(env.grid_height, env.grid_width),
            num_objects=20,
            action_dim=env.action_space.n,
            vocab_size=dataset.get_vocab_size(),
            d_model=128
        )
        
        # Try to create agent
        agent = EnhancedPPOAgent(policy=model)
        
        # Try a forward pass
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        instruction = "test"
        instruction_tokens = dataset.tokenize_instruction(instruction).unsqueeze(0)
        
        # This is where it might fail
        action, log_prob, value = agent.select_action(state_tensor, instruction_tokens)
        
        print("   ✅ Training components compatible")
        
    except Exception as e:
        print(f"   ❌ Training compatibility test failed: {e}")
        issues_found.append(f"Training compatibility error: {e}")
        
        # Try to identify the specific issue
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    
    if not issues_found:
        print("✅ No issues found! System appears ready.")
    else:
        print(f"❌ Found {len(issues_found)} issues:\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 RECOMMENDED FIXES:")
        
        if any("forward() method incompatible" in issue for issue in issues_found):
            print("\n1. Fix the model architecture:")
            print("   - Remove 'actions' parameter from forward()")
            print("   - Or modify PPO agent to provide actions")
            print("   - Use the fixed architecture from 'complete_fix_script.py'")
        
        if any("duplicate step methods" in issue for issue in issues_found):
            print("\n2. Fix the environment:")
            print("   - Remove duplicate step() method definitions")
            print("   - Keep only one implementation")
    
    return len(issues_found) == 0

def quick_fix_attempt():
    """Attempt to apply quick fixes"""
    print("\n🔧 ATTEMPTING QUICK FIXES...")
    print("="*60)
    
    # Fix 1: Create a wrapper for the model
    print("\n1️⃣ Creating model wrapper to fix compatibility...")
    
    wrapper_code = '''
import torch
from typing import Dict, Optional

class ModelWrapper(torch.nn.Module):
    """Wrapper to make enhanced model compatible with PPO agent"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, state: torch.Tensor, instruction_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Check if base model expects actions
        try:
            # Try without actions first
            return self.base_model(state, instruction_tokens)
        except TypeError:
            # If it fails, try with dummy actions
            batch_size = state.shape[0]
            dummy_actions = torch.zeros(batch_size, dtype=torch.long)
            return self.base_model(state, instruction_tokens, dummy_actions)
    
    def get_causal_loss(self, *args, **kwargs):
        if hasattr(self.base_model, 'get_causal_loss'):
            return self.base_model.get_causal_loss(*args, **kwargs)
        else:
            return torch.tensor(0.0)
'''
    
    with open('model_wrapper.py', 'w') as f:
        f.write(wrapper_code)
    
    print("   ✅ Model wrapper created")
    
    # Test the wrapper
    try:
        exec(wrapper_code)
        print("   ✅ Wrapper code is valid")
    except Exception as e:
        print(f"   ❌ Wrapper has errors: {e}")
    
    print("\n💡 To use the wrapper:")
    print("   from model_wrapper import ModelWrapper")
    print("   wrapped_model = ModelWrapper(your_model)")
    print("   agent = EnhancedPPOAgent(policy=wrapped_model)")

def main():
    """Run diagnostics and fixes"""
    
    # Run diagnostics
    system_ok = diagnose_system()
    
    if not system_ok:
        # Try quick fixes
        quick_fix_attempt()
        
        print("\n📝 NEXT STEPS:")
        print("1. Run 'python complete_fix_script.py' to use the fixed version")
        print("2. Or apply the fixes manually based on the diagnostics above")
        print("3. Check that enhanced_causal_env.py doesn't have duplicate step() methods")
    else:
        print("\n✅ System appears ready for training!")
        print("   Run 'python main_train.py' to start training")

if __name__ == "__main__":
    main()