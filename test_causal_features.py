#!/usr/bin/env python3
"""test_causal_features.py"""

import sys
sys.path.append('.')

def test_causal_imports():
    """Test that all causal components import correctly"""
    print("Testing causal feature imports...")
    
    try:
        from models.causal_graph_module import (
            CausalGraphLayer, InterventionalPredictor, CounterfactualReasoning,
            TemporalCausalChain, ObjectCentricCausal, CausalCuriosityReward
        )
        print("✅ Causal graph module imports successful")
    except Exception as e:
        print(f"❌ Causal graph module import failed: {e}")
        return False
    
    try:
        from models.enhanced_transformer_policy import EnhancedTransformerPolicy
        print("✅ Enhanced transformer imports successful")
    except Exception as e:
        print(f"❌ Enhanced transformer import failed: {e}")
        return False
    
    try:
        from agents.enhanced_ppo_agent import EnhancedPPOAgent
        print("✅ Enhanced PPO agent imports successful")
    except Exception as e:
        print(f"❌ Enhanced PPO agent import failed: {e}")
        return False
    
    return True

def test_causal_functionality():
    """Test basic causal functionality"""
    import torch
    from models.causal_graph_module import CausalGraphLayer
    
    print("Testing causal functionality...")
    
    try:
        # Test causal graph layer
        causal_layer = CausalGraphLayer(num_objects=10, hidden_dim=64)
        test_input = torch.randn(1, 64)
        output = causal_layer(test_input)
        
        assert 'causal_graph' in output
        assert 'temporal_delays' in output
        print("✅ Causal graph layer working")
        
        return True
    except Exception as e:
        print(f"❌ Causal functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced Causal RL Implementation")
    print("=" * 50)
    
    import_success = test_causal_imports()
    if import_success:
        functionality_success = test_causal_functionality()
        
        if functionality_success:
            print("\n✅ All tests passed! Ready to run enhanced training.")
            print("Run: python scripts/train_enhanced_causal_rl.py")
        else:
            print("\n❌ Functionality tests failed.")
    else:
        print("\n❌ Import tests failed.")