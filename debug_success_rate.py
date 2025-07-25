#!/usr/bin/env python3
"""
Debug Success Rate Issue - Find why evaluation shows 0%
"""

import torch
import numpy as np
import sys
sys.path.append('.')
sys.path.append('./envs')
sys.path.append('./agents')
sys.path.append('./language')

from envs.enhanced_causal_env import EnhancedCausalEnv
from agents.enhanced_ppo_agent import EnhancedPPOAgent
from language.instruction_processor import InstructionDataset

def debug_single_episode_step_by_step():
    """Debug a single episode step by step"""
    print("🔍 DEBUGGING SINGLE EPISODE STEP BY STEP")
    print("=" * 50)
    
    # Create environment
    env = EnhancedCausalEnv(config_name='intervention_test')
    
    # Create a simple policy for testing
    from models.enhanced_transformer_policy import EnhancedTransformerPolicy
    
    grid_size = (env.grid_height, env.grid_width)
    num_objects = 20
    action_dim = env.action_space.n
    
    policy = EnhancedTransformerPolicy(
        grid_size=grid_size,
        num_objects=num_objects,
        action_dim=action_dim,
        d_model=128,  # Smaller for testing
        nhead=4,
        num_layers=2
    )
    
    agent = EnhancedPPOAgent(policy=policy)
    
    # Create instruction dataset
    instruction_dataset = InstructionDataset()
    instruction = instruction_dataset.get_random_instruction()
    instruction_tokens = instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
    
    print(f"Using instruction: '{instruction}'")
    
    # Reset environment
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
    
    print("\nInitial state:")
    env.render()
    
    episode_reward = 0
    step_rewards = []
    
    for step in range(100):
        print(f"\n--- Step {step} ---")
        
        # Select action
        action, log_prob, value = agent.select_action(state_tensor, instruction_tokens, deterministic=True)
        print(f"Action selected: {action}")
        
        # Take step
        next_state, reward, done, truncated, info = env.step(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.long).unsqueeze(0)
        
        episode_reward += reward
        step_rewards.append(reward)
        
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Truncated: {truncated}")
        print(f"Total reward so far: {episode_reward}")
        print(f"Info: {info}")
        
        if step % 5 == 0 or reward != -0.01:  # Show environment when interesting
            env.render()
        
        state_tensor = next_state_tensor
        
        if done:
            print(f"\n🏁 EPISODE FINISHED!")
            print(f"Final reward: {reward}")
            print(f"Total episode reward: {episode_reward}")
            print(f"Success?: {reward > 5}")
            break
            
        if truncated:
            print(f"\n⏰ EPISODE TRUNCATED!")
            print(f"Total episode reward: {episode_reward}")
            break
    
    return episode_reward, reward, done

def debug_evaluation_function():
    """Debug the specific evaluation function logic"""
    print("\n🔍 DEBUGGING EVALUATION FUNCTION")
    print("=" * 50)
    
    # Recreate the evaluation logic from comprehensive_train.py
    env = EnhancedCausalEnv(config_name='intervention_test')
    instruction_dataset = InstructionDataset()
    
    # Create agent (dummy)
    from models.enhanced_transformer_policy import EnhancedTransformerPolicy
    
    grid_size = (env.grid_height, env.grid_width)
    policy = EnhancedTransformerPolicy(grid_size, 20, env.action_space.n, d_model=64, nhead=2, num_layers=1)
    agent = EnhancedPPOAgent(policy=policy)
    
    # Test the evaluation logic
    print("Testing evaluation on 5 episodes...")
    
    successes = 0
    for episode in range(5):
        print(f"\nEvaluation Episode {episode + 1}:")
        
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        instruction = instruction_dataset.get_random_instruction()
        instruction_tokens = instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
        
        for step in range(100):
            # This is the EXACT logic from comprehensive_train.py
            action, _, _ = agent.select_action(state_tensor, instruction_tokens, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            
            if done:
                print(f"  Episode finished at step {step}")
                print(f"  Final reward: {reward}")
                print(f"  Checking: reward > 5? {reward > 5}")
                
                if reward > 5:  # This is the SUCCESS CONDITION
                    successes += 1
                    print(f"  ✅ SUCCESS counted!")
                else:
                    print(f"  ❌ Not counted as success (reward = {reward})")
                break
                
            if truncated:
                print(f"  Episode truncated at step {step}")
                print(f"  Final reward: {reward}")
                print(f"  ❌ Not counted as success (truncated)")
                break
    
    success_rate = successes / 5
    print(f"\nEvaluation Result: {successes}/5 = {success_rate:.3f}")
    
    return success_rate

def debug_reward_threshold():
    """Debug what rewards we're actually getting"""
    print("\n🔍 DEBUGGING REWARD THRESHOLDS")
    print("=" * 50)
    
    env = EnhancedCausalEnv(config_name='intervention_test')
    
    print("Testing manual solution to see rewards...")
    
    # Manual solution
    state, _ = env.reset()
    total_reward = 0
    step_rewards = []
    
    # Move to switch (agent starts at 1,1, switch at 2,1)
    print("1. Moving to switch...")
    state, reward, done, truncated, _ = env.step(1)  # Down
    total_reward += reward
    step_rewards.append(reward)
    print(f"   Step reward: {reward}, Total: {total_reward}")
    
    # Activate switch
    print("2. Activating switch...")
    state, reward, done, truncated, _ = env.step(4)  # Interact
    total_reward += reward
    step_rewards.append(reward)
    print(f"   Step reward: {reward}, Total: {total_reward}")
    
    # Move towards goal (8,8)
    print("3. Moving to goal...")
    moves_to_goal = [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3]  # Down then right
    
    for i, move in enumerate(moves_to_goal):
        state, reward, done, truncated, _ = env.step(move)
        total_reward += reward
        step_rewards.append(reward)
        print(f"   Move {i+1}: action={move}, reward={reward}, total={total_reward}, done={done}")
        
        if done:
            print(f"   🏆 GOAL REACHED!")
            print(f"   Final reward for this step: {reward}")
            print(f"   Total episode reward: {total_reward}")
            break
    
    print(f"\nFinal Analysis:")
    print(f"All step rewards: {step_rewards}")
    print(f"Total episode reward: {total_reward}")
    print(f"Final step reward: {step_rewards[-1] if step_rewards else 0}")
    print(f"Would this count as success (reward > 5)?: {step_rewards[-1] > 5 if step_rewards else False}")
    
    return total_reward, step_rewards

def main():
    """Run all debugging tests"""
    print("🚨 SUCCESS RATE DEBUGGING SESSION")
    print("=" * 60)
    
    # Test 1: Single episode
    episode_reward, final_reward, done = debug_single_episode_step_by_step()
    
    # Test 2: Manual solution
    manual_total, manual_rewards = debug_reward_threshold()
    
    # Test 3: Evaluation function
    eval_success_rate = debug_evaluation_function()
    
    # Summary
    print("\n📋 DEBUGGING SUMMARY")
    print("=" * 40)
    print(f"Single episode final reward: {final_reward}")
    print(f"Manual solution final reward: {manual_rewards[-1] if manual_rewards else 'N/A'}")
    print(f"Evaluation success rate: {eval_success_rate}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    if final_reward > 5:
        print("✅ Agent IS getting successful rewards")
        if eval_success_rate == 0:
            print("❌ But evaluation function is broken - need to fix evaluation logic")
        else:
            print("✅ Evaluation function is working")
    else:
        print("❌ Agent is NOT getting successful rewards")
        print("Issue is in training, not evaluation")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if manual_rewards and manual_rewards[-1] > 5:
        print("1. Manual solution works - environment is correct")
        print("2. Check agent action selection logic")
        print("3. Check if deterministic vs stochastic makes difference")
    else:
        print("1. Environment might have issues")
        print("2. Reward structure might be wrong")
    
    if eval_success_rate == 0 and final_reward > 5:
        print("3. Fix evaluation function - reward threshold or logic issue")

if __name__ == "__main__":
    main()