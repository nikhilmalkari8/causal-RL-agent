#!/usr/bin/env python3
"""
Environment Test and Verification Script
Tests that the causal environment works correctly and is solvable
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from envs.enhanced_causal_env import EnhancedCausalEnv

def test_environment_basic():
    """Test basic environment functionality"""
    print("🧪 Testing Basic Environment Functionality...")
    
    # Test environment creation
    env = EnhancedCausalEnv(config_name="intervention_test")
    print(f"✅ Environment created: {env.grid_height}x{env.grid_width} grid")
    
    # Test reset
    state, info = env.reset()
    print(f"✅ Environment reset: state shape {state.shape}")
    print(f"   Agent position: {env.agent_pos}")
    print(f"   Switch position: {[rule.trigger_pos for rule in env.causal_rules if rule.trigger_type.name == 'SWITCH']}")
    print(f"   Door position: {[rule.effect_pos for rule in env.causal_rules if rule.effect_type.name == 'DOOR_CLOSED']}")
    
    # Test rendering
    print("\n📺 Initial Environment State:")
    env.render()
    
    return env

def test_manual_solution():
    """Test that the environment is solvable with manual actions"""
    print("\n🎯 Testing Manual Solution...")
    
    env = EnhancedCausalEnv(config_name="intervention_test")
    state, _ = env.reset()
    
    print("Initial state:")
    env.render()
    
    # Manual solution sequence for intervention_test environment
    # Based on the config: agent at (1,1), switch at (2,1), door at (5,4), goal at (8,8)
    
    solution_steps = [
        ("Move down to switch", 1),  # Move down to reach switch at (2,1)
        ("Activate switch", 4),      # Interact to activate switch
        ("Move right", 3),           # Start moving toward door
        ("Move right", 3),
        ("Move right", 3), 
        ("Move down", 1),
        ("Move down", 1),
        ("Move down", 1),            # Now at door (5,4) - should be open
        ("Move right", 3),           # Move through door
        ("Move down", 1),            # Move toward goal
        ("Move down", 1),
        ("Move down", 1),
        ("Move right", 3),           # Move toward goal (8,8)
        ("Move right", 3),
        ("Move right", 3),
        ("Move right", 3),
    ]
    
    total_reward = 0
    step_count = 0
    
    for description, action in solution_steps:
        print(f"\nStep {step_count + 1}: {description} (action {action})")
        
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        env.render()
        print(f"Reward: {reward}, Total: {total_reward}, Done: {done}")
        
        if done:
            print(f"🎉 Goal reached in {step_count} steps! Total reward: {total_reward}")
            return True
        
        if truncated:
            print(f"❌ Episode truncated at step {step_count}")
            return False
        
        if step_count > 20:  # Safety limit
            print(f"❌ Too many steps, stopping")
            return False
    
    print(f"❌ Manual solution failed. Final reward: {total_reward}")
    return False

def test_random_agent():
    """Test environment with random agent to establish baseline"""
    print("\n🎲 Testing Random Agent Performance...")
    
    env = EnhancedCausalEnv(config_name="intervention_test")
    
    num_episodes = 100
    successes = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(100):  # Max steps per episode
            action = env.action_space.sample()  # Random action
            state, reward, done, truncated, _ = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        if episode_reward > 0:
            successes += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 20 == 0:
            success_rate = successes / (episode + 1)
            avg_reward = np.mean(total_rewards)
            print(f"Episodes {episode + 1}: Success rate: {success_rate:.3f}, Avg reward: {avg_reward:.3f}")
    
    final_success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 Random Agent Results ({num_episodes} episodes):")
    print(f"   Success Rate: {final_success_rate:.3f}")
    print(f"   Average Reward: {avg_reward:.3f}")
    print(f"   Average Episode Length: {avg_length:.1f}")
    
    return final_success_rate, avg_reward, avg_length

def visualize_environment():
    """Create a visual representation of the environment layout"""
    print("\n🎨 Creating Environment Visualization...")
    
    env = EnhancedCausalEnv(config_name="intervention_test")
    state, _ = env.reset()
    
    # Create a more detailed visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create color map for different objects
    colors = {
        0: 'white',      # Empty
        1: 'red',        # Agent
        2: 'gray',       # Wall  
        3: 'blue',       # Switch
        4: 'brown',      # Door closed
        5: 'green',      # Door open
        9: 'gold'        # Goal
    }
    
    # Create colored grid
    colored_grid = np.zeros((env.grid_height, env.grid_width, 3))
    
    for i in range(env.grid_height):
        for j in range(env.grid_width):
            cell_value = env.grid[i, j]
            if cell_value == 0:  # Empty
                colored_grid[i, j] = [1, 1, 1]  # White
            elif cell_value == 1:  # Agent
                colored_grid[i, j] = [1, 0, 0]  # Red
            elif cell_value == 2:  # Wall
                colored_grid[i, j] = [0.5, 0.5, 0.5]  # Gray
            elif cell_value == 3:  # Switch
                colored_grid[i, j] = [0, 0, 1]  # Blue
            elif cell_value == 4:  # Door closed
                colored_grid[i, j] = [0.6, 0.3, 0]  # Brown
            elif cell_value == 5:  # Door open
                colored_grid[i, j] = [0, 0.8, 0]  # Green
            elif cell_value == 9:  # Goal
                colored_grid[i, j] = [1, 0.8, 0]  # Gold
    
    ax.imshow(colored_grid)
    ax.set_title("Causal Environment Layout\n(Red=Agent, Blue=Switch, Brown=Door, Gold=Goal)", fontsize=14)
    
    # Add grid lines
    for i in range(env.grid_height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(env.grid_width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5)
    
    # Add position labels
    ax.text(env.agent_pos[1], env.agent_pos[0], 'A', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add causal rule annotations
    for rule in env.causal_rules:
        switch_pos = rule.trigger_pos
        door_pos = rule.effect_pos
        ax.annotate('', xy=(door_pos[1], door_pos[0]), xytext=(switch_pos[1], switch_pos[0]),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                   alpha=0.7)
    
    ax.set_xticks(range(env.grid_width))
    ax.set_yticks(range(env.grid_height))
    plt.tight_layout()
    plt.savefig('environment_layout.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Environment visualization saved as 'environment_layout.png'")

def main():
    """Run all environment tests"""
    print("🚀 ENVIRONMENT VERIFICATION SUITE")
    print("=" * 50)
    
    # Test 1: Basic functionality
    env = test_environment_basic()
    
    # Test 2: Manual solution
    manual_success = test_manual_solution()
    
    if not manual_success:
        print("❌ CRITICAL: Manual solution failed! Environment may not be solvable.")
        print("   Please check environment configuration.")
        return False
    
    # Test 3: Random baseline
    random_success, random_reward, random_length = test_random_agent()
    
    # Test 4: Visualization
    visualize_environment()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 ENVIRONMENT VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"✅ Environment Creation: PASS")
    print(f"✅ Manual Solution: {'PASS' if manual_success else 'FAIL'}")
    print(f"✅ Random Agent Baseline: {random_success:.1%} success rate")
    print(f"✅ Visualization: PASS")
    
    if manual_success and random_success < 0.3:  # Random should be < 30% for good task difficulty
        print("\n🎯 ENVIRONMENT STATUS: READY FOR TRAINING")
        print(f"   Task appears to be properly challenging (random agent: {random_success:.1%})")
        print(f"   Environment is solvable (manual solution: PASS)")
        return True
    else:
        print("\n⚠️ ENVIRONMENT STATUS: NEEDS ADJUSTMENT")
        if not manual_success:
            print("   - Manual solution failed - check environment logic")
        if random_success >= 0.3:
            print(f"   - Random success rate too high ({random_success:.1%}) - task may be too easy")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Environment verification complete! Ready to proceed with model training.")
    else:
        print("\n❌ Environment verification failed! Please fix issues before training.")