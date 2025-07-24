#!/usr/bin/env python3
"""
Complete Phase 1 Training Script
Trains enhanced transformer agent and compares with baselines
"""

import torch
import numpy as np
import random
import argparse
import os
import time
from typing import Dict, List
import matplotlib.pyplot as plt

# Import all our components
from envs.enhanced_causal_env import EnhancedCausalEnv
from models.enhanced_transformer_policy import EnhancedTransformerPolicy
from models.baseline_models import BaselineComparison
from agents.enhanced_ppo_agent import EnhancedPPOAgent, CausalExperienceCollector
from language.instruction_processor import InstructionDataset
from evaluation.evaluation_framework import run_full_evaluation

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_directories():
    """Create necessary directories for saving results"""
    directories = ['models', 'results', 'logs', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train_agent(config: Dict) -> EnhancedPPOAgent:
    """
    Train the enhanced transformer agent
    """
    print("🚀 Starting Enhanced Transformer Agent Training...")
    
    # Create environment
    env = EnhancedCausalEnv(
        config_name=config['env_config'],
        partial_observability=config['partial_obs'],
        max_steps=config['max_episode_steps']
    )
    
    # Get environment info
    grid_size = (env.grid_height, env.grid_width)
    num_objects = max(20, len(env.config.get('objects', [])) + 10)  # Ensure adequate vocabulary
    action_dim = env.action_space.n
    
    # Create language dataset
    instruction_dataset = InstructionDataset()
    vocab_size = instruction_dataset.get_vocab_size()
    
    print(f"Environment setup: {grid_size} grid, {num_objects} objects, {action_dim} actions")
    print(f"Language vocab size: {vocab_size}")
    
    # Create enhanced transformer policy
    policy = EnhancedTransformerPolicy(
        grid_size=grid_size,
        num_objects=num_objects,
        action_dim=action_dim,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        vocab_size=vocab_size
    )
    
    # Create enhanced PPO agent
    agent = EnhancedPPOAgent(
        policy=policy,
        lr=config['learning_rate'],
        gamma=config['gamma'],
        clip_epsilon=config['clip_epsilon'],
        causal_loss_coef=config['causal_loss_coef']
    )
    
    # Create experience collector
    collector = CausalExperienceCollector(env)
    
    # Training loop
    episode_rewards = []
    success_rates = []
    best_success_rate = 0.0
    
    print(f"Training for {config['max_episodes']} episodes...")
    
    for episode in range(config['max_episodes']):
        # Collect trajectory
        instruction = instruction_dataset.get_random_instruction()
        instruction_tokens = instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
        
        trajectory = collector.collect_trajectory(
            agent, 
            max_steps=config['max_episode_steps'],
            instruction_tokens=instruction_tokens
        )
        
        # Update agent
        loss_dict = agent.update(trajectory)
        
        # Track performance
        episode_reward = sum(trajectory['rewards'])
        episode_rewards.append(episode_reward)
        
        # Periodic evaluation
        if (episode + 1) % config['eval_frequency'] == 0:
            success_rate = evaluate_agent_quick(agent, env, instruction_dataset)
            success_rates.append(success_rate)
            
            print(f"Episode {episode + 1}:")
            print(f"  Reward: {episode_reward:.3f}")
            print(f"  Success Rate: {success_rate:.3f}")
            print(f"  Policy Loss: {loss_dict['policy_loss']:.4f}")
            print(f"  Causal Loss: {loss_dict['causal_loss']:.4f}")
            print(f"  LR: {loss_dict['learning_rate']:.2e}")
            
            # Save best model
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'agent_state_dict': agent.optimizer.state_dict(),
                    'episode': episode,
                    'success_rate': success_rate,
                    'config': config
                }, f'models/best_transformer_agent_{success_rate:.3f}.pth')
                print(f"  💾 New best model saved! Success rate: {success_rate:.3f}")
        
        # Early stopping
        if len(success_rates) >= 5 and success_rates[-1] > 0.8:
            print(f"🎉 Early stopping! Achieved {success_rates[-1]:.1%} success rate")
            break
    
    # Final training statistics
    print(f"\n📊 Training Complete!")
    print(f"Best Success Rate: {best_success_rate:.1%}")
    print(f"Final Episode Reward: {episode_rewards[-1]:.3f}")
    
    return agent, episode_rewards, success_rates

def train_baselines(config: Dict) -> Dict[str, any]:
    """
    Train all baseline models for comparison
    """
    print("🔧 Training Baseline Models...")
    
    env = EnhancedCausalEnv(config_name=config['env_config'])
    grid_size = (env.grid_height, env.grid_width)
    num_objects = len(env.config['objects']) + 3
    action_dim = env.action_space.n
    
    # Create baseline comparison
    baseline_comparison = BaselineComparison(grid_size, num_objects, action_dim)
    baseline_comparison.create_standard_baselines()
    
    trained_baselines = {}
    
    for name, baseline_model in baseline_comparison.get_all_baselines().items():
        if name == "Random":
            # Random baseline doesn't need training
            trained_baselines[name] = baseline_model
            continue
        
        print(f"Training {name} baseline...")
        
        # Create simple PPO agent for baseline
        baseline_agent = EnhancedPPOAgent(
            policy=baseline_model,
            lr=config['learning_rate'] * 0.5,  # Slightly lower LR for baselines
            causal_loss_coef=0.0  # No causal loss for baselines
        )
        
        # Train for fewer episodes (baselines are simpler)
        baseline_episodes = config['max_episodes'] // 2
        
        for episode in range(baseline_episodes):
            # Simple trajectory collection (no language)
            state, _ = env.reset()
            
            trajectory = {
                'states': [], 'actions': [], 'log_probs': [], 
                'rewards': [], 'values': [], 'masks': []
            }
            
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            
            for step in range(config['max_episode_steps']):
                if hasattr(baseline_model, 'forward'):
                    action, log_prob, value = baseline_agent.select_action(state_tensor)
                else:
                    # Handle random baseline
                    action = env.action_space.sample()
                    log_prob = torch.tensor(0.0)
                    value = torch.tensor(0.0)
                
                next_state, reward, done, truncated, _ = env.step(action)
                
                trajectory['states'].append(state_tensor.squeeze(0))
                trajectory['actions'].append(action)
                trajectory['log_probs'].append(log_prob)
                trajectory['rewards'].append(reward)
                trajectory['values'].append(value)
                trajectory['masks'].append(0.0 if done else 1.0)
                
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                
                if done or truncated:
                    break
            
            # Compute returns and advantages
            if hasattr(baseline_model, 'forward'):
                final_value = baseline_agent.select_action(state_tensor)[2] if not done else torch.zeros(1)
                returns, advantages = baseline_agent.compute_gae(
                    trajectory['rewards'], trajectory['values'], 
                    trajectory['masks'], final_value
                )
                trajectory['returns'] = returns
                trajectory['advantages'] = advantages
                
                # Update baseline
                baseline_agent.update(trajectory)
        
        trained_baselines[name] = baseline_agent if hasattr(baseline_model, 'forward') else baseline_model
        print(f"✅ {name} baseline training complete")
    
    return trained_baselines

def evaluate_agent_quick(agent, env, instruction_dataset, num_episodes: int = 20) -> float:
    """Quick evaluation for tracking training progress"""
    successes = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        # Use simple instruction that matches environment
        simple_instructions = [
            "First activate the switch then go to the goal",
            "Use the switch to open the door",
            "Press the switch and then reach the goal"
        ]
        instruction = np.random.choice(simple_instructions)
        instruction_tokens = instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
        
        for step in range(100):
            # Use STOCHASTIC action selection (same as training)
            action, _, _ = agent.select_action(state_tensor, instruction_tokens, deterministic=False)
            state, reward, done, truncated, _ = env.step(action)
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            
            if done:
                if reward > 0:
                    successes += 1
                break
                
            if truncated:
                break
    
    return successes / num_episodes

def save_training_plots(episode_rewards: List[float], success_rates: List[float], config: Dict):
    """Save training progress plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards During Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Success rates
    eval_episodes = np.arange(0, len(episode_rewards), config['eval_frequency'])[:len(success_rates)]
    ax2.plot(eval_episodes, success_rates, 'o-')
    ax2.set_title('Success Rate During Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.grid(True)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'plots/training_progress_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training and evaluation pipeline"""
    parser = argparse.ArgumentParser(description='Train Phase 1 Causal RL Agent')
    parser.add_argument('--max_episodes', type=int, default=2000, help='Maximum training episodes')
    parser.add_argument('--eval_frequency', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--env_config', type=str, default='default', help='Environment configuration')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only evaluate')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'max_episodes': args.max_episodes,
        'eval_frequency': args.eval_frequency,
        'env_config': args.env_config if args.env_config != 'default' else 'intervention_test',  # Use working environment
        'max_episode_steps': 100,
        'partial_obs': False,
        
        # Model hyperparameters
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        
        # Training hyperparameters
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'clip_epsilon': 0.2,
        'causal_loss_coef': 0.1,
    }
    
    print("🎯 Phase 1: Rock Solid Foundation Training")
    print("=" * 50)
    
    # Setup
    set_seed(args.seed)
    create_directories()
    
    # Create language dataset
    instruction_dataset = InstructionDataset()
    
    if not args.skip_training:
        # Training phase
        print("\n🏗️ TRAINING PHASE")
        
        # Train main agent
        agent, episode_rewards, success_rates = train_agent(config)
        
        # Train baselines
        trained_baselines = train_baselines(config)
        
        # Save training plots
        save_training_plots(episode_rewards, success_rates, config)
        
    else:
        # Load pre-trained model
        if not args.model_path:
            print("❌ Error: --model_path required when --skip_training is used")
            return
        
        print(f"\n📁 Loading pre-trained model from {args.model_path}")
        
        # Recreate agent architecture
        env = EnhancedCausalEnv(config_name=config['env_config'])
        grid_size = (env.grid_height, env.grid_width)
        num_objects = len(env.config['objects']) + 3
        action_dim = env.action_space.n
        vocab_size = instruction_dataset.get_vocab_size()
        
        policy = EnhancedTransformerPolicy(
            grid_size=grid_size, num_objects=num_objects, action_dim=action_dim,
            d_model=config['d_model'], nhead=config['nhead'], 
            num_layers=config['num_layers'], vocab_size=vocab_size
        )
        
        agent = EnhancedPPOAgent(policy=policy)
        
        # Load checkpoint
        checkpoint = torch.load(args.model_path)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Train baselines for comparison
        trained_baselines = train_baselines(config)
    
    # Evaluation phase
    print("\n🧪 COMPREHENSIVE EVALUATION PHASE")
    
    # Run full evaluation
    env_class = EnhancedCausalEnv
    results = run_full_evaluation(agent, trained_baselines, env_class, instruction_dataset)
    
    # Print summary
    print("\n🎊 PHASE 1 COMPLETE!")
    print("=" * 50)
    
    main_result = results.get('main_agent')
    if main_result:
        print(f"🏆 Enhanced Transformer Agent Results:")
        print(f"   Success Rate: {main_result.success_rate:.1%}")
        print(f"   Causal Understanding: {main_result.causal_understanding_score:.1%}")
        print(f"   Generalization: {main_result.generalization_score:.1%}")
        print(f"   Language Following: {main_result.instruction_following_score:.1%}")
        
        # Compare with best baseline
        best_baseline_success = 0.0
        best_baseline_name = "None"
        
        for name, result in results.items():
            if name != 'main_agent' and result.success_rate > best_baseline_success:
                best_baseline_success = result.success_rate
                best_baseline_name = name
        
        improvement = main_result.success_rate - best_baseline_success
        print(f"\n📈 Improvement over best baseline ({best_baseline_name}): {improvement:+.1%}")
        
        if improvement > 0.15:  # 15% improvement threshold
            print("✅ SUCCESS: Significant improvement achieved!")
            print("Ready to proceed to Phase 2: Causal World Models")
        else:
            print("⚠️  Warning: Improvement may not be sufficient for publication")
            print("Consider additional training or hyperparameter tuning")
    
    print(f"\n📁 All results saved in results/ directory")
    print(f"📊 Evaluation plots saved in plots/ directory")
    print(f"💾 Best model saved in models/ directory")

if __name__ == "__main__":
    main()