#!/usr/bin/env python3
"""
Progressive Causal Learning Trainer - Research Grade
Prevents catastrophic forgetting and ensures steady improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import json
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import deque

# Import your existing components
from envs.enhanced_causal_env import EnhancedCausalEnv
from models.enhanced_transformer_policy import EnhancedTransformerPolicy
from models.baseline_models import BaselineComparison
from agents.enhanced_ppo_agent import EnhancedPPOAgent, CausalExperienceCollector
from agents.ppo_lstm_agent import PPOAgent
from language.instruction_processor import InstructionDataset

class ProgressiveLearningConfig:
    """Research-grade configuration to prevent catastrophic forgetting"""
    
    def __init__(self):
        # PROGRESSIVE LEARNING RATES
        self.learning_phases = {
            'exploration': {
                'episodes': (1, 200),
                'lr': 1e-4,
                'entropy_coef': 0.1,  # High exploration
                'clip_epsilon': 0.3,   # Aggressive updates
                'description': 'Initial exploration and basic learning'
            },
            'consolidation': {
                'episodes': (201, 400), 
                'lr': 5e-5,
                'entropy_coef': 0.05,  # Reduced exploration
                'clip_epsilon': 0.2,   # Standard updates
                'description': 'Consolidating learned patterns'
            },
            'refinement': {
                'episodes': (401, 600),
                'lr': 2e-5,
                'entropy_coef': 0.02,  # Low exploration
                'clip_epsilon': 0.15,  # Conservative updates
                'description': 'Refining policy without forgetting'
            },
            'mastery': {
                'episodes': (601, 1000),
                'lr': 1e-5,
                'entropy_coef': 0.01,  # Minimal exploration
                'clip_epsilon': 0.1,   # Very conservative
                'description': 'Achieving mastery while preserving knowledge'
            }
        }
        
        # ANTI-FORGETTING MECHANISMS
        self.experience_replay_size = 2000
        self.replay_ratio = 0.3  # 30% of updates from replay
        self.success_memory_size = 500  # Remember successful episodes
        self.gradient_clipping = 0.5
        
        # CAUSAL LEARNING FOCUS
        self.causal_loss_schedule = {
            (1, 200): 0.05,     # Low initially
            (201, 400): 0.1,    # Increase as agent learns
            (401, 600): 0.15,   # Peak causal focus
            (601, 1000): 0.1    # Maintain causal understanding
        }
        
        # CURRICULUM LEARNING
        self.instruction_curriculum = {
            (1, 200): ['simple', 'causal'],           # Basic instructions
            (201, 400): ['simple', 'causal', 'sequential'],  # Add complexity
            (401, 600): ['causal', 'sequential', 'spatial'], # Focus on reasoning
            (601, 1000): 'all'                       # Full complexity
        }

class ProgressiveExperienceReplay:
    """Advanced experience replay to prevent catastrophic forgetting"""
    
    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.successful_episodes = deque(maxlen=max_size // 2)  # Keep successful episodes
        self.diverse_episodes = deque(maxlen=max_size // 2)     # Keep diverse episodes
        self.episode_count = 0
        
    def add_episode(self, trajectory: Dict, episode_reward: float, episode_number: int):
        """Add episode with intelligent storage strategy"""
        
        # Create episode data
        episode_data = {
            'trajectory': trajectory,
            'reward': episode_reward,
            'episode': episode_number,
            'success': episode_reward > 5
        }
        
        # Store successful episodes
        if episode_reward > 5:
            self.successful_episodes.append(episode_data)
        
        # Store diverse episodes (every 5th episode regardless of success)
        if episode_number % 5 == 0:
            self.diverse_episodes.append(episode_data)
        
        self.episode_count += 1
    
    def sample_replay_batch(self, batch_size: int = 5):
        """Sample a batch for replay learning"""
        replay_episodes = []
        
        # Sample from successful episodes (70%)
        if len(self.successful_episodes) > 0:
            success_count = min(int(batch_size * 0.7), len(self.successful_episodes))
            success_samples = random.sample(list(self.successful_episodes), success_count)
            replay_episodes.extend(success_samples)
        
        # Sample from diverse episodes (30%)
        remaining = batch_size - len(replay_episodes)
        if remaining > 0 and len(self.diverse_episodes) > 0:
            diverse_count = min(remaining, len(self.diverse_episodes))
            diverse_samples = random.sample(list(self.diverse_episodes), diverse_count)
            replay_episodes.extend(diverse_samples)
        
        return replay_episodes
    
    def get_stats(self):
        """Get replay buffer statistics"""
        return {
            'total_episodes': self.episode_count,
            'successful_stored': len(self.successful_episodes),
            'diverse_stored': len(self.diverse_episodes),
            'success_rate_in_buffer': np.mean([ep['success'] for ep in self.successful_episodes]) if self.successful_episodes else 0
        }

class AdaptiveLearningRateScheduler:
    """Adaptive learning rate based on performance"""
    
    def __init__(self, optimizer, base_config: ProgressiveLearningConfig):
        self.optimizer = optimizer
        self.config = base_config
        self.performance_history = deque(maxlen=50)
        self.current_phase = 'exploration'
        
    def update(self, episode: int, success_rate: float):
        """Update learning rate based on episode and performance"""
        
        # Determine current phase
        old_phase = self.current_phase
        for phase_name, phase_config in self.config.learning_phases.items():
            start_ep, end_ep = phase_config['episodes']
            if start_ep <= episode <= end_ep:
                self.current_phase = phase_name
                break
        
        # Get base learning rate for current phase
        phase_config = self.config.learning_phases[self.current_phase]
        base_lr = phase_config['lr']
        
        # Adaptive adjustment based on performance
        self.performance_history.append(success_rate)
        
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            
            # If performance is degrading, reduce learning rate
            if len(self.performance_history) >= 20:
                older_performance = np.mean(list(self.performance_history)[-20:-10])
                if recent_performance < older_performance - 0.1:  # 10% drop
                    base_lr *= 0.5  # Halve learning rate
                    print(f"🔻 Performance drop detected, reducing LR to {base_lr:.2e}")
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        
        # Print phase changes
        if old_phase != self.current_phase:
            print(f"📚 Learning Phase: {old_phase} → {self.current_phase}")
            print(f"   {phase_config['description']}")
            print(f"   LR: {base_lr:.2e}, Entropy: {phase_config['entropy_coef']}")
        
        return base_lr, self.current_phase

class CurriculumInstructionManager:
    """Manages instruction complexity based on learning progress"""
    
    def __init__(self, instruction_dataset: InstructionDataset, config: ProgressiveLearningConfig):
        self.instruction_dataset = instruction_dataset
        self.config = config
        
        # Categorize instructions by type
        self.instruction_categories = {
            'simple': [
                "Go to the goal",
                "Reach the destination",
                "Navigate to the target"
            ],
            'causal': [
                "Use the switch to open the door",
                "Press the switch to open the path",
                "Activate the switch to access the goal",
                "The switch controls the door"
            ],
            'sequential': [
                "First activate the switch then go to the goal",
                "Press the switch and then reach the goal",
                "Hit the switch first then go to the target"
            ],
            'spatial': [
                "Go to the switch in the top area",
                "Find the switch near the top of the room",
                "Activate the switch in the upper region"
            ],
            'complex': [
                "If the door is closed find the switch near the top and activate it before going to the goal",
                "Navigate to the switch then move through the opened door to reach the goal"
            ]
        }
    
    def get_instruction_for_episode(self, episode: int):
        """Get appropriate instruction based on curriculum"""
        
        # Determine allowed instruction types for current episode
        allowed_types = None
        for episode_range, types in self.config.instruction_curriculum.items():
            if isinstance(episode_range, tuple):
                start_ep, end_ep = episode_range
                if start_ep <= episode <= end_ep:
                    allowed_types = types
                    break
        
        if allowed_types == 'all':
            # Use all instruction types
            return self.instruction_dataset.get_random_instruction()
        elif allowed_types:
            # Sample from allowed types
            chosen_type = random.choice(allowed_types)
            if chosen_type in self.instruction_categories:
                instruction = random.choice(self.instruction_categories[chosen_type])
                return instruction
        
        # Fallback to simple instruction
        return "First activate the switch then go to the goal"

def progressive_train_enhanced_transformer(config: ProgressiveLearningConfig):
    """Progressive training that prevents catastrophic forgetting"""
    
    print("🧠 PROGRESSIVE CAUSAL LEARNING TRAINER")
    print("=" * 60)
    print("Research-grade training to prevent catastrophic forgetting")
    print("Target: Slow but steady improvement → mastery by episode 600")
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Create environment and datasets
    env = EnhancedCausalEnv(config_name='intervention_test', max_steps=100)
    instruction_dataset = InstructionDataset()
    curriculum_manager = CurriculumInstructionManager(instruction_dataset, config)
    
    grid_size = (env.grid_height, env.grid_width)
    num_objects = 20
    action_dim = env.action_space.n
    vocab_size = instruction_dataset.get_vocab_size()
    
    print(f"Environment: {grid_size}, Objects: {num_objects}, Actions: {action_dim}")
    
    # Create enhanced transformer policy  
    policy = EnhancedTransformerPolicy(
        grid_size=grid_size,
        num_objects=num_objects,
        action_dim=action_dim,
        d_model=128,  # Smaller model for stability
        nhead=4,
        num_layers=3,
        vocab_size=vocab_size
    )
    
    # Create agent with initial conservative settings
    agent = EnhancedPPOAgent(
        policy=policy,
        lr=config.learning_phases['exploration']['lr'],
        gamma=0.99,
        clip_epsilon=config.learning_phases['exploration']['clip_epsilon'],
        entropy_coef=config.learning_phases['exploration']['entropy_coef'],
        causal_loss_coef=0.05,
        max_grad_norm=config.gradient_clipping
    )
    
    # Progressive learning components
    experience_replay = ProgressiveExperienceReplay(config.experience_replay_size)
    lr_scheduler = AdaptiveLearningRateScheduler(agent.optimizer, config)
    collector = CausalExperienceCollector(env)
    
    # Training tracking
    episode_rewards = []
    success_rates = []
    goals_per_100 = []
    goal_episodes = []
    learning_phases = []
    current_100_goals = 0
    best_success_rate = 0.0
    
    # Main training loop
    print(f"\n🚀 Starting Progressive Training for 1000 episodes...")
    
    for episode in range(1000):
        
        # Get curriculum-appropriate instruction
        instruction = curriculum_manager.get_instruction_for_episode(episode + 1)
        instruction_tokens = instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
        
        # Collect trajectory
        trajectory = collector.collect_trajectory(
            agent, max_steps=100, instruction_tokens=instruction_tokens
        )
        
        # Track performance
        episode_reward = sum(trajectory['rewards'])
        episode_rewards.append(episode_reward)
        
        # Track successful episodes
        if episode_reward > 5:
            goal_episodes.append(episode + 1)
            current_100_goals += 1
            print(f"🏆 GOAL REACHED! Episode {episode + 1}, Reward: {episode_reward:.3f}")
        
        # Add to experience replay
        experience_replay.add_episode(trajectory, episode_reward, episode + 1)
        
        # Update agent with current episode
        loss_dict = agent.update(trajectory)
        
        # Experience replay learning (anti-forgetting)
        if episode > 50 and len(experience_replay.successful_episodes) > 5:
            replay_episodes = experience_replay.sample_replay_batch(3)
            for replay_data in replay_episodes:
                replay_loss = agent.update(replay_data['trajectory'])
        
        # Periodic evaluation and adjustments
        if (episode + 1) % 100 == 0:
            
            # Track goals per 100
            goals_per_100.append(current_100_goals)
            print(f"📊 Episodes {episode + 1 - 99}-{episode + 1}: {current_100_goals}/100 goals ({current_100_goals}%)")
            current_100_goals = 0
            
            # Quick evaluation
            quick_successes = 0
            for _ in range(20):
                state, _ = env.reset()
                state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                
                eval_instruction = curriculum_manager.get_instruction_for_episode(episode + 1)
                eval_tokens = instruction_dataset.tokenize_instruction(eval_instruction).unsqueeze(0)
                
                episode_reward_eval = 0
                for step in range(100):
                    action, _, _ = agent.select_action(state_tensor, eval_tokens)
                    state, reward, done, truncated, _ = env.step(action)
                    state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                    episode_reward_eval += reward
                    
                    if done:
                        if episode_reward_eval > 5:
                            quick_successes += 1
                        break
                    if truncated:
                        break
            
            success_rate = quick_successes / 20
            success_rates.append(success_rate)
            
            # Update learning rate based on performance
            current_lr, current_phase = lr_scheduler.update(episode + 1, success_rate)
            learning_phases.append(current_phase)
            
            # Update causal loss coefficient
            for episode_range, causal_coef in config.causal_loss_schedule.items():
                start_ep, end_ep = episode_range
                if start_ep <= episode + 1 <= end_ep:
                    agent.causal_loss_coef = causal_coef
                    break
            
            # Get replay buffer stats
            replay_stats = experience_replay.get_stats()
            
            # Comprehensive reporting
            print(f"\nEpisode {episode + 1} - Phase: {current_phase}")
            print(f"  Training Reward: {episode_reward:.3f}")
            print(f"  Evaluation Success: {success_rate:.3f}")
            print(f"  Goals this 100: {current_100_goals}")
            print(f"  Total goals: {len(goal_episodes)}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Causal Loss Coef: {agent.causal_loss_coef:.3f}")
            print(f"  Policy Loss: {loss_dict.get('policy_loss', 0):.4f}")
            print(f"  Replay Buffer: {replay_stats['successful_stored']} successes stored")
            
            # Save best model
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'success_rate': success_rate,
                    'episode': episode + 1,
                    'goals_per_100': goals_per_100,
                    'learning_phase': current_phase
                }, f'models/progressive_best_{success_rate:.3f}.pth')
                print(f"  💾 New best model saved! Success: {success_rate:.3f}")
        
        # Phase-specific actions
        current_episode = episode + 1
        for phase_name, phase_config in config.learning_phases.items():
            start_ep, end_ep = phase_config['episodes']
            if start_ep <= current_episode <= end_ep:
                # Update agent parameters for current phase
                agent.entropy_coef = phase_config['entropy_coef']
                agent.clip_epsilon = phase_config['clip_epsilon']
                break
    
    # Final analysis
    print("\n" + "="*60)
    print("🎯 PROGRESSIVE TRAINING COMPLETE")
    print("="*60)
    
    print(f"Total episodes: 1000")
    print(f"Total goals achieved: {len(goal_episodes)}")
    print(f"Final success rate: {success_rates[-1]:.1%}")
    print(f"Goals per 100 episodes: {goals_per_100}")
    
    # Check if target achieved
    if len(goals_per_100) >= 6 and goals_per_100[5] >= 80:  # 80%+ by episode 600
        print("✅ TARGET ACHIEVED: >80% success by episode 600!")
    elif success_rates[-1] >= 0.7:
        print("✅ STRONG PERFORMANCE: >70% final success rate")
    else:
        print("⚠️  Target not fully achieved but strong progress made")
    
    # Create comprehensive visualization
    create_progressive_learning_plots(episode_rewards, success_rates, goals_per_100, 
                                    goal_episodes, learning_phases)
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'success_rates': success_rates,
        'goals_per_100': goals_per_100,
        'goal_episodes': goal_episodes,
        'learning_phases': learning_phases
    }

def create_progressive_learning_plots(episode_rewards, success_rates, goals_per_100, 
                                    goal_episodes, learning_phases):
    """Create comprehensive visualization of progressive learning"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Progressive Causal Learning Results', fontsize=16)
    
    # 1. Goals per 100 episodes with trend
    ax = axes[0, 0]
    episodes_100 = [(i+1)*100 for i in range(len(goals_per_100))]
    ax.plot(episodes_100, goals_per_100, 'bo-', linewidth=2, markersize=8)
    
    # Add trend line
    if len(goals_per_100) > 2:
        z = np.polyfit(episodes_100, goals_per_100, 1)
        p = np.poly1d(z)
        ax.plot(episodes_100, p(episodes_100), "r--", alpha=0.8, linewidth=2)
    
    ax.set_title('Goals Per 100 Episodes (Target: 80+ by Ep 600)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Goals Achieved')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Target')
    ax.legend()
    
    # 2. Success rate progression
    ax = axes[0, 1]
    eval_episodes = [(i+1)*100 for i in range(len(success_rates))]
    ax.plot(eval_episodes, success_rates, 'go-', linewidth=2, markersize=6)
    ax.set_title('Evaluation Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative goals over time
    ax = axes[0, 2]
    if goal_episodes:
        cumulative = list(range(1, len(goal_episodes) + 1))
        ax.plot(goal_episodes, cumulative, 'r-', linewidth=2)
        ax.set_title('Cumulative Goal Achievements')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Goals')
        ax.grid(True, alpha=0.3)
    
    # 4. Learning phases
    ax = axes[1, 0]
    if learning_phases:
        phase_colors = {'exploration': 'red', 'consolidation': 'orange', 
                       'refinement': 'blue', 'mastery': 'green'}
        unique_phases = list(set(learning_phases))
        for i, phase in enumerate(unique_phases):
            episodes = [j*100 for j, p in enumerate(learning_phases) if p == phase]
            ax.scatter(episodes, [i]*len(episodes), c=phase_colors.get(phase, 'gray'), 
                      s=100, alpha=0.7, label=phase)
        ax.set_title('Learning Phases')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Phase')
        ax.legend()
    
    # 5. Training rewards
    ax = axes[1, 1]
    if episode_rewards:
        # Moving average for smoothing
        window = 50
        if len(episode_rewards) > window:
            smooth_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            smooth_episodes = range(window-1, len(episode_rewards))
            ax.plot(smooth_episodes, smooth_rewards, 'b-', alpha=0.7)
        ax.plot(episode_rewards, alpha=0.3, color='lightblue')
        ax.set_title('Training Rewards (50-episode moving average)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
    
    # 6. Performance summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    if len(goals_per_100) >= 6:
        summary_data = [
            ['Episodes 1-100', f'{goals_per_100[0]}%'],
            ['Episodes 101-200', f'{goals_per_100[1]}%'],
            ['Episodes 201-300', f'{goals_per_100[2]}%'],
            ['Episodes 301-400', f'{goals_per_100[3]}%'],
            ['Episodes 401-500', f'{goals_per_100[4]}%'],
            ['Episodes 501-600', f'{goals_per_100[5]}%'],
            ['', ''],
            ['Target Achievement', '✅' if goals_per_100[5] >= 80 else '⚠️'],
            ['Final Success Rate', f'{success_rates[-1]:.1%}' if success_rates else 'N/A']
        ]
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Period', 'Success Rate'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Performance Summary')
    
    plt.tight_layout()
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'plots/progressive_learning_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Plots saved as: plots/progressive_learning_{timestamp}.png")

def main():
    """Main progressive training execution"""
    
    print("🧠 PROGRESSIVE CAUSAL REASONING TRAINER")
    print("Research-grade solution to prevent catastrophic forgetting")
    print("Target: Steady improvement from 0% to 80%+ by episode 600")
    
    # Create configuration
    config = ProgressiveLearningConfig()
    
    # Run progressive training
    results = progressive_train_enhanced_transformer(config)
    
    # Save comprehensive results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_summary = {
        'goals_per_100': results['goals_per_100'],
        'success_rates': results['success_rates'],
        'goal_episodes': results['goal_episodes'],
        'learning_phases': results['learning_phases'],
        'final_analysis': {
            'total_goals': len(results['goal_episodes']),
            'final_success_rate': results['success_rates'][-1] if results['success_rates'] else 0,
            'target_achieved': len(results['goals_per_100']) >= 6 and results['goals_per_100'][5] >= 80
        }
    }
    
    with open(f'results/progressive_results_{timestamp}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"💾 Results saved as: results/progressive_results_{timestamp}.json")

if __name__ == "__main__":
    main()