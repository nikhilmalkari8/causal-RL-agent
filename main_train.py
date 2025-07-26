#!/usr/bin/env python3
"""
MAIN TRAINING SCRIPT - Complete Working System
This connects all your existing components with the enhanced causal learning
"""

import torch
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Set up paths
import sys
sys.path.append('.')

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_directories():
    """Create output directories"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

class CausalTrainingSystem:
    """Complete causal training system using your existing components"""
    
    def __init__(self):
        print("🧠 INITIALIZING CAUSAL TRAINING SYSTEM")
        print("=" * 50)
        
        # Import your existing working components
        from envs.enhanced_causal_env import EnhancedCausalEnv
        from language.instruction_processor import InstructionDataset
        from agents.enhanced_ppo_agent import EnhancedPPOAgent
        
        # Import the enhanced causal architecture
        from models.enhanced_causal_architecture import EnhancedCausalTransformer
        
        print("✅ All components imported successfully")
        
        # Create environment
        self.env = EnhancedCausalEnv(config_name='intervention_test', max_steps=100)
        print(f"✅ Environment: {self.env.grid_height}x{self.env.grid_width} grid")
        
        # Create language dataset
        self.instruction_dataset = InstructionDataset()
        print(f"✅ Language system: {self.instruction_dataset.get_vocab_size()} vocab size")
        
        # Create enhanced model with explicit causal graph learning
        self.model = EnhancedCausalTransformer(
            grid_size=(self.env.grid_height, self.env.grid_width),
            num_objects=20,
            action_dim=self.env.action_space.n,
            vocab_size=self.instruction_dataset.get_vocab_size(),
            d_model=128  # Smaller for stability
        )
        print(f"✅ Enhanced Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create enhanced PPO agent with STRONG causal learning
        self.agent = EnhancedPPOAgent(
            policy=self.model,
            lr=1e-4,
            entropy_coef=0.1,      # High exploration
            causal_loss_coef=2.0,  # VERY strong causal focus
            clip_epsilon=0.2,
            gamma=0.99
        )
        print(f"✅ PPO Agent: causal_loss_coef = 2.0 (very strong)")
        
        # Training state
        self.episode_rewards = []
        self.success_rates = []
        self.causal_understanding_scores = []
        self.intervention_success_rates = []
        
    def run_complete_training(self, max_episodes=1000):
        """Run complete training with systematic causal learning"""
        
        print(f"\n🚀 STARTING ENHANCED CAUSAL TRAINING")
        print(f"Target: >80% success rate with true causal understanding")
        print("-" * 60)
        
        for episode in range(max_episodes):
            
            # Training curriculum: introduce interventions gradually
            stage = self.get_training_stage(episode)
            use_intervention = self.should_use_intervention(episode, stage)
            
            if use_intervention:
                print(f"🔬 Episode {episode}: Testing with intervention ({stage})")
            
            # Collect trajectory with enhanced causal learning
            trajectory = self.collect_enhanced_trajectory(use_intervention)
            
            # Update agent with causal learning
            loss_dict = self.agent.update(trajectory)
            
            # Track performance
            episode_reward = sum(trajectory['rewards'])
            self.episode_rewards.append(episode_reward)
            
            # Measure causal understanding
            causal_score = self.measure_causal_understanding(trajectory, use_intervention)
            self.causal_understanding_scores.append(causal_score)
            
            # Periodic evaluation
            if (episode + 1) % 50 == 0:
                eval_results = self.comprehensive_evaluation()
                self.success_rates.append(eval_results['success_rate'])
                self.intervention_success_rates.append(eval_results['intervention_robustness'])
                
                print(f"\n📊 Episode {episode + 1} - Stage: {stage}")
                print(f"   Success Rate: {eval_results['success_rate']:.1%}")
                print(f"   Causal Understanding: {eval_results['causal_understanding']:.1%}")
                print(f"   Intervention Robustness: {eval_results['intervention_robustness']:.1%}")
                print(f"   Episode Reward: {episode_reward:.2f}")
                print(f"   Policy Loss: {loss_dict.get('policy_loss', 0):.4f}")
                print(f"   Causal Loss: {loss_dict.get('causal_loss', 0):.4f}")
                
                # Save best model
                if eval_results['success_rate'] > 0.8 and eval_results['causal_understanding'] > 0.7:
                    self.save_best_model(eval_results, episode)
                    print(f"   💾 BREAKTHROUGH MODEL SAVED!")
                    
                    # Check if we've achieved the target
                    if (eval_results['success_rate'] > 0.85 and 
                        eval_results['causal_understanding'] > 0.8):
                        print(f"\n🎉 TARGET ACHIEVED!")
                        print(f"   Success: {eval_results['success_rate']:.1%}")
                        print(f"   Causal Understanding: {eval_results['causal_understanding']:.1%}")
                        break
        
        # Final comprehensive analysis
        final_results = self.final_analysis()
        return final_results
    
    def get_training_stage(self, episode):
        """Get current training stage for curriculum"""
        if episode < 200:
            return "basic_learning"
        elif episode < 400:
            return "causal_discovery"
        elif episode < 600:
            return "intervention_training"
        else:
            return "counterfactual_reasoning"
    
    def should_use_intervention(self, episode, stage):
        """Decide whether to use intervention this episode"""
        if stage == "basic_learning":
            return False
        elif stage == "causal_discovery":
            return episode % 8 == 0  # 12.5% intervention rate
        elif stage == "intervention_training":
            return episode % 4 == 0  # 25% intervention rate
        else:  # counterfactual_reasoning
            return episode % 3 == 0  # 33% intervention rate
    
    def collect_enhanced_trajectory(self, use_intervention=False):
        """Collect trajectory with enhanced causal learning"""
        
        # Apply intervention if needed
        original_config = None
        if use_intervention:
            original_config = self.apply_intervention()
        
        # Reset environment
        state, _ = self.env.reset()
        
        # Get causal instruction
        causal_instructions = [
            "First activate the switch then go to the goal",
            "Use the switch to open the door then reach the goal",
            "Press the switch to open the path and go to the goal",
            "The switch controls the door - activate it before going to goal",
            "You must use the switch to reach the goal"
        ]
        instruction = random.choice(causal_instructions)
        instruction_tokens = self.instruction_dataset.tokenize_instruction(instruction)
        
        # Collect trajectory
        trajectory = {
            'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [],
            'instruction_tokens': [], 'switch_states': [], 'door_states': [],
            'intervention_used': use_intervention
        }
        
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        state_history = [state.copy()]
        
        for step in range(100):
            # Get action using enhanced model
            action, log_prob, value = self.agent.select_action(
                state_tensor, instruction_tokens.unsqueeze(0)
            )
            
            # Take step in environment
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Enhanced reward shaping for causal learning
            enhanced_reward = self.compute_enhanced_reward(action, state, next_state, reward)
            
            # Store trajectory
            trajectory['states'].append(state_tensor.squeeze(0))
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(enhanced_reward)
            trajectory['values'].append(value)
            trajectory['instruction_tokens'].append(instruction_tokens)
            
            # Track causal states for enhanced learning
            switch_state = 1 if len(self.env.activated_objects) > 0 else 0
            door_state = 1 if any(self.env.grid.flatten() == 5) else 0
            trajectory['switch_states'].append(switch_state)
            trajectory['door_states'].append(door_state)
            
            # Update state
            state = next_state
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            state_history.append(state.copy())
            
            if done or truncated:
                break
        
        # Restore environment if intervention was applied
        if use_intervention and original_config:
            self.restore_environment(original_config)
        
        # Compute returns and advantages
        final_value = torch.zeros(1) if done else self.agent.select_action(state_tensor, instruction_tokens.unsqueeze(0))[2]
        masks = [1.0] * (len(trajectory['rewards']) - 1) + [0.0 if done else 1.0]
        
        returns, advantages = self.agent.compute_gae(
            trajectory['rewards'], trajectory['values'], masks, final_value
        )
        
        trajectory['returns'] = returns
        trajectory['advantages'] = advantages
        
        return trajectory
    
    def compute_enhanced_reward(self, action, state, next_state, base_reward):
        """Compute enhanced rewards for causal learning"""
        enhanced_reward = base_reward
        
        # BIG reward for causal actions
        if action == 4:  # Interact action
            agent_pos = tuple(self.env.agent_pos)
            switch_positions = [rule.trigger_pos for rule in self.env.causal_rules 
                              if rule.trigger_type.name == 'SWITCH']
            if agent_pos in switch_positions:
                enhanced_reward += 5.0  # BIG reward for switch activation
                print(f"      🔧 Switch activated! Bonus +5.0")
        
        # Reward causal chain completion
        if len(self.env.activated_objects) > 0:  # Switch activated
            if any(self.env.grid.flatten() == 5):  # Door opened
                enhanced_reward += 2.0  # Reward causal understanding
        
        # Reward goal achievement
        if base_reward > 10:  # Goal reached
            enhanced_reward += 10.0  # Extra bonus for success
        
        return enhanced_reward
    
    def apply_intervention(self):
        """Apply random intervention to test causal understanding"""
        interventions = ["remove_switch", "move_switch", "block_door"]
        intervention = random.choice(interventions)
        
        original_config = {}
        
        if intervention == "remove_switch":
            # Remove switch temporarily
            for rule in self.env.causal_rules:
                if rule.trigger_type.name == 'SWITCH':
                    original_config['switch_pos'] = rule.trigger_pos
                    self.env.grid[rule.trigger_pos] = 0  # Remove switch
                    break
                    
        elif intervention == "move_switch":
            # Move switch to different position
            for rule in self.env.causal_rules:
                if rule.trigger_type.name == 'SWITCH':
                    original_config['switch_pos'] = rule.trigger_pos
                    # Find new position
                    new_pos = (2, 3)  # Different position
                    self.env.grid[rule.trigger_pos] = 0  # Remove from old
                    self.env.grid[new_pos] = 3  # Place at new
                    rule.trigger_pos = new_pos
                    break
        
        return original_config
    
    def restore_environment(self, original_config):
        """Restore environment after intervention"""
        if 'switch_pos' in original_config:
            # Restore switch
            self.env.grid[original_config['switch_pos']] = 3
            for rule in self.env.causal_rules:
                if rule.trigger_type.name == 'SWITCH':
                    rule.trigger_pos = original_config['switch_pos']
                    break
    
    def measure_causal_understanding(self, trajectory, used_intervention):
        """Measure agent's causal understanding from trajectory"""
        switch_activated = any(trajectory['switch_states'])
        goal_reached = sum(trajectory['rewards']) > 15
        
        if used_intervention:
            # If intervention was used, good understanding = failing appropriately
            return 1.0 if not goal_reached else 0.0
        else:
            # Normal episode: good understanding = activate switch AND reach goal
            if switch_activated and goal_reached:
                return 1.0
            elif switch_activated:
                return 0.5  # Partial understanding
            else:
                return 0.0
    
    def comprehensive_evaluation(self, num_episodes=30):
        """Comprehensive evaluation of the agent"""
        print(f"      🧪 Running comprehensive evaluation...")
        
        # Test 1: Normal performance
        normal_success = self.evaluate_normal_performance(num_episodes // 2)
        
        # Test 2: Intervention robustness
        intervention_robustness = self.evaluate_intervention_robustness(num_episodes // 2)
        
        # Test 3: Causal understanding
        causal_understanding = self.evaluate_causal_understanding(num_episodes // 2)
        
        return {
            'success_rate': normal_success,
            'intervention_robustness': intervention_robustness,
            'causal_understanding': causal_understanding
        }
    
    def evaluate_normal_performance(self, num_episodes):
        """Evaluate normal task performance"""
        successes = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            
            instruction = "First activate the switch then go to the goal"
            instruction_tokens = self.instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
            
            for step in range(100):
                action, _, _ = self.agent.select_action(state_tensor, instruction_tokens, deterministic=True)
                state, reward, done, truncated, _ = self.env.step(action)
                state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                
                if done:
                    if reward > 10:
                        successes += 1
                    break
                if truncated:
                    break
        
        return successes / num_episodes
    
    def evaluate_intervention_robustness(self, num_episodes):
        """Evaluate robustness to interventions"""
        appropriate_responses = 0
        
        for _ in range(num_episodes):
            # Apply intervention
            original_config = self.apply_intervention()
            
            state, _ = self.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            
            instruction = "First activate the switch then go to the goal"
            instruction_tokens = self.instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
            
            episode_reward = 0
            for step in range(100):
                action, _, _ = self.agent.select_action(state_tensor, instruction_tokens, deterministic=True)
                state, reward, done, truncated, _ = self.env.step(action)
                state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                episode_reward += reward
                
                if done or truncated:
                    break
            
            # Good robustness = failing when switch is removed (shows causal understanding)
            if episode_reward < 5:  # Failed as expected
                appropriate_responses += 1
            
            # Restore environment
            if original_config:
                self.restore_environment(original_config)
        
        return appropriate_responses / num_episodes
    
    def evaluate_causal_understanding(self, num_episodes):
        """Evaluate causal understanding specifically"""
        causal_behaviors = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            
            instruction = "Use the switch to reach the goal"
            instruction_tokens = self.instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
            
            switch_activated = False
            goal_reached = False
            
            for step in range(100):
                action, _, _ = self.agent.select_action(state_tensor, instruction_tokens, deterministic=True)
                
                if action == 4 and not switch_activated:  # First switch activation
                    switch_activated = True
                
                state, reward, done, truncated, _ = self.env.step(action)
                state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                
                if done:
                    if reward > 10:
                        goal_reached = True
                    break
                if truncated:
                    break
            
            # Good causal understanding = activate switch before reaching goal
            if switch_activated and goal_reached:
                causal_behaviors += 1
        
        return causal_behaviors / num_episodes
    
    def save_best_model(self, eval_results, episode):
        """Save the best performing model"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'agent_state_dict': self.agent.optimizer.state_dict(),
            'eval_results': eval_results,
            'episode': episode,
            'timestamp': timestamp
        }, f'models/causal_breakthrough_{eval_results["success_rate"]:.3f}_{timestamp}.pth')
    
    def final_analysis(self):
        """Comprehensive final analysis"""
        print(f"\n" + "="*60)
        print(f"🎯 FINAL CAUSAL LEARNING ANALYSIS")
        print(f"="*60)
        
        if len(self.success_rates) > 0:
            final_success = self.success_rates[-1]
            final_causal = np.mean(self.causal_understanding_scores[-50:])
            final_intervention = self.intervention_success_rates[-1] if self.intervention_success_rates else 0
            
            print(f"📊 FINAL METRICS:")
            print(f"   Task Success Rate: {final_success:.1%}")
            print(f"   Causal Understanding: {final_causal:.1%}")
            print(f"   Intervention Robustness: {final_intervention:.1%}")
            
            # Success criteria
            if final_success > 0.8 and final_causal > 0.7:
                print(f"\n🎉 BREAKTHROUGH SUCCESS!")
                print(f"   Agent demonstrates true causal reasoning!")
                success_level = "BREAKTHROUGH"
            elif final_success > 0.6:
                print(f"\n🚀 STRONG IMPROVEMENT!")
                print(f"   Significant progress toward causal understanding!")
                success_level = "STRONG"
            else:
                print(f"\n💪 FOUNDATION BUILT!")
                print(f"   Core causal learning mechanisms working!")
                success_level = "FOUNDATION"
            
            # Create comprehensive plots
            self.create_analysis_plots()
            
            return {
                'success_rate': final_success,
                'causal_understanding': final_causal,
                'intervention_robustness': final_intervention,
                'success_level': success_level
            }
        else:
            print(f"❌ No evaluation data collected")
            return {'success_level': 'FAILED'}
    
    def create_analysis_plots(self):
        """Create comprehensive analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Causal Learning Results', fontsize=16)
        
        # Episode rewards
        axes[0,0].plot(self.episode_rewards, alpha=0.7)
        axes[0,0].set_title('Training Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True, alpha=0.3)
        
        # Success rates
        if self.success_rates:
            eval_episodes = range(50, len(self.episode_rewards) + 1, 50)[:len(self.success_rates)]
            axes[0,1].plot(eval_episodes, self.success_rates, 'g-o', linewidth=2)
            axes[0,1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target')
            axes[0,1].set_title('Success Rate')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Success Rate')
            axes[0,1].set_ylim(0, 1)
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Causal understanding
        axes[1,0].plot(self.causal_understanding_scores, 'r-', alpha=0.7)
        if len(self.causal_understanding_scores) > 50:
            # Moving average
            window = 50
            causal_ma = np.convolve(self.causal_understanding_scores, np.ones(window)/window, mode='valid')
            axes[1,0].plot(range(window-1, len(self.causal_understanding_scores)), causal_ma, 'r-', linewidth=3)
        axes[1,0].set_title('Causal Understanding Score')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Causal Score')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].grid(True, alpha=0.3)
        
        # Intervention robustness
        if self.intervention_success_rates:
            eval_episodes = range(50, len(self.episode_rewards) + 1, 50)[:len(self.intervention_success_rates)]
            axes[1,1].plot(eval_episodes, self.intervention_success_rates, 'b-o', linewidth=2)
            axes[1,1].set_title('Intervention Robustness')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Robustness Score')
            axes[1,1].set_ylim(0, 1)
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/causal_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Analysis plots saved: plots/causal_analysis_{timestamp}.png")

def main():
    """Main execution function"""
    print("🧠 ENHANCED CAUSAL RL TRAINING SYSTEM")
    print("Implementing true causal reasoning with explicit graph learning")
    print("")
    
    # Setup
    set_seed(42)
    create_directories()
    
    # Test environment first
    print("🧪 Testing environment...")
    try:
        from envs.enhanced_causal_env import EnhancedCausalEnv
        test_env = EnhancedCausalEnv(config_name='intervention_test')
        state, _ = test_env.reset()
        print("✅ Environment test passed")
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return
    
    # Initialize and run training system
    try:
        training_system = CausalTrainingSystem()
        final_results = training_system.run_complete_training(max_episodes=1000)
        
        print(f"\n🎊 TRAINING COMPLETE!")
        print(f"Final Success Level: {final_results['success_level']}")
        
        if final_results['success_level'] == "BREAKTHROUGH":
            print(f"🎓 Ready for publication and Phase 2!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()