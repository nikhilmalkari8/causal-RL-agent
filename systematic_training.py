#!/usr/bin/env python3
"""
Systematic Training Pipeline for Phase 1
Trains all models (baselines + transformer) and provides comprehensive comparison
"""

import torch
import numpy as np
import random
import os
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict, deque
import seaborn as sns

# Import all components
from envs.enhanced_causal_env import EnhancedCausalEnv
from models.enhanced_transformer_policy import EnhancedTransformerPolicy
from models.baseline_models import LSTMBaseline, CNNBaseline, MLPBaseline, RandomBaseline
from agents.enhanced_ppo_agent import EnhancedPPOAgent
from language.instruction_processor import InstructionDataset

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        # Environment
        self.env_config = "intervention_test"
        self.max_episode_steps = 100
        
        # Training
        self.max_episodes = 1000
        self.eval_frequency = 50
        self.eval_episodes = 20
        
        # Model architectures
        self.hidden_dim = 256
        self.learning_rate = 1e-4
        
        # PPO hyperparameters
        self.gamma = 0.95
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Transformer specific
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        
        # Random seeds for multiple runs
        self.seeds = [42, 123, 456, 789, 999]

class ModelTrainer:
    """Handles training of different model types"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results = defaultdict(list)
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Create language dataset
        self.instruction_dataset = InstructionDataset()
        
    def create_environment(self):
        """Create training environment"""
        return EnhancedCausalEnv(
            config_name=self.config.env_config,
            max_steps=self.config.max_episode_steps
        )
    
    def evaluate_agent(self, agent, model_name: str, num_episodes: int = 20) -> Dict[str, float]:
        """Evaluate agent performance"""
        env = self.create_environment()
        
        successes = 0
        total_rewards = []
        episode_lengths = []
        switch_activations = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            
            # Get instruction for transformer models
            instruction_tokens = None
            if hasattr(agent.policy, 'encode_language'):
                instruction = self.instruction_dataset.get_random_instruction()
                instruction_tokens = self.instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
            
            episode_reward = 0
            steps = 0
            activated_switch = False
            
            for step in range(self.config.max_episode_steps):
                # Select action based on model type
                if hasattr(agent, 'select_action'):
                    action, _, _ = agent.select_action(state_tensor, instruction_tokens, deterministic=True)
                else:
                    # Handle different baseline types
                    if hasattr(agent, 'get_action_distribution'):
                        dist = agent.get_action_distribution(state_tensor)
                        action = dist.sample().item()
                    else:
                        # Random baseline
                        action = env.action_space.sample()
                
                next_state, reward, done, truncated, info = env.step(action)
                
                # Track switch activation
                if not activated_switch and len(env.activated_objects) > 0:
                    activated_switch = True
                
                episode_reward += reward
                steps += 1
                
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                
                if done or truncated:
                    break
            
            if episode_reward > 0:
                successes += 1
            if activated_switch:
                switch_activations += 1
                
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        return {
            'success_rate': successes / num_episodes,
            'avg_reward': np.mean(total_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'switch_activation_rate': switch_activations / num_episodes,
            'reward_std': np.std(total_rewards)
        }
    
    def train_random_baseline(self, seed: int) -> Dict[str, float]:
        """Train (evaluate) random baseline"""
        print(f"  Training Random baseline (seed {seed})...")
        
        set_seed(seed)
        env = self.create_environment()
        
        # Random baseline doesn't need training, just evaluation
        random_agent = RandomBaseline(env.action_space.n)
        
        # Evaluate
        results = self.evaluate_agent(random_agent, "Random")
        
        print(f"    Success Rate: {results['success_rate']:.3f}")
        return results
    
    def train_mlp_baseline(self, seed: int) -> Dict[str, float]:
        """Train MLP baseline"""
        print(f"  Training MLP baseline (seed {seed})...")
        
        set_seed(seed)
        env = self.create_environment()
        grid_size = (env.grid_height, env.grid_width)
        num_objects = 20
        action_dim = env.action_space.n
        
        # Create model and agent
        policy = MLPBaseline(grid_size, num_objects, action_dim, hidden_dim=self.config.hidden_dim)
        agent = EnhancedPPOAgent(
            policy=policy,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            clip_epsilon=self.config.clip_epsilon,
            causal_loss_coef=0.0  # No causal loss for baselines
        )
        
        # Training loop
        training_rewards = []
        
        for episode in range(self.config.max_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory(agent, env, use_instructions=False)
            
            # Update agent
            agent.update(trajectory)
            
            # Track progress
            episode_reward = sum(trajectory['rewards'])
            training_rewards.append(episode_reward)
            
            # Periodic evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_results = self.evaluate_agent(agent, "MLP")
                print(f"    Episode {episode + 1}: Success Rate: {eval_results['success_rate']:.3f}, "
                      f"Avg Reward: {eval_results['avg_reward']:.3f}")
                
                # Early stopping if doing well
                if eval_results['success_rate'] > 0.8:
                    print(f"    Early stopping at episode {episode + 1}")
                    break
        
        # Final evaluation
        final_results = self.evaluate_agent(agent, "MLP")
        final_results['training_rewards'] = training_rewards
        
        print(f"    Final Success Rate: {final_results['success_rate']:.3f}")
        
        # Save model
        torch.save(policy.state_dict(), f'models/mlp_baseline_seed{seed}.pth')
        
        return final_results
    
    def train_cnn_baseline(self, seed: int) -> Dict[str, float]:
        """Train CNN baseline"""
        print(f"  Training CNN baseline (seed {seed})...")
        
        set_seed(seed)
        env = self.create_environment()
        grid_size = (env.grid_height, env.grid_width)
        num_objects = 20
        action_dim = env.action_space.n
        
        # Create model and agent
        policy = CNNBaseline(grid_size, num_objects, action_dim, hidden_dim=self.config.hidden_dim)
        agent = EnhancedPPOAgent(
            policy=policy,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            clip_epsilon=self.config.clip_epsilon,
            causal_loss_coef=0.0
        )
        
        # Training loop
        training_rewards = []
        
        for episode in range(self.config.max_episodes):
            trajectory = self.collect_trajectory(agent, env, use_instructions=False)
            agent.update(trajectory)
            
            episode_reward = sum(trajectory['rewards'])
            training_rewards.append(episode_reward)
            
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_results = self.evaluate_agent(agent, "CNN")
                print(f"    Episode {episode + 1}: Success Rate: {eval_results['success_rate']:.3f}")
                
                if eval_results['success_rate'] > 0.8:
                    break
        
        final_results = self.evaluate_agent(agent, "CNN")
        final_results['training_rewards'] = training_rewards
        
        print(f"    Final Success Rate: {final_results['success_rate']:.3f}")
        
        torch.save(policy.state_dict(), f'models/cnn_baseline_seed{seed}.pth')
        return final_results
    
    def train_lstm_baseline(self, seed: int) -> Dict[str, float]:
        """Train LSTM baseline"""
        print(f"  Training LSTM baseline (seed {seed})...")
        
        set_seed(seed)
        env = self.create_environment()
        grid_size = (env.grid_height, env.grid_width)
        num_objects = 20
        action_dim = env.action_space.n
        
        # Create model and agent
        policy = LSTMBaseline(grid_size, num_objects, action_dim, hidden_dim=self.config.hidden_dim)
        agent = EnhancedPPOAgent(
            policy=policy,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            clip_epsilon=self.config.clip_epsilon,
            causal_loss_coef=0.0
        )
        
        # Training loop with LSTM hidden state management
        training_rewards = []
        
        for episode in range(self.config.max_episodes):
            trajectory = self.collect_trajectory(agent, env, use_instructions=False, use_lstm=True)
            agent.update(trajectory)
            
            episode_reward = sum(trajectory['rewards'])
            training_rewards.append(episode_reward)
            
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_results = self.evaluate_agent(agent, "LSTM")
                print(f"    Episode {episode + 1}: Success Rate: {eval_results['success_rate']:.3f}")
                
                if eval_results['success_rate'] > 0.8:
                    break
        
        final_results = self.evaluate_agent(agent, "LSTM")
        final_results['training_rewards'] = training_rewards
        
        print(f"    Final Success Rate: {final_results['success_rate']:.3f}")
        
        torch.save(policy.state_dict(), f'models/lstm_baseline_seed{seed}.pth')
        return final_results
    
    def train_transformer_agent(self, seed: int) -> Dict[str, float]:
        """Train enhanced transformer agent"""
        print(f"  Training Enhanced Transformer (seed {seed})...")
        
        set_seed(seed)
        env = self.create_environment()
        grid_size = (env.grid_height, env.grid_width)
        num_objects = 20
        action_dim = env.action_space.n
        vocab_size = self.instruction_dataset.get_vocab_size()
        
        # Create enhanced transformer policy
        policy = EnhancedTransformerPolicy(
            grid_size=grid_size,
            num_objects=num_objects,
            action_dim=action_dim,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            vocab_size=vocab_size
        )
        
        # Create enhanced PPO agent with causal loss
        agent = EnhancedPPOAgent(
            policy=policy,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            clip_epsilon=self.config.clip_epsilon,
            causal_loss_coef=0.1  # Enable causal loss
        )
        
        # Training loop
        training_rewards = []
        causal_losses = []
        
        for episode in range(self.config.max_episodes):
            # Collect trajectory with language instructions
            trajectory = self.collect_trajectory(agent, env, use_instructions=True)
            
            # Update agent
            loss_dict = agent.update(trajectory)
            
            # Track metrics
            episode_reward = sum(trajectory['rewards'])
            training_rewards.append(episode_reward)
            causal_losses.append(loss_dict['causal_loss'])
            
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_results = self.evaluate_agent(agent, "Transformer")
                print(f"    Episode {episode + 1}: Success Rate: {eval_results['success_rate']:.3f}, "
                      f"Causal Loss: {loss_dict['causal_loss']:.4f}")
                
                if eval_results['success_rate'] > 0.8:
                    break
        
        final_results = self.evaluate_agent(agent, "Transformer")
        final_results['training_rewards'] = training_rewards
        final_results['causal_losses'] = causal_losses
        
        print(f"    Final Success Rate: {final_results['success_rate']:.3f}")
        
        torch.save(policy.state_dict(), f'models/transformer_agent_seed{seed}.pth')
        return final_results
    
    def collect_trajectory(self, agent, env, use_instructions: bool = False, use_lstm: bool = False):
        """Collect a single trajectory"""
        state, _ = env.reset()
        
        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'masks': [],
            'next_states': []
        }
        
        # Add instruction tokens if needed
        instruction_tokens = None
        if use_instructions:
            instruction = self.instruction_dataset.get_random_instruction()
            instruction_tokens = self.instruction_dataset.tokenize_instruction(instruction).unsqueeze(0)
            trajectory['instruction_tokens'] = []
        
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        hidden_state = None
        
        for step in range(self.config.max_episode_steps):
            # Select action
            if use_lstm and hasattr(agent.policy, 'forward') and 'hidden' in agent.policy.forward.__code__.co_varnames:
                action, log_prob, value, hidden_state = agent.select_action_with_hidden(state_tensor, hidden_state)
            else:
                action, log_prob, value = agent.select_action(state_tensor, instruction_tokens)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.long).unsqueeze(0)
            
            # Store experience
            trajectory['states'].append(state_tensor.squeeze(0))
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['masks'].append(0.0 if done else 1.0)
            trajectory['next_states'].append(next_state_tensor.squeeze(0))
            
            if use_instructions:
                trajectory['instruction_tokens'].append(instruction_tokens.squeeze(0) if instruction_tokens is not None else None)
            
            # Update state
            state = next_state
            state_tensor = next_state_tensor
            
            if done or truncated:
                break
        
        # Compute returns and advantages
        final_value = torch.zeros(1) if done else agent.select_action(state_tensor, instruction_tokens)[2]
        returns, advantages = agent.compute_gae(
            trajectory['rewards'],
            trajectory['values'],
            trajectory['masks'],
            final_value
        )
        
        trajectory['returns'] = returns
        trajectory['advantages'] = advantages
        
        return trajectory
    
    def train_all_models(self):
        """Train all models across multiple seeds"""
        models = {
            'Random': self.train_random_baseline,
            'MLP': self.train_mlp_baseline,
            'CNN': self.train_cnn_baseline,
            'LSTM': self.train_lstm_baseline,
            'Transformer': self.train_transformer_agent
        }
        
        all_results = defaultdict(list)
        
        print("🚀 SYSTEMATIC TRAINING PIPELINE")
        print("=" * 60)
        
        for model_name, train_func in models.items():
            print(f"\n🔧 Training {model_name} Model")
            print("-" * 40)
            
            model_results = []
            
            for seed in self.config.seeds:
                try:
                    results = train_func(seed)
                    model_results.append(results)
                    print(f"  Seed {seed} complete: {results['success_rate']:.3f} success rate")
                except Exception as e:
                    print(f"  Seed {seed} failed: {e}")
                    continue
            
            all_results[model_name] = model_results
            
            # Print summary for this model
            if model_results:
                success_rates = [r['success_rate'] for r in model_results]
                mean_success = np.mean(success_rates)
                std_success = np.std(success_rates)
                print(f"  📊 {model_name} Summary: {mean_success:.3f} ± {std_success:.3f} success rate")
            else:
                print(f"  ❌ {model_name} training failed for all seeds")
        
        return all_results
    
    def save_results(self, results: Dict):
        """Save results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = []
            for result in model_results:
                # Remove non-serializable elements like training_rewards list if too long
                clean_result = {}
                for key, value in result.items():
                    if key in ['training_rewards', 'causal_losses']:
                        # Just save summary statistics for large lists
                        if isinstance(value, list) and len(value) > 0:
                            clean_result[f'{key}_mean'] = float(np.mean(value))
                            clean_result[f'{key}_std'] = float(np.std(value))
                            clean_result[f'{key}_final'] = float(value[-1])
                        else:
                            clean_result[key] = value
                    else:
                        clean_result[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
                serializable_results[model_name].append(clean_result)
        
        with open(f'results/training_results_{timestamp}.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to results/training_results_{timestamp}.json")
        return f'results/training_results_{timestamp}.json'
    
    def generate_comparison_plots(self, results: Dict):
        """Generate comprehensive comparison plots"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Extract data for plotting
        model_names = list(results.keys())
        success_rates = {}
        rewards = {}
        
        for model in model_names:
            if results[model]:  # Check if we have results for this model
                success_rates[model] = [r['success_rate'] for r in results[model]]
                rewards[model] = [r['avg_reward'] for r in results[model]]
            else:
                success_rates[model] = [0.0] * len(self.config.seeds)
                rewards[model] = [0.0] * len(self.config.seeds)
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phase 1 Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. Success Rate Comparison (Box Plot)
        ax1 = axes[0, 0]
        success_data = [success_rates[model] for model in model_names]
        box_plot = ax1.boxplot(success_data, labels=model_names, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors[:len(model_names)]):
            patch.set_facecolor(color)
        
        ax1.set_title('Success Rate Distribution')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 2. Average Reward Comparison (Bar Plot)
        ax2 = axes[0, 1]
        mean_rewards = [np.mean(rewards[model]) for model in model_names]
        std_rewards = [np.std(rewards[model]) for model in model_names]
        
        bars = ax2.bar(model_names, mean_rewards, yerr=std_rewards, capsize=5, 
                      color=colors[:len(model_names)], alpha=0.7)
        ax2.set_title('Average Reward Comparison')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, mean_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        # 3. Success Rate Summary Statistics
        ax3 = axes[1, 0]
        means = [np.mean(success_rates[model]) for model in model_names]
        stds = [np.std(success_rates[model]) for model in model_names]
        
        x_pos = np.arange(len(model_names))
        bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=colors[:len(model_names)], alpha=0.7)
        
        ax3.set_title('Success Rate: Mean ± Std')
        ax3.set_ylabel('Success Rate')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean_val, std_val) in enumerate(zip(means, stds)):
            ax3.text(i, mean_val + std_val + 0.02, f'{mean_val:.3f}±{std_val:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        # 4. Performance Ranking
        ax4 = axes[1, 1]
        
        # Create ranking based on mean success rate
        model_performance = [(model, np.mean(success_rates[model])) for model in model_names]
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        ranked_models = [x[0] for x in model_performance]
        ranked_scores = [x[1] for x in model_performance]
        
        bars = ax4.barh(ranked_models, ranked_scores, 
                       color=colors[:len(model_names)], alpha=0.7)
        ax4.set_title('Model Ranking by Success Rate')
        ax4.set_xlabel('Success Rate')
        ax4.set_xlim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, ranked_scores):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plot_filename = f'plots/model_comparison_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comparison plots saved to {plot_filename}")
        return plot_filename
    
    def generate_training_curves(self, results: Dict):
        """Generate training curve plots"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Only plot models that have training rewards
        models_with_curves = {}
        for model_name, model_results in results.items():
            if model_results and 'training_rewards' in model_results[0]:
                models_with_curves[model_name] = model_results
        
        if not models_with_curves:
            print("No training curves to plot (no training_rewards data)")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Training Rewards Over Time
        ax1 = axes[0]
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_with_curves)))
        
        for (model_name, model_results), color in zip(models_with_curves.items(), colors):
            # Average training curves across seeds
            all_curves = []
            max_length = 0
            
            for result in model_results:
                if 'training_rewards' in result:
                    curve = result['training_rewards']
                    all_curves.append(curve)
                    max_length = max(max_length, len(curve))
            
            if all_curves:
                # Pad curves to same length and average
                padded_curves = []
                for curve in all_curves:
                    padded = curve + [curve[-1]] * (max_length - len(curve))
                    padded_curves.append(padded)
                
                mean_curve = np.mean(padded_curves, axis=0)
                std_curve = np.std(padded_curves, axis=0)
                episodes = np.arange(len(mean_curve))
                
                ax1.plot(episodes, mean_curve, label=model_name, color=color, linewidth=2)
                ax1.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                               alpha=0.2, color=color)
        
        ax1.set_title('Training Progress: Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Causal Loss (for Transformer)
        ax2 = axes[1]
        if 'Transformer' in models_with_curves:
            transformer_results = models_with_curves['Transformer']
            
            causal_curves = []
            for result in transformer_results:
                if 'causal_losses' in result:
                    causal_curves.append(result['causal_losses'])
            
            if causal_curves:
                max_length = max(len(curve) for curve in causal_curves)
                padded_curves = []
                for curve in causal_curves:
                    padded = curve + [curve[-1]] * (max_length - len(curve))
                    padded_curves.append(padded)
                
                mean_curve = np.mean(padded_curves, axis=0)
                std_curve = np.std(padded_curves, axis=0)
                episodes = np.arange(len(mean_curve))
                
                ax2.plot(episodes, mean_curve, color='red', linewidth=2, label='Causal Loss')
                ax2.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                               alpha=0.2, color='red')
                
                ax2.set_title('Transformer: Causal Loss During Training')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Causal Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No Causal Loss Data', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Causal Loss (No Data)')
        else:
            ax2.text(0.5, 0.5, 'Transformer Not Trained', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Causal Loss (No Transformer)')
        
        plt.tight_layout()
        curve_filename = f'plots/training_curves_{timestamp}.png'
        plt.savefig(curve_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Training curves saved to {curve_filename}")
        return curve_filename
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate a comprehensive text summary report"""
        report = []
        report.append("🎯 PHASE 1 TRAINING RESULTS SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        # Calculate statistics for each model
        model_stats = {}
        for model_name, model_results in results.items():
            if model_results:
                success_rates = [r['success_rate'] for r in model_results]
                avg_rewards = [r['avg_reward'] for r in model_results]
                switch_rates = [r.get('switch_activation_rate', 0) for r in model_results]
                
                model_stats[model_name] = {
                    'success_mean': np.mean(success_rates),
                    'success_std': np.std(success_rates),
                    'reward_mean': np.mean(avg_rewards),
                    'reward_std': np.std(avg_rewards),
                    'switch_mean': np.mean(switch_rates),
                    'num_seeds': len(model_results)
                }
            else:
                model_stats[model_name] = {
                    'success_mean': 0.0, 'success_std': 0.0,
                    'reward_mean': 0.0, 'reward_std': 0.0,
                    'switch_mean': 0.0, 'num_seeds': 0
                }
        
        # Overall summary
        report.append("📊 MODEL PERFORMANCE OVERVIEW")
        report.append("-" * 40)
        
        for model_name, stats in model_stats.items():
            report.append(f"🔧 {model_name}:")
            report.append(f"   Success Rate: {stats['success_mean']:.1%} ± {stats['success_std']:.1%}")
            report.append(f"   Avg Reward: {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}")
            report.append(f"   Switch Activation: {stats['switch_mean']:.1%}")
            report.append(f"   Seeds Completed: {stats['num_seeds']}/{len(self.config.seeds)}")
            report.append("")
        
        # Ranking
        report.append("🏆 MODEL RANKING (by Success Rate)")
        report.append("-" * 40)
        
        ranked_models = sorted(model_stats.items(), key=lambda x: x[1]['success_mean'], reverse=True)
        
        for rank, (model_name, stats) in enumerate(ranked_models, 1):
            report.append(f"{rank}. {model_name}: {stats['success_mean']:.1%}")
        
        report.append("")
        
        # Performance gaps
        if len(ranked_models) > 1:
            best_model, best_stats = ranked_models[0]
            
            report.append("📈 PERFORMANCE GAPS")
            report.append("-" * 40)
            
            for model_name, stats in ranked_models[1:]:
                gap = best_stats['success_mean'] - stats['success_mean']
                report.append(f"{best_model} vs {model_name}: +{gap:.1%}")
            
            report.append("")
        
        # Statistical significance (basic)
        report.append("📋 KEY FINDINGS")
        report.append("-" * 40)
        
        if 'Transformer' in model_stats and model_stats['Transformer']['success_mean'] > 0.5:
            report.append("✅ Enhanced Transformer achieves strong performance (>50% success)")
        else:
            report.append("⚠️ Enhanced Transformer performance below 50% - needs investigation")
        
        # Compare with random baseline
        if 'Random' in model_stats:
            random_performance = model_stats['Random']['success_mean']
            learned_models = [name for name in model_stats.keys() if name != 'Random']
            
            learning_occurred = any(model_stats[name]['success_mean'] > random_performance + 0.1 
                                  for name in learned_models)
            
            if learning_occurred:
                report.append("✅ Evidence of learning: Models outperform random baseline")
            else:
                report.append("❌ Minimal learning: Models barely exceed random performance")
        
        # Environment difficulty assessment
        max_success = max(stats['success_mean'] for stats in model_stats.values())
        if max_success < 0.3:
            report.append("⚠️ Environment may be too difficult (max success < 30%)")
        elif max_success > 0.9:
            report.append("⚠️ Environment may be too easy (max success > 90%)")
        else:
            report.append("✅ Environment difficulty appears appropriate")
        
        report.append("")
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 40)
        
        best_model_name = ranked_models[0][0]
        best_performance = ranked_models[0][1]['success_mean']
        
        if best_performance > 0.7:
            report.append("✅ PHASE 1 SUCCESS: Ready to proceed to Phase 2")
            report.append(f"   Best model ({best_model_name}) achieves {best_performance:.1%} success")
        elif best_performance > 0.4:
            report.append("🔄 PARTIAL SUCCESS: Consider improvements before Phase 2")
            report.append("   - Hyperparameter tuning")
            report.append("   - Additional training episodes")
            report.append("   - Architecture modifications")
        else:
            report.append("❌ PHASE 1 INCOMPLETE: Significant issues need addressing")
            report.append("   - Check environment solvability")
            report.append("   - Review model architectures")
            report.append("   - Investigate training dynamics")
        
        # Save report
        report_text = "\n".join(report)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f'results/phase1_summary_{timestamp}.txt'
        
        with open(report_filename, 'w') as f:
            f.write(report_text)
        
        print(f"📋 Summary report saved to {report_filename}")
        return report_text

def main():
    """Main execution function"""
    print("🚀 PHASE 1 SYSTEMATIC TRAINING PIPELINE")
    print("=" * 60)
    print("This will train all models and provide comprehensive comparison")
    print("")
    
    # Create configuration
    config = TrainingConfig()
    
    print("📋 Training Configuration:")
    print(f"   Environment: {config.env_config}")
    print(f"   Max Episodes: {config.max_episodes}")
    print(f"   Seeds: {config.seeds}")
    print(f"   Evaluation Frequency: {config.eval_frequency}")
    print("")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train all models
    print("🏗️ Starting systematic training...")
    results = trainer.train_all_models()
    
    # Save and analyze results
    print("\n📊 ANALYSIS AND REPORTING")
    print("=" * 60)
    
    # Save raw results
    results_file = trainer.save_results(results)
    
    # Generate plots
    comparison_plot = trainer.generate_comparison_plots(results)
    training_curves = trainer.generate_training_curves(results)
    
    # Generate summary report
    summary_report = trainer.generate_summary_report(results)
    print("\n" + summary_report)
    
    print("\n🎊 PHASE 1 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Results saved in: results/")
    print(f"📊 Plots saved in: plots/")
    print(f"💾 Models saved in: models/")

if __name__ == "__main__":
    main()