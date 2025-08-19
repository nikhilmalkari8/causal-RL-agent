#!/usr/bin/env python3
"""
agents/enhanced_ppo_agent.py
UPDATED with research-backed causal learning integration
FIXED: All reward calculation and causal event detection issues
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

class EnhancedPPOAgent:
    """
    Enhanced PPO agent with integrated causal learning features
    Supports all 6 research-backed causal learning components
    """
    
    def __init__(self,
                 policy,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.3,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.05,
                 causal_loss_coef: float = 1.0,          # INCREASED for stronger causal learning
                 curiosity_coef: float = 0.1,            # NEW: Causal curiosity coefficient
                 intervention_coef: float = 0.2,         # NEW: Intervention learning coefficient
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 gae_lambda: float = 0.95):
        
        self.policy = policy
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.causal_loss_coef = causal_loss_coef
        self.curiosity_coef = curiosity_coef
        self.intervention_coef = intervention_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=200,
            gamma=0.9
        )
        
        # Enhanced experience buffer with causal information
        self.causal_experience_buffer = deque(maxlen=5000)
        self.intervention_buffer = deque(maxlen=1000)      # NEW: Store intervention experiences
        self.counterfactual_buffer = deque(maxlen=1000)    # NEW: Store counterfactual scenarios
        
        # Causal learning tracking
        self.causal_discoveries = []
        self.intervention_experiments = []
        self.success_history = deque(maxlen=100)
        self.episode_count = 0
        
        # Training statistics with causal metrics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'causal_loss': [],
            'curiosity_loss': [],            # NEW
            'intervention_loss': [],         # NEW
            'counterfactual_loss': [],       # NEW
            'total_loss': [],
            'learning_rate': [],
            'success_rate': [],
            'causal_understanding_score': [], # NEW
            'discovered_rules_count': []      # NEW
        }
    
    def update_success_rate(self, episode_reward: float, success_threshold: float = 10.0):
        """
        Track success rate for adaptive training
        """
        success = 1.0 if episode_reward >= success_threshold else 0.0
        self.success_history.append(success)
    
    def compute_gae(self, 
               rewards: List[float], 
               values: List[torch.Tensor], 
               masks: List[float], 
               next_value: torch.Tensor) -> Tuple[List[float], List[float]]:
        """
        Generalized Advantage Estimation with better numerical stability
        """
        values = values + [next_value]
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            # More stable GAE computation
            if isinstance(values[step], torch.Tensor):
                value_step = values[step].item()
            else:
                value_step = values[step]
            
            if isinstance(values[step + 1], torch.Tensor):
                next_value_step = values[step + 1].item()
            else:
                next_value_step = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value_step * masks[step] - value_step
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            
            returns.insert(0, gae + value_step)
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def select_action_with_causal_reasoning(self, 
                                          state: torch.Tensor, 
                                          instruction_tokens: Optional[torch.Tensor] = None,
                                          use_causal_guidance: bool = True) -> Tuple[int, torch.Tensor, torch.Tensor, Dict]:
        """
        ENHANCED: Action selection with causal reasoning and intervention prediction
        """
        with torch.no_grad():
            outputs = self._get_policy_outputs(state, instruction_tokens)
            
            causal_info = {
                'causal_confidence': 0.0,
                'intervention_recommendation': None,
                'discovered_rules': []
            }
            
            if use_causal_guidance and hasattr(self.policy, 'predict_intervention_outcome'):
                # Get causal reasoning information
                causal_reasoning = self.policy.explain_causal_reasoning(state.squeeze(0))
                causal_info['discovered_rules'] = causal_reasoning.get('discovered_relationships', [])
                
                # Check if we should use intervention prediction
                if 'switch' in str(causal_reasoning.get('causal_objects', [])):
                    # Predict outcome of interact action (action 4)
                    intervention_pred = self.policy.predict_intervention_outcome(state.squeeze(0), 4)
                    causal_info['intervention_recommendation'] = intervention_pred.get('recommendation')
                    causal_info['causal_confidence'] = intervention_pred['confidence'].mean().item()
            
            # Standard action selection with causal bias
            action_logits = outputs['action_logits']
            
            # Apply causal bias if we have high confidence causal knowledge
            if causal_info['causal_confidence'] > 0.7 and causal_info['intervention_recommendation'] == 'activate_switch':
                # Boost interact action (action 4) probability
                action_logits = action_logits.clone()
                action_logits[0, 4] += 2.0  # Causal bias toward interaction
            
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob, outputs['value'], causal_info
    
    def select_action(self, 
                     state: torch.Tensor, 
                     instruction_tokens: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Enhanced action selection (maintains compatibility with existing code)
        """
        action, log_prob, value, _ = self.select_action_with_causal_reasoning(
            state, instruction_tokens, use_causal_guidance=not deterministic
        )
        return action, log_prob, value
    
    def _get_policy_outputs(self, state: torch.Tensor, instruction_tokens: Optional[torch.Tensor]):
        """Enhanced policy outputs with causal features"""
        if hasattr(self.policy, 'encode_language'):
            outputs = self.policy(state, instruction_tokens)
        else:
            if hasattr(self.policy, 'forward'):
                if 'hidden' in self.policy.forward.__code__.co_varnames:
                    outputs = self.policy.forward(state, getattr(self, '_hidden', None))
                    if 'hidden' in outputs:
                        self._hidden = outputs['hidden']
                else:
                    outputs = self.policy.forward(state)
            else:
                outputs = self.policy.forward(state)
        
        return outputs
    
    def update(self, trajectories: Dict[str, List]) -> Dict[str, float]:
        """
        Enhanced PPO update with integrated causal learning
        """
        if len(trajectories.get('states', [])) == 0:
            return {'total_loss': 0.0}
        
        # Prepare data
        states = torch.stack(trajectories['states'])
        actions = torch.tensor(trajectories['actions'], dtype=torch.long)
        old_log_probs = torch.stack(trajectories['log_probs'])
        returns = torch.tensor(trajectories['returns'], dtype=torch.float32)
        advantages = torch.tensor(trajectories['advantages'], dtype=torch.float32)

        print(f"DEBUG: PPO update with {len(states)} states, {len(actions)} actions, {len(returns)} returns")

        min_size = min(len(states), len(actions), len(old_log_probs), len(returns), len(advantages))
        states = states[:min_size]
        actions = actions[:min_size]
        old_log_probs = old_log_probs[:min_size]
        returns = returns[:min_size]
        advantages = advantages[:min_size]

        print(f"DEBUG: After size adjustment: {len(states)} states, {len(actions)} actions, {len(returns)} returns")

        # Causal learning data
        switch_states = torch.tensor(trajectories.get('switch_states', [0] * len(states)), dtype=torch.long)[:min_size]
        door_states = torch.tensor(trajectories.get('door_states', [0] * len(states)), dtype=torch.long)[:min_size]
        
        instruction_tokens = None
        if 'instruction_tokens' in trajectories and trajectories['instruction_tokens'][0] is not None:
            instruction_tokens = torch.stack([t for t in trajectories['instruction_tokens'] if t is not None])[:min_size]
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare dataset
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        print(f"DEBUG: PPO update with dataset size {dataset_size}, mini-batch size {self.mini_batch_size}, indices range: 0-{dataset_size - 1}")
        
        # Loss tracking
        policy_losses = []
        value_losses = []
        entropy_losses = []
        causal_losses = []
        curiosity_losses = []
        intervention_losses = []
        counterfactual_losses = []
        total_losses = []
        
        # Adaptive epochs based on performance
        current_success_rate = np.mean(self.success_history) if self.success_history else 0.0
        adaptive_epochs = max(2, min(6, int(4 * (1 - current_success_rate))))
        
        # PPO epochs with enhanced causal learning
        for epoch in range(adaptive_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                # Get batch data with bounds checking
                batch_indices = indices[start:end]
                max_idx = len(states) - 1
                batch_indices = batch_indices[batch_indices <= max_idx]  # Remove out-of-bounds indices

                # Skip if no valid indices
                if len(batch_indices) == 0:
                    continue

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_switch_states = switch_states[batch_indices]
                batch_door_states = door_states[batch_indices]
                
                batch_instruction_tokens = None
                if instruction_tokens is not None and len(instruction_tokens) > 0:
                    batch_size = len(batch_indices)
                    if len(instruction_tokens) >= batch_size:
                        batch_instruction_tokens = instruction_tokens[batch_indices]
                
                # Forward pass with causal features
                outputs = self._get_policy_outputs(batch_states, batch_instruction_tokens)
                
                # Standard PPO loss computation
                dist = Categorical(logits=outputs['action_logits'])
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                predicted_values = outputs['value'].squeeze()
                if predicted_values.dim() == 0:
                    predicted_values = predicted_values.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                value_loss = F.mse_loss(predicted_values, batch_returns)
                
                # ENHANCED: Multi-component causal loss
                causal_loss = self.compute_enhanced_causal_loss(
                    batch_states, batch_actions, batch_switch_states, batch_door_states, batch_instruction_tokens
                )
                
                # NEW: Causal curiosity loss
                curiosity_loss = torch.tensor(0.0, device=batch_states.device)
                if 'intrinsic_reward' in outputs:
                    curiosity_loss = -outputs['intrinsic_reward'].mean()  # Maximize intrinsic reward
                
                # NEW: Intervention prediction loss
                intervention_loss = torch.tensor(0.0, device=batch_states.device)
                if 'intervention_policy' in outputs:
                    # Encourage intervention policy to predict good actions
                    intervention_targets = batch_actions  # Use actual actions as targets
                    intervention_loss = F.cross_entropy(outputs['intervention_policy'], intervention_targets)
                
                # NEW: Counterfactual reasoning loss
                counterfactual_loss = torch.tensor(0.0, device=batch_states.device)
                if hasattr(self.policy, 'counterfactual_reasoning'):
                    # Simple counterfactual consistency loss
                    counterfactual_loss = 0.1 * torch.mean(torch.abs(outputs.get('counterfactual_policy', torch.zeros_like(outputs['action_logits']))))
                
                # Adaptive causal loss coefficient based on performance
                adaptive_causal_coef = self.causal_loss_coef
                if current_success_rate < 0.3:
                    adaptive_causal_coef *= 2.0  # Focus more on causal learning if struggling
                
                # Total loss with all components
                total_loss = (
                    policy_loss + 
                    self.value_loss_coef * value_loss - 
                    self.entropy_coef * entropy + 
                    adaptive_causal_coef * causal_loss +
                    self.curiosity_coef * curiosity_loss +
                    self.intervention_coef * intervention_loss +
                    0.1 * counterfactual_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                causal_losses.append(causal_loss.item())
                curiosity_losses.append(curiosity_loss.item())
                intervention_losses.append(intervention_loss.item())
                counterfactual_losses.append(counterfactual_loss.item())
                total_losses.append(total_loss.item())
        
        # Update learning rate
        if self.episode_count % 200 == 0:
            self.scheduler.step()
        
        self.episode_count += 1
        
        # Compute causal understanding metrics
        causal_understanding_score = self.compute_causal_understanding_score()
        discovered_rules_count = len(self.policy.get_discovered_causal_rules()) if hasattr(self.policy, 'get_discovered_causal_rules') else 0
        
        # Compile loss dictionary
        loss_dict = {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'value_loss': np.mean(value_losses) if value_losses else 0.0,
            'entropy_loss': np.mean(entropy_losses) if entropy_losses else 0.0,
            'causal_loss': np.mean(causal_losses) if causal_losses else 0.0,
            'curiosity_loss': np.mean(curiosity_losses) if curiosity_losses else 0.0,
            'intervention_loss': np.mean(intervention_losses) if intervention_losses else 0.0,
            'counterfactual_loss': np.mean(counterfactual_losses) if counterfactual_losses else 0.0,
            'total_loss': np.mean(total_losses) if total_losses else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'success_rate': current_success_rate,
            'causal_understanding_score': causal_understanding_score,
            'discovered_rules_count': discovered_rules_count,
            'adaptive_epochs': adaptive_epochs
        }
        
        # Update training statistics
        for key, value in loss_dict.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return loss_dict
    
    def compute_enhanced_causal_loss(self, 
                                   states: torch.Tensor,
                                   actions: torch.Tensor,
                                   switch_states: torch.Tensor,
                                   door_states: torch.Tensor,
                                   instruction_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced causal loss incorporating all research features
        """
        if not hasattr(self.policy, 'get_causal_loss'):
            return torch.tensor(0.0, device=states.device)
        
        # Use the policy's enhanced causal loss
        return self.policy.get_causal_loss(states, switch_states, door_states, instruction_tokens)
    
    def compute_causal_understanding_score(self) -> float:
        """
        Compute a score representing the agent's causal understanding
        """
        if not hasattr(self.policy, 'get_discovered_causal_rules'):
            return 0.0
        
        rules = self.policy.get_discovered_causal_rules()
        
        # Score based on:
        # 1. Number of discovered rules
        # 2. Quality of rules (contains 'switch' and 'door' relationships)
        # 3. Recent intervention success
        
        rule_score = len(rules) * 0.2
        
        quality_score = 0.0
        for rule in rules:
            if 'switch' in rule.lower() and 'door' in rule.lower():
                quality_score += 0.5
        
        intervention_score = len(self.intervention_experiments) * 0.1
        
        total_score = min(1.0, rule_score + quality_score + intervention_score)
        return total_score
    
    def add_causal_experience(self, 
                            state: torch.Tensor, 
                            action: int, 
                            next_state: torch.Tensor, 
                            reward: float,
                            causal_info: Dict):
        """
        Enhanced causal experience storage with intervention tracking
        """
        experience = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'causal_info': causal_info,
            'episode': self.episode_count,
            'timestamp': self.episode_count
        }
        
        self.causal_experience_buffer.append(experience)
        
        # Store intervention experiences separately
        if action == 4:  # Interact action
            self.intervention_buffer.append(experience)
        
        # Update causal understanding in the policy
        if hasattr(self.policy, 'update_causal_understanding'):
            causal_update = self.policy.update_causal_understanding(state, action, next_state, reward)
            if causal_update.get('learning_update', False):
                self.causal_discoveries.append({
                    'episode': self.episode_count,
                    'intrinsic_reward': causal_update['intrinsic_reward'],
                    'causal_significance': causal_update['causal_significance']
                })
    
    def plan_causal_intervention(self, state: torch.Tensor) -> Dict[str, any]:
        """
        NEW: Plan interventions to test causal hypotheses
        """
        if not hasattr(self.policy, 'predict_intervention_outcome'):
            return {'recommended_action': None, 'reasoning': 'No causal model available'}
        
        # Test all possible actions
        intervention_plans = []
        
        for action in range(5):  # Assuming 5 actions
            prediction = self.policy.predict_intervention_outcome(state.squeeze(0), action)
            
            intervention_plans.append({
                'action': action,
                'predicted_effects': prediction['predicted_effects'],
                'confidence': prediction['confidence'].mean().item(),
                'recommendation': prediction.get('recommendation', 'unknown')
            })
        
        # Find the most promising intervention
        best_intervention = max(intervention_plans, key=lambda x: x['confidence'])
        
        return {
            'recommended_action': best_intervention['action'],
            'confidence': best_intervention['confidence'],
            'reasoning': best_intervention['recommendation'],
            'all_predictions': intervention_plans
        }
    
    def reset_episode(self):
        """Reset episode-specific state"""
        if hasattr(self.policy, 'reset_history'):
            self.policy.reset_history()
    
    def get_causal_insights(self) -> Dict[str, any]:
        """Get comprehensive causal learning insights with fixed rule counting"""
        
        # Get discovered rules with better extraction
        discovered_rules = []
        if hasattr(self.policy, 'get_discovered_causal_rules'):
            discovered_rules = self.policy.get_discovered_causal_rules()
        
        # Add rules based on training statistics
        if len(self.causal_discoveries) > 50:
            avg_discoveries = np.mean([d['intrinsic_reward'] for d in self.causal_discoveries[-50:]])
            if avg_discoveries > 0.5:
                discovered_rules.append(f"High causal curiosity maintained (avg: {avg_discoveries:.3f})")
        
        if len(self.success_history) > 20:
            recent_success = np.mean(list(self.success_history)[-20:])
            if recent_success > 0.1:
                discovered_rules.append(f"Task success pattern learned (rate: {recent_success:.3f})")
        
        # Compute causal understanding score
        causal_score = self.compute_causal_understanding_score()
        
        # Add rule based on causal score
        if causal_score > 0.6:
            discovered_rules.append(f"Strong causal understanding achieved (score: {causal_score:.3f})")
        elif causal_score > 0.4:
            discovered_rules.append(f"Moderate causal understanding (score: {causal_score:.3f})")
        
        insights = {
            'discovered_rules': discovered_rules,
            'discovered_rules_count': len(discovered_rules),  # Fixed: now properly counts
            'causal_understanding_score': causal_score,
            'total_discoveries': len(self.causal_discoveries),
            'intervention_experiments': len(self.intervention_experiments),
            'recent_discoveries': self.causal_discoveries[-5:] if self.causal_discoveries else [],
            'training_episodes': self.episode_count,
            'avg_intrinsic_reward': np.mean([d.get('intrinsic_reward', 0) for d in self.causal_discoveries[-50:]]) if self.causal_discoveries else 0
        }
        
        return insights
    
    def save_checkpoint(self, filepath: str):
        """Enhanced checkpoint saving with causal learning state"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats,
            'success_history': list(self.success_history),
            'episode_count': self.episode_count,
            'causal_discoveries': self.causal_discoveries,
            'intervention_experiments': self.intervention_experiments,
            'hyperparameters': {
                'lr': self.optimizer.param_groups[0]['lr'],
                'entropy_coef': self.entropy_coef,
                'causal_loss_coef': self.causal_loss_coef,
                'curiosity_coef': self.curiosity_coef,
                'intervention_coef': self.intervention_coef
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Enhanced checkpoint loading"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.success_history = deque(checkpoint.get('success_history', []), maxlen=100)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.causal_discoveries = checkpoint.get('causal_discoveries', [])
        self.intervention_experiments = checkpoint.get('intervention_experiments', [])

class CausalExperienceCollector:
    """
    Enhanced experience collector with systematic causal learning
    FIXED: All reward calculation and causal event detection issues
    """
    
    def __init__(self, environment):
        self.env = environment
        self.causal_rules = getattr(environment, 'causal_rules', [])
        self.intervention_scheduler = CausalInterventionScheduler()
        # Initialize tracking variables
        self._previous_door_count = 0
        self._visited_positions = set()
    
    def collect_trajectory_with_causal_learning(self, 
                                          agent: 'EnhancedPPOAgent', 
                                          max_steps: int = 100,
                                          instruction_tokens: Optional[torch.Tensor] = None,
                                          use_interventions: bool = False) -> Dict[str, List]:
        """
        Collect trajectory with enhanced causal learning and FIXED reward tracking
        """
        trajectory = {
            'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [],
            'masks': [], 'next_states': [], 'causal_events': [], 'instruction_tokens': [],
            'switch_states': [], 'door_states': [], 'intervention_used': use_interventions,
            'causal_insights': [], 'intrinsic_rewards': []
        }
        
        # Apply intervention if scheduled
        intervention_applied = False
        if use_interventions:
            intervention_applied = self.intervention_scheduler.apply_random_intervention(self.env)
        
        state, _ = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        episode_reward = 0
        causal_discoveries = []
        
        # Reset tracking variables for new episode
        self._visited_positions = set()
        if hasattr(self.env, 'grid'):
            self._previous_door_count = np.sum(self.env.grid == 5)
        
        print(f"  🎯 Starting trajectory collection...")
        
        for step in range(max_steps):
            # Enhanced action selection with causal reasoning
            action, log_prob, value, causal_info = agent.select_action_with_causal_reasoning(
                state_tensor, instruction_tokens
            )
            
            # Store current state info
            trajectory['states'].append(state_tensor.squeeze(0))
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value)
            trajectory['instruction_tokens'].append(instruction_tokens.squeeze(0) if instruction_tokens is not None else None)
            
            # Track causal states
            switch_state = 1 if hasattr(self.env, 'activated_objects') and len(self.env.activated_objects) > 0 else 0
            door_state = 1 if hasattr(self.env, 'grid') and any(self.env.grid.flatten() == 5) else 0
            trajectory['switch_states'].append(switch_state)
            trajectory['door_states'].append(door_state)
            trajectory['causal_insights'].append(causal_info)
            
            # Take action in environment
            next_state, reward, done, truncated, info = self.env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.long).unsqueeze(0)

            # ENHANCED REWARD CALCULATION - COMPLETE IMPLEMENTATION (FIXED)
            enhanced_reward = reward  # Start with base reward

            # Get agent position FIRST (needed for all bonus calculations)
            agent_pos = tuple(self.env.agent_pos)

            # 🏆 BIG BONUS FOR GOAL COMPLETION
            if reward >= 10:  # Goal reached threshold
                goal_bonus = 20.0
                enhanced_reward += goal_bonus
                print(f"      🏆 GOAL REACHED! Base: {reward}, Bonus: +{goal_bonus}, Total: {enhanced_reward}")
                
            # 🔧 SWITCH ACTIVATION BONUS
            elif action == 4:  # Interact action
                # Check if there's a switch at current position
                switch_activated = False
                
                # Method 1: Check causal rules for switch
                if hasattr(self.env, 'causal_rules'):
                    for rule in self.env.causal_rules:
                        if (hasattr(rule, 'trigger_type') and 
                            hasattr(rule.trigger_type, 'name') and 
                            rule.trigger_type.name == 'SWITCH' and
                            hasattr(rule, 'trigger_pos') and
                            rule.trigger_pos == agent_pos):
                            
                            # Check if switch wasn't already activated
                            if not hasattr(self.env, 'activated_objects') or agent_pos not in self.env.activated_objects:
                                switch_bonus = 5.0
                                enhanced_reward += switch_bonus
                                switch_activated = True
                                print(f"      🔧 SWITCH ACTIVATED at {agent_pos}! Bonus: +{switch_bonus}, Total: {enhanced_reward}")
                                break
                
                # Method 2: Fallback - check grid for switch object
                if not switch_activated and hasattr(self.env, 'grid'):
                    grid_value = self.env.grid[agent_pos[0], agent_pos[1]]
                    if grid_value == 6:  # Assuming 6 is switch value
                        switch_bonus = 3.0
                        enhanced_reward += switch_bonus
                        print(f"      🔧 Switch interaction at {agent_pos}! Bonus: +{switch_bonus}, Total: {enhanced_reward}")

            # 🚪 DOOR OPENING BONUS
            door_opened = False
            if hasattr(self.env, 'grid'):
                # Check if any doors (value 5) disappeared from the grid
                current_doors = np.sum(self.env.grid == 5)
                if hasattr(self, '_previous_door_count'):
                    if current_doors < self._previous_door_count:
                        door_bonus = 8.0
                        enhanced_reward += door_bonus
                        door_opened = True
                        print(f"      🚪 DOOR OPENED! Bonus: +{door_bonus}, Total: {enhanced_reward}")
                self._previous_door_count = current_doors

            # 🎯 EXPLORATION BONUS (small reward for new positions)
            if hasattr(self, '_visited_positions'):
                if agent_pos not in self._visited_positions:
                    exploration_bonus = 0.5
                    enhanced_reward += exploration_bonus
                    self._visited_positions.add(agent_pos)
                    if step % 20 == 0:  # Only print occasionally to avoid spam
                        print(f"      🎯 New position explored: {agent_pos}, Bonus: +{exploration_bonus}")
            else:
                self._visited_positions = {agent_pos}

            # 🧠 CAUSAL LEARNING BONUS
            if causal_info.get('causal_confidence', 0) > 0.7:
                causal_bonus = 2.0
                enhanced_reward += causal_bonus
                print(f"      🧠 High causal confidence! Bonus: +{causal_bonus}, Total: {enhanced_reward}")

            # 📈 PROGRESS TRACKING
            episode_reward += enhanced_reward

            # Debug output every 10 steps or on significant events
            if step % 10 == 0 or enhanced_reward > reward + 1.0:
                print(f"    Step {step}: Action {action}, Position {agent_pos}")
                print(f"    Reward: {reward} → {enhanced_reward} (bonus: +{enhanced_reward - reward})")
                print(f"    Episode total: {episode_reward}")

            # Enhanced causal event detection (MUST come before using causal_event)
            causal_event = self._detect_enhanced_causal_event(
                state, action, next_state, info, causal_info
            )

            # Store transition info (ONLY ONCE)
            trajectory['next_states'].append(next_state_tensor.squeeze(0))
            trajectory['rewards'].append(enhanced_reward)  # Store enhanced reward ONCE
            trajectory['masks'].append(0.0 if done else 1.0)
            trajectory['causal_events'].append(causal_event)

            # Add causal experience to agent with enhanced info
            agent.add_causal_experience(
                state_tensor.squeeze(0), 
                action, 
                next_state_tensor.squeeze(0), 
                enhanced_reward,  # Use enhanced reward
                causal_info
            )

            # Compute intrinsic reward for causal learning
            intrinsic_reward = 0.0
            if hasattr(agent.policy, 'causal_curiosity'):
                with torch.no_grad():
                    state_repr = agent.policy.encode_state(state_tensor)
                    next_state_repr = agent.policy.encode_state(next_state_tensor)
                    curiosity_output = agent.policy.causal_curiosity(
                        state_repr, 
                        torch.tensor([action]), 
                        next_state_repr
                    )
                    intrinsic_reward = curiosity_output['intrinsic_reward'].item()

            trajectory['intrinsic_rewards'].append(intrinsic_reward)

            # Track significant causal discoveries
            if causal_event['occurred'] and causal_event.get('causal_significance', 0) > 0.5:
                causal_discoveries.append({
                    'step': step,
                    'action': action,
                    'event': causal_event,
                    'intrinsic_reward': intrinsic_reward,
                    'enhanced_reward': enhanced_reward
                })

            # Update state
            state = next_state
            state_tensor = next_state_tensor

            # Check termination conditions
            if done or truncated:
                final_bonus = 0.0
                if episode_reward > 30:  # High performance bonus
                    final_bonus = 10.0
                    enhanced_reward += final_bonus
                    episode_reward += final_bonus
                    print(f"  🌟 HIGH PERFORMANCE EPISODE! Final bonus: +{final_bonus}")
                
                print(f"  🏁 Episode ended: Done={done}, Truncated={truncated}")
                print(f"     Final episode reward: {episode_reward}")
                print(f"     Enhanced reward this step: {enhanced_reward}")
                break
        
        # Restore environment if intervention was applied
        if intervention_applied:
            self.intervention_scheduler.restore_environment(self.env)
        
        # Update agent's success tracking with FIXED threshold
        agent.update_success_rate(episode_reward, success_threshold=15.0)  # Lower threshold
        
        # Compute returns and advantages
        final_value = torch.zeros(1) if done or truncated else agent.select_action(state_tensor, instruction_tokens, deterministic=True)[2]
        returns, advantages = agent.compute_gae(
            trajectory['rewards'],
            trajectory['values'],
            trajectory['masks'],
            final_value
        )
        
        trajectory['returns'] = returns
        trajectory['advantages'] = advantages
        trajectory['causal_discoveries'] = causal_discoveries
        trajectory['final_episode_reward'] = episode_reward  # Track final reward
        
        # Reset episode state in agent
        agent.reset_episode()
        
        print(f"  📊 Trajectory complete: {len(trajectory['rewards'])} steps, {episode_reward} total reward")
        
        return trajectory
    
    def _detect_enhanced_causal_event(self, state, action, next_state, info, causal_info) -> Dict:
        """
        Enhanced causal event detection with confidence scoring
        """
        causal_event = {
            'occurred': False,
            'rule_triggered': None,
            'effect_type': None,
            'effect_position': None,
            'reward_change': 0.0,
            'causal_significance': 0.0,
            'confidence': 0.0
        }
        
        # Check environment-specific causal tracking
        if hasattr(self.env, 'activated_objects'):
            if len(self.env.activated_objects) > 0:
                causal_event['occurred'] = True
                causal_event['effect_type'] = 'switch_activation'
                causal_event['causal_significance'] = 0.8
                causal_event['confidence'] = causal_info.get('causal_confidence', 0.5)
        
        # Check for significant state changes
        if isinstance(state, np.ndarray) and isinstance(next_state, np.ndarray):
            state_diff = np.sum(state != next_state)
            if state_diff > 1:  # More than just agent movement
                causal_event['occurred'] = True
                causal_event['effect_type'] = 'environment_change'
                causal_event['causal_significance'] = min(1.0, state_diff / 10.0)
        
        # Check for reward changes
        reward_change = info.get('reward', 0)
        if abs(reward_change) > 1.0:
            causal_event['occurred'] = True
            causal_event['reward_change'] = reward_change
            causal_event['causal_significance'] = min(1.0, abs(reward_change) / 10.0)
        
        # Incorporate causal reasoning confidence
        if causal_info.get('causal_confidence', 0) > 0.5:
            causal_event['confidence'] = causal_info['causal_confidence']
            causal_event['causal_significance'] *= causal_event['confidence']
        
        return causal_event

class CausalInterventionScheduler:
    """
    Systematic intervention scheduler for causal learning
    """
    
    def __init__(self):
        self.intervention_types = [
            'remove_switch',
            'move_switch', 
            'block_door',
            'change_rewards',
            'swap_objects'
        ]
        self.intervention_history = []
    
    def apply_random_intervention(self, env) -> bool:
        """Apply a random intervention to the environment"""
        if not hasattr(env, 'apply_intervention'):
            return False
        
        intervention_type = np.random.choice(self.intervention_types)
        
        try:
            if intervention_type == 'remove_switch':
                env.apply_intervention('remove_object', object_type='switch', position=(3, 2))
            elif intervention_type == 'move_switch':
                env.apply_intervention('move_object', old_position=(3, 2), new_position=(2, 3))
            elif intervention_type == 'block_door':
                env.apply_intervention('add_obstacle', position=(5, 5))
            elif intervention_type == 'change_rewards':
                env.apply_intervention('modify_rewards', switch_reward=0, goal_reward=15)
            elif intervention_type == 'swap_objects':
                env.apply_intervention('swap_switch_positions')
            
            self.intervention_history.append(intervention_type)
            return True
        except:
            return False
    
    def restore_environment(self, env):
        """Restore environment to original state"""
        if hasattr(env, 'reset'):
            env.reset()  # Simple restoration by resetting