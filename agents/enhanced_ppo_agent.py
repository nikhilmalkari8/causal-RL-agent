import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

class EnhancedPPOAgent:
    """
    Enhanced PPO agent with:
    - Causal auxiliary loss
    - Intervention-aware training
    - Language instruction integration
    - Advanced experience replay
    """
    
    def __init__(self,
                 policy,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 causal_loss_coef: float = 0.1,
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
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=lr * 0.1
        )
        
        # Experience buffer for causal learning
        self.causal_experience_buffer = deque(maxlen=10000)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'causal_loss': [],
            'total_loss': [],
            'learning_rate': []
        }
    
    def select_action(self, 
                     state: torch.Tensor, 
                     instruction_tokens: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using the policy
        
        Returns:
            action (int): Selected action
            log_prob (torch.Tensor): Log probability of action
            value (torch.Tensor): State value estimate
        """
        with torch.no_grad():
            # Check if policy supports instruction tokens
            if hasattr(self.policy, 'encode_language'):
                # Enhanced transformer policy
                outputs = self.policy(state, instruction_tokens)
            else:
                # Baseline policies (CNN, MLP, LSTM without instruction support)
                if hasattr(self.policy, 'forward'):
                    if 'hidden' in self.policy.forward.__code__.co_varnames:
                        # LSTM baseline
                        outputs = self.policy.forward(state, getattr(self, '_hidden', None))
                        if 'hidden' in outputs:
                            self._hidden = outputs['hidden']
                    else:
                        # CNN, MLP baselines
                        outputs = self.policy.forward(state)
                else:
                    # Random baseline
                    outputs = self.policy.forward(state)
            
            if deterministic:
                action = torch.argmax(outputs['action_logits'], dim=-1)
                dist = Categorical(logits=outputs['action_logits'])
                log_prob = dist.log_prob(action)
            else:
                dist = Categorical(logits=outputs['action_logits'])
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob, outputs['value']
    
    def compute_gae(self, 
                   rewards: List[float], 
                   values: List[torch.Tensor], 
                   masks: List[float], 
                   next_value: torch.Tensor) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation
        
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        values = values + [next_value]
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            
            returns.insert(0, gae + values[step].item())
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def compute_causal_loss(self, 
                           states: torch.Tensor,
                           actions: torch.Tensor,
                           next_states: torch.Tensor,
                           causal_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute auxiliary causal prediction loss
        This helps the model learn causal relationships
        """
        if not hasattr(self.policy, 'predict_causal_effect'):
            return torch.tensor(0.0, device=states.device)
        
        # Predict causal effects
        causal_predictions = self.policy.predict_causal_effect(states, actions)
        
        if causal_targets is None:
            # Create pseudo-targets from state transitions
            # This is a simplified version - in practice you'd have more sophisticated causal targets
            state_diffs = (next_states - states).view(states.shape[0], -1)
            causal_targets = torch.sum(state_diffs != 0, dim=1).float()
        
        # Compute loss (cross-entropy for discrete targets, MSE for continuous)
        if causal_targets.dtype == torch.long:
            loss = F.cross_entropy(causal_predictions, causal_targets)
        else:
            # Convert predictions to single value for MSE
            pred_values = torch.sum(F.softmax(causal_predictions, dim=-1) * torch.arange(causal_predictions.shape[-1], device=causal_predictions.device), dim=-1)
            loss = F.mse_loss(pred_values, causal_targets)
        
        return loss
    
    def update(self, trajectories: Dict[str, List]) -> Dict[str, float]:
        """
        Update policy using PPO with causal auxiliary loss
        
        Args:
            trajectories: Dictionary containing trajectory data
                - states: List of state tensors
                - actions: List of actions
                - log_probs: List of log probabilities
                - returns: List of returns
                - advantages: List of advantages
                - instruction_tokens: List of instruction tokens (optional)
                - next_states: List of next states (for causal loss)
        
        Returns:
            Dictionary of loss components
        """
        # Prepare data
        states = torch.stack(trajectories['states'])
        actions = torch.tensor(trajectories['actions'], dtype=torch.long)
        old_log_probs = torch.stack(trajectories['log_probs'])
        returns = torch.tensor(trajectories['returns'], dtype=torch.float32)
        advantages = torch.tensor(trajectories['advantages'], dtype=torch.float32)
        
        # Optional components
        instruction_tokens = None
        if 'instruction_tokens' in trajectories and trajectories['instruction_tokens'][0] is not None:
            instruction_tokens = torch.stack(trajectories['instruction_tokens'])
        
        next_states = None
        if 'next_states' in trajectories:
            next_states = torch.stack(trajectories['next_states'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare dataset
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        # Store loss components
        policy_losses = []
        value_losses = []
        entropy_losses = []
        causal_losses = []
        total_losses = []
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                batch_instruction_tokens = None
                if instruction_tokens is not None:
                    batch_instruction_tokens = instruction_tokens[batch_indices]
                
                batch_next_states = None
                if next_states is not None:
                    batch_next_states = next_states[batch_indices]
                
                # Forward pass - handle different policy types
                if hasattr(self.policy, 'encode_language'):
                    # Enhanced transformer policy
                    outputs = self.policy(batch_states, batch_instruction_tokens)
                else:
                    # Baseline policies
                    if hasattr(self.policy, 'forward'):
                        if 'hidden' in self.policy.forward.__code__.co_varnames:
                            # LSTM baseline
                            outputs = self.policy.forward(batch_states, None)  # No hidden state in batch updates
                        else:
                            # CNN, MLP baselines
                            outputs = self.policy.forward(batch_states)
                    else:
                        # Random baseline
                        outputs = self.policy.forward(batch_states)
                
                # Compute policy loss
                dist = Categorical(logits=outputs['action_logits'])
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss - fix tensor shape mismatch
                predicted_values = outputs['value'].squeeze()
                if predicted_values.dim() == 0:
                    predicted_values = predicted_values.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                value_loss = F.mse_loss(predicted_values, batch_returns)
                
                # Compute causal loss
                causal_loss = torch.tensor(0.0, device=batch_states.device)
                if batch_next_states is not None:
                    causal_loss = self.compute_causal_loss(batch_states, batch_actions, batch_next_states)
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coef * value_loss - 
                             self.entropy_coef * entropy + 
                             self.causal_loss_coef * causal_loss)
                
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
                total_losses.append(total_loss.item())
        
        # Update learning rate
        self.scheduler.step()
        
        # Compute average losses
        loss_dict = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'causal_loss': np.mean(causal_losses),
            'total_loss': np.mean(total_losses),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        # Update training statistics
        for key, value in loss_dict.items():
            self.training_stats[key].append(value)
        
        return loss_dict
    
    def add_causal_experience(self, 
                            state: torch.Tensor, 
                            action: int, 
                            next_state: torch.Tensor, 
                            causal_effect: Dict):
        """Add experience to causal learning buffer"""
        experience = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'causal_effect': causal_effect
        }
        self.causal_experience_buffer.append(experience)
    
    def train_causal_predictor(self, batch_size: int = 32, num_batches: int = 10):
        """
        Additional training on causal prediction task
        This is called periodically to improve causal understanding
        """
        if len(self.causal_experience_buffer) < batch_size:
            return {}
        
        losses = []
        
        for _ in range(num_batches):
            # Sample batch from causal experience
            batch = random.sample(self.causal_experience_buffer, batch_size)
            
            states = torch.stack([exp['state'] for exp in batch])
            actions = torch.tensor([exp['action'] for exp in batch])
            next_states = torch.stack([exp['next_state'] for exp in batch])
            
            # Compute causal loss
            causal_loss = self.compute_causal_loss(states, actions, next_states)
            
            # Update only causal components
            self.optimizer.zero_grad()
            (self.causal_loss_coef * causal_loss).backward()
            self.optimizer.step()
            
            losses.append(causal_loss.item())
        
        return {'causal_predictor_loss': np.mean(losses)}
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats,
            'causal_experience_buffer': list(self.causal_experience_buffer)
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        if 'causal_experience_buffer' in checkpoint:
            self.causal_experience_buffer = deque(checkpoint['causal_experience_buffer'], maxlen=10000)
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.training_stats
    
    def reset_stats(self):
        """Reset training statistics"""
        for key in self.training_stats:
            self.training_stats[key] = []

class CausalExperienceCollector:
    """
    Utility class for collecting experiences with causal annotations
    """
    
    def __init__(self, environment):
        self.env = environment
        self.causal_rules = getattr(environment, 'causal_rules', [])
    
    def collect_trajectory(self, 
                          agent: EnhancedPPOAgent, 
                          max_steps: int = 100,
                          instruction_tokens: Optional[torch.Tensor] = None) -> Dict[str, List]:
        """
        Collect a complete trajectory with causal annotations
        """
        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'masks': [],
            'next_states': [],
            'causal_events': [],
            'instruction_tokens': []
        }
        
        state, _ = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(state_tensor, instruction_tokens)
            
            # Store current state info
            trajectory['states'].append(state_tensor.squeeze(0))
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value)
            trajectory['instruction_tokens'].append(instruction_tokens.squeeze(0) if instruction_tokens is not None else None)
            
            # Take action in environment
            next_state, reward, done, truncated, info = self.env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.long).unsqueeze(0)
            
            # Detect causal events
            causal_event = self._detect_causal_event(state, action, next_state, info)
            
            # Store transition info
            trajectory['next_states'].append(next_state_tensor.squeeze(0))
            trajectory['rewards'].append(reward)
            trajectory['masks'].append(0.0 if done else 1.0)
            trajectory['causal_events'].append(causal_event)
            
            # Add to causal experience buffer
            if causal_event['occurred']:
                agent.add_causal_experience(state_tensor.squeeze(0), action, next_state_tensor.squeeze(0), causal_event)
            
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
    
    def _detect_causal_event(self, state, action, next_state, info) -> Dict:
        """
        Detect if a causal event occurred in this transition
        """
        causal_event = {
            'occurred': False,
            'rule_triggered': None,
            'effect_type': None,
            'effect_position': None
        }
        
        # Check if any causal rules were triggered
        if hasattr(self.env, '_count_triggered_rules'):
            # Use environment's causal tracking if available
            triggered_count = info.get('causal_rules_triggered', 0)
            if triggered_count > 0:
                causal_event['occurred'] = True
                causal_event['rule_triggered'] = triggered_count
        
        # Simple heuristic: check for significant state changes
        state_diff = np.sum(state != next_state)
        if state_diff > 1:  # More than just agent movement
            causal_event['occurred'] = True
            causal_event['effect_type'] = 'state_change'
        
        return causal_event
    
    def collect_batch(self, 
                     agent: EnhancedPPOAgent,
                     num_trajectories: int,
                     max_steps_per_trajectory: int = 100) -> List[Dict[str, List]]:
        """
        Collect a batch of trajectories
        """
        trajectories = []
        
        for _ in range(num_trajectories):
            trajectory = self.collect_trajectory(agent, max_steps_per_trajectory)
            trajectories.append(trajectory)
        
        return trajectories