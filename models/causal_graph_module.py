#!/usr/bin/env python3
"""
models/causal_graph_module.py
Enhanced Causal Graph Learning Module - Research-backed implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DiscoveredCausalRule:
    """Represents a discovered causal relationship"""
    cause_object: int
    effect_object: int
    relationship_strength: float
    temporal_delay: int
    intervention_verified: bool = False

class CausalGraphLayer(nn.Module):
    """
    RESEARCH FEATURE 1: Explicit Causal Graph Representation
    Based on "Neural Causal Models" (Bengio et al.)
    """
    
    def __init__(self, num_objects=20, hidden_dim=128, max_delay=5):
        super().__init__()
        self.num_objects = num_objects
        self.hidden_dim = hidden_dim
        self.max_delay = max_delay
        
        # Learnable adjacency matrix: P(effect | cause, action)
        self.causal_adjacency = nn.Parameter(torch.zeros(num_objects, num_objects))
        
        # Causal strength predictor
        self.causal_strength = nn.Linear(hidden_dim, num_objects * num_objects)
        
        # Temporal delay predictor
        self.delay_predictor = nn.Linear(hidden_dim, num_objects * num_objects * max_delay)
        
        # Object-specific causal encoders
        self.object_encoders = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_objects)
        ])
        
        # Intervention effect predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(hidden_dim + num_objects, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects)
        )
        
        # Discovered rules storage
        self.discovered_rules = []
        
    def forward(self, state_embed: torch.Tensor, action_one_hot: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Learn causal relationships from state embeddings
        
        Args:
            state_embed: (batch_size, hidden_dim) state representation
            action_one_hot: (batch_size, num_actions) one-hot action (optional)
        """
        batch_size = state_embed.shape[0]
        
        # Predict causal strength matrix
        causal_logits = self.causal_strength(state_embed)
        causal_probs = F.softmax(causal_logits.view(batch_size, self.num_objects, self.num_objects), dim=-1)
        
        # Predict temporal delays
        delay_logits = self.delay_predictor(state_embed)
        delay_probs = F.softmax(delay_logits.view(batch_size, self.num_objects, self.num_objects, self.max_delay), dim=-1)
        
        # Combine with learnable adjacency matrix
        base_adjacency = torch.sigmoid(self.causal_adjacency).unsqueeze(0).expand(batch_size, -1, -1)
        final_causal_graph = causal_probs * base_adjacency
        
        # Predict intervention effects if action provided
        intervention_effects = None
        if action_one_hot is not None:
            intervention_input = torch.cat([state_embed, action_one_hot], dim=1)
            intervention_effects = self.intervention_predictor(intervention_input)
        
        return {
            'causal_graph': final_causal_graph,
            'temporal_delays': delay_probs,
            'intervention_effects': intervention_effects,
            'adjacency_matrix': base_adjacency
        }
    
    def update_discovered_rules(self, causal_graph: torch.Tensor, threshold: float = 0.7):
        """Update discovered causal rules based on learned graph"""
        avg_graph = causal_graph.mean(dim=0)  # Average across batch
        
        new_rules = []
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if i != j and avg_graph[i, j] > threshold:
                    rule = DiscoveredCausalRule(
                        cause_object=i,
                        effect_object=j,
                        relationship_strength=avg_graph[i, j].item(),
                        temporal_delay=1  # Simplified
                    )
                    new_rules.append(rule)
        
        self.discovered_rules = new_rules
        return new_rules

class InterventionalPredictor(nn.Module):
    """
    RESEARCH FEATURE 2: Interventional Prediction
    Predicts "what if I do action X?" - Based on Pearl's do-calculus
    """
    
    def __init__(self, state_dim=128, action_dim=5, num_objects=20):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        
        # Intervention encoder
        self.intervention_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        
        # Effect predictor for each object
        self.effect_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim // 2),
                nn.ReLU(),
                nn.Linear(state_dim // 2, 1)
            ) for _ in range(num_objects)
        ])
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, num_objects)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict intervention effects: do(action=interact_switch) → P(door_opens)
        """
        batch_size = state.shape[0]
        
        # Encode intervention
        intervention_repr = self.intervention_encoder(torch.cat([state, action], dim=1))
        
        # Predict effects for each object
        effects = []
        for predictor in self.effect_predictors:
            effect = predictor(intervention_repr)
            effects.append(effect)
        
        effects = torch.cat(effects, dim=1)  # (batch_size, num_objects)
        
        # Estimate confidence
        confidence = torch.sigmoid(self.confidence_estimator(intervention_repr))
        
        return {
            'predicted_effects': effects,
            'confidence': confidence,
            'intervention_representation': intervention_repr
        }

class CounterfactualReasoning(nn.Module):
    """
    RESEARCH FEATURE 3: Counterfactual Reasoning
    "What would happen if I hadn't activated the switch?"
    """
    
    def __init__(self, state_dim=128, action_dim=5):
        super().__init__()
        
        # Counterfactual world model
        self.counterfactual_encoder = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        
        # Outcome predictor
        self.outcome_predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1)
        )
        
        # Causal effect estimator
        self.causal_effect_estimator = nn.Sequential(
            nn.Linear(state_dim + 1, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, actual_outcome: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compare actual vs counterfactual outcomes
        """
        # Create counterfactual scenario (no action)
        no_action = torch.zeros_like(action)
        counterfactual_input = torch.cat([state, state, no_action], dim=1)  # state, alt_state, no_action
        
        # Encode counterfactual scenario
        counterfactual_repr = self.counterfactual_encoder(counterfactual_input)
        
        # Predict counterfactual outcome
        counterfactual_outcome = self.outcome_predictor(counterfactual_repr)
        
        # Estimate causal effect
        # Estimate causal effect (fix tensor dimensions)
        if actual_outcome.dim() == 1:
            actual_outcome = actual_outcome.unsqueeze(1)  # Make it 2D
        if counterfactual_repr.dim() == 1:
            counterfactual_repr = counterfactual_repr.unsqueeze(1)

        # Ensure both tensors have same batch size
        if actual_outcome.shape[0] != counterfactual_repr.shape[0]:
            actual_outcome = actual_outcome.expand(counterfactual_repr.shape[0], -1)

        causal_input = torch.cat([counterfactual_repr, actual_outcome], dim=1)
        causal_effect = self.causal_effect_estimator(causal_input)
        
        return {
            'counterfactual_outcome': counterfactual_outcome,
            'causal_effect': causal_effect,
            'factual_vs_counterfactual': actual_outcome - counterfactual_outcome
        }

class TemporalCausalChain(nn.Module):
    """
    RESEARCH FEATURE 4: Temporal Dependency Modeling
    Learn temporal causal chains: switch(t) → door(t+1) → goal_reachable(t+2)
    """
    
    def __init__(self, state_dim=128, num_objects=20, max_delay=5):
        super().__init__()
        self.max_delay = max_delay
        self.num_objects = num_objects
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(state_dim, state_dim, batch_first=True)
        
        # Delay predictor
        self.causal_delay = nn.Linear(state_dim, max_delay)
        
        # Chain predictor
        self.chain_predictor = nn.Sequential(
            nn.Linear(state_dim + max_delay, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, num_objects)
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(state_dim, 4, batch_first=True)
    
    def forward(self, state_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Model temporal causal chains from state sequence
        
        Args:
            state_sequence: (batch_size, seq_len, state_dim)
        """
        batch_size, seq_len, state_dim = state_sequence.shape
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(state_sequence)
        
        # Temporal attention
        attended_states, attention_weights = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # Predict delays for each timestep
        delays = []
        chain_predictions = []
        
        for t in range(seq_len):
            # Predict causal delay
            delay_logits = self.causal_delay(attended_states[:, t])
            delay_probs = F.softmax(delay_logits, dim=1)
            delays.append(delay_probs)
            
            # Predict causal chain
            chain_input = torch.cat([attended_states[:, t], delay_probs], dim=1)
            chain_pred = self.chain_predictor(chain_input)
            chain_predictions.append(chain_pred)
        
        return {
            'temporal_chains': torch.stack(chain_predictions, dim=1),
            'predicted_delays': torch.stack(delays, dim=1),
            'attention_weights': attention_weights,
            'lstm_representations': lstm_out
        }

class ObjectCentricCausal(nn.Module):
    """
    RESEARCH FEATURE 5: Object-Centric Representations
    Each object has causal properties - based on "Object-Centric Learning"
    """
    
    def __init__(self, num_objects=20, object_dim=64, d_model=128):
        super().__init__()
        self.num_objects = num_objects
        self.object_dim = object_dim
        
        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(d_model, object_dim * num_objects),
            nn.ReLU(),
            nn.Linear(object_dim * num_objects, object_dim * num_objects)
        )
        
        # Causal relations attention
        self.causal_relations = nn.MultiheadAttention(object_dim, 4, batch_first=True)
        
        # Object-specific causal properties
        self.causal_properties = nn.ModuleList([
            nn.Sequential(
                nn.Linear(object_dim, object_dim // 2),
                nn.ReLU(),
                nn.Linear(object_dim // 2, 3)  # [can_cause, can_be_affected, causal_strength]
            ) for _ in range(num_objects)
        ])
        
        # Interaction predictor
        self.interaction_predictor = nn.Sequential(
            nn.Linear(object_dim * 2, object_dim),
            nn.ReLU(),
            nn.Linear(object_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Learn object-centric causal representations
        """
        batch_size = state.shape[0]
        
        # Encode objects
        object_repr = self.object_encoder(state)
        objects = object_repr.view(batch_size, self.num_objects, self.object_dim)
        
        # Learn causal relations between objects
        causal_matrix, attention_weights = self.causal_relations(objects, objects, objects)
        
        # Predict causal properties for each object
        causal_props = []
        for i, prop_predictor in enumerate(self.causal_properties):
            props = prop_predictor(objects[:, i])
            causal_props.append(props)
        
        causal_properties = torch.stack(causal_props, dim=1)  # (batch, num_objects, 3)
        
        # Predict all pairwise interactions
        interactions = []
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if i != j:
                    pair_input = torch.cat([objects[:, i], objects[:, j]], dim=1)
                    interaction = self.interaction_predictor(pair_input)
                    interactions.append(interaction)
        
        interaction_matrix = torch.stack(interactions, dim=1).view(batch_size, self.num_objects, self.num_objects-1)
        
        return {
            'object_representations': objects,
            'causal_matrix': causal_matrix,
            'causal_properties': causal_properties,
            'interaction_matrix': interaction_matrix,
            'attention_weights': attention_weights
        }

class CausalCuriosityReward(nn.Module):
    """
    RESEARCH FEATURE 6: Causal Curiosity
    Intrinsic motivation for discovering causal relationships
    Based on "Causal Curiosity" (Sekar et al.)
    """
    
    def __init__(self, state_dim=128, action_dim=5):
        super().__init__()
        
        # Causal effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        
        # Prediction error estimator
        self.error_estimator = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 1)
        )
        
        # Causal novelty detector
        self.novelty_detector = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute intrinsic reward for causal discovery
        """
        # Predict effect of action
        action_one_hot = F.one_hot(action.long(), num_classes=5).float()
        predicted_effect = self.effect_predictor(torch.cat([state, action_one_hot], dim=1))
        
        # Compute prediction error
        error_input = torch.cat([predicted_effect, next_state], dim=1)
        prediction_error = self.error_estimator(error_input)
        
        # Detect causal novelty
        causal_novelty = self.novelty_detector(predicted_effect)
        
        # Compute intrinsic reward
        causal_surprise = F.mse_loss(predicted_effect, next_state, reduction='none').mean(dim=1, keepdim=True)
        intrinsic_reward = torch.exp(-causal_surprise)  # Higher reward for accurate predictions
        
        # Bonus for novel causal relationships
        novelty_bonus = torch.sigmoid(causal_novelty)
        
        total_intrinsic_reward = intrinsic_reward + 0.1 * novelty_bonus
        
        return {
            'intrinsic_reward': total_intrinsic_reward,
            'prediction_error': prediction_error,
            'causal_novelty': causal_novelty,
            'predicted_effect': predicted_effect
        }

def causal_curiosity_reward(predicted_effect: torch.Tensor, actual_effect: torch.Tensor) -> torch.Tensor:
    """
    Standalone causal curiosity reward function
    """
    causal_surprise = F.mse_loss(predicted_effect, actual_effect, reduction='none').mean(dim=1)
    intrinsic_reward = torch.exp(-causal_surprise)
    return intrinsic_reward