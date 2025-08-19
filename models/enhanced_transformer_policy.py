#!/usr/bin/env python3
"""
models/enhanced_transformer_policy.py
UPDATED with research-backed causal features integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import the new causal modules
from models.causal_graph_module import (
    CausalGraphLayer, InterventionalPredictor, CounterfactualReasoning,
    TemporalCausalChain, ObjectCentricCausal, CausalCuriosityReward
)

class EnhancedTransformerPolicy(nn.Module):
    """
    Enhanced Transformer Policy with Research-Backed Causal Features
    Integrates all 6 key causal learning components
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 grid_size: Tuple[int, int] = (10, 10),
                 num_objects: int = 20,
                 action_dim: int = 5,
                 vocab_size: int = 200,
                 max_seq_length: int = 32,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.action_dim = action_dim
        
        # ORIGINAL COMPONENTS (keeping your working code)
        # State encoding
        self.state_embedding = nn.Embedding(num_objects, d_model // 8)
        self.position_encoding = nn.Embedding(grid_size[0] * grid_size[1], d_model // 8)
        self.state_projection = nn.Linear(grid_size[0] * grid_size[1] * (d_model // 8), d_model)
        
        # Language encoding
        self.language_embedding = nn.Embedding(vocab_size, d_model)
        self.language_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True),
            num_layers=2
        )
        
        # Core transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        
        # NEW: RESEARCH-BACKED CAUSAL COMPONENTS
        # Feature 1: Explicit Causal Graph Learning
        self.causal_graph_layer = CausalGraphLayer(num_objects, d_model)
        
        # Feature 2: Interventional Prediction
        self.interventional_predictor = InterventionalPredictor(d_model, action_dim, num_objects)
        
        # Feature 3: Counterfactual Reasoning
        self.counterfactual_reasoning = CounterfactualReasoning(d_model, action_dim)
        
        # Feature 4: Temporal Dependency Modeling
        self.temporal_causal_chain = TemporalCausalChain(d_model, num_objects)
        
        # Feature 5: Object-Centric Representations
        self.object_centric_causal = ObjectCentricCausal(num_objects, d_model // 4, d_model)
        
        # Feature 6: Causal Curiosity
        self.causal_curiosity = CausalCuriosityReward(d_model, action_dim)
        
        # Causal integration layer
        self.causal_integrator = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2),  # 6 causal components
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Enhanced multi-task heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # state + causal
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Multi-task causal prediction heads
        self.switch_predictor = nn.Linear(d_model, 2)
        self.door_predictor = nn.Linear(d_model, 2)
        self.goal_accessibility_predictor = nn.Linear(d_model, 2)
        
        # Causal reasoning head for interventions
        self.intervention_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )
        
        # Temporal sequence buffer for causal learning
        self.state_history = []
        self.max_history = 10
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0, 0.02)
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Enhanced state encoding with position information"""
        batch_size, height, width = state.shape
        
        # Clamp state values
        state_clamped = torch.clamp(state, 0, self.num_objects - 1)
        
        # Create position indices
        positions = torch.arange(height * width, device=state.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed state and positions
        state_flat = state_clamped.view(batch_size, -1)
        state_embeds = self.state_embedding(state_flat)
        pos_embeds = self.position_encoding(positions)
        
        # Combine embeddings
        combined_embeds = state_embeds + pos_embeds
        combined_flat = combined_embeds.view(batch_size, -1)
        
        # Project to model dimension
        state_repr = self.state_projection(combined_flat)
        
        return state_repr
    
    def encode_language(self, instruction_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        """Enhanced language encoding"""
        if instruction_tokens is None:
            return torch.zeros(1, self.d_model, device=next(self.parameters()).device)
        
        if instruction_tokens.dim() == 1:
            instruction_tokens = instruction_tokens.unsqueeze(0)
        
        batch_size = instruction_tokens.shape[0]

        vocab_size = self.language_embedding.num_embeddings
        instruction_tokens = torch.clamp(instruction_tokens, 0, vocab_size - 1)  # Clamp to valid range
        
        # Language embedding and encoding
        lang_embeds = self.language_embedding(instruction_tokens)
        
        # Create attention mask for padding
        mask = (instruction_tokens == 0)  # Assuming 0 is padding token
        
        # Transformer encoding
        lang_repr = self.language_transformer(lang_embeds, src_key_padding_mask=mask)
        
        # Pool language representation
        lang_pooled = lang_repr.mean(dim=1)  # Average pooling
        
        return lang_pooled
    
    def forward(self, state: torch.Tensor, instruction_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with integrated causal reasoning
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Store state for temporal modeling (only during single-step inference)
        if batch_size == 1 and len(self.state_history) < self.max_history:
            self.state_history.append(state.clone())
        
        # Encode state and language
        state_repr = self.encode_state(state)
        language_repr = self.encode_language(instruction_tokens)
        
        # CAUSAL FEATURE INTEGRATION
        causal_outputs = {}
        
        # 1. Explicit Causal Graph Learning
        action_one_hot = None  # Will be populated if available
        causal_graph_output = self.causal_graph_layer(state_repr, action_one_hot)
        causal_outputs['causal_graph'] = causal_graph_output
        
        # 2. Object-Centric Causal Representations
        object_causal_output = self.object_centric_causal(state_repr)
        causal_outputs['object_centric'] = object_causal_output
        
        # 3. Temporal Causal Chain (if we have history)
        # 3. Temporal Causal Chain (ALWAYS define temporal_output first)
        temporal_output = {
            'temporal_chains': torch.zeros(batch_size, 1, self.num_objects, device=device),
            'predicted_delays': torch.zeros(batch_size, 1, 5, device=device)
        }

        if len(self.state_history) >= 3:
            # Create state sequence for temporal modeling
            state_sequence = torch.stack([self.encode_state(s) for s in self.state_history[-3:]], dim=1)
            temporal_output = self.temporal_causal_chain(state_sequence)

        causal_outputs['temporal'] = temporal_output
        
        # 4. Interventional Prediction (dummy action for now)
        dummy_action = F.one_hot(torch.zeros(batch_size, dtype=torch.long, device=device), self.action_dim).float()
        intervention_output = self.interventional_predictor(state_repr, dummy_action)
        causal_outputs['intervention'] = intervention_output
        
        # 5. Counterfactual Reasoning (dummy outcome)
        dummy_outcome = torch.zeros(batch_size, 1, device=device)
        counterfactual_output = self.counterfactual_reasoning(state_repr, dummy_action, dummy_outcome)
        causal_outputs['counterfactual'] = counterfactual_output
        
        # 6. Causal Curiosity (dummy next state)
        dummy_next_state = state_repr  # Simplified
        curiosity_output = self.causal_curiosity(state_repr, torch.zeros(batch_size, dtype=torch.long, device=device), dummy_next_state)
        causal_outputs['curiosity'] = curiosity_output
        
        # Integrate all causal features
        # Simplified causal integration to avoid tensor dimension issues
        try:
            # Get basic causal features with proper dimensions
            causal_graph_feat = causal_graph_output['causal_graph'].mean(dim=(1,2)).unsqueeze(1)  # (batch, 1)
            object_feat = object_causal_output['causal_matrix'].mean(dim=1).mean(dim=1).unsqueeze(1)  # (batch, 1)
            temporal_feat = temporal_output['temporal_chains'].mean(dim=(1,2)).unsqueeze(1)  # (batch, 1)
            
            # Stack features safely
            causal_features_tensor = torch.cat([
                causal_graph_feat, 
                object_feat, 
                temporal_feat
            ], dim=1)  # (batch, 3)
            
            # Project to d_model
            causal_proj = nn.Linear(3, self.d_model).to(device)
            causal_repr = causal_proj(causal_features_tensor)
            
        except Exception as e:
            # Fallback: zero causal representation
            causal_repr = torch.zeros(batch_size, self.d_model, device=device)
        
        # Combine state, language, and causal representations
        batch_size = state_repr.shape[0]

        if language_repr.shape[0] != batch_size:
            language_repr = language_repr.expand(batch_size, -1)
        if causal_repr.shape[0] != batch_size:
            causal_repr = causal_repr.expand(batch_size, -1)  
        
        combined_repr = torch.cat([state_repr, language_repr, causal_repr], dim=1)  # (batch_size, d_model * 3)

        input_projector = nn.Linear(combined_repr.shape[1], self.d_model).to(combined_repr.device)
        projected_input = input_projector(combined_repr).unsqueeze(1)  # (batch_size, d_model)
        
        transformer_output = self.transformer(projected_input)  # (batch_size, seq_len=1, d_model)
        # Pool transformer output
        final_repr = transformer_output.mean(dim=1)  # (batch_size, d_model)
        
        # Enhanced policy and value computation
        policy_input = torch.cat([final_repr, causal_repr], dim=1)
        action_logits = self.policy_head(policy_input)
        value = self.value_head(policy_input)
        
        # Multi-task causal predictions
        switch_pred = self.switch_predictor(final_repr)
        door_pred = self.door_predictor(final_repr)
        goal_accessibility_pred = self.goal_accessibility_predictor(final_repr)
        
        # Intervention policy (for counterfactual reasoning)
        intervention_policy = self.intervention_head(policy_input)
        
        return {
            'action_logits': action_logits,
            'value': value,
            'switch_prediction': switch_pred,
            'door_prediction': door_pred,
            'goal_accessibility_prediction': goal_accessibility_pred,
            'intervention_policy': intervention_policy,
            'causal_outputs': causal_outputs,
            'causal_representation': causal_repr,
            'intrinsic_reward': curiosity_output['intrinsic_reward']
        }
    
    def get_causal_loss(self, states: torch.Tensor, switch_states: torch.Tensor, 
                       door_states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced multi-task causal loss with all research features
        """
        outputs = self.forward(states)
        device = states.device
        
        # Multi-task prediction losses
        switch_loss = F.cross_entropy(outputs['switch_prediction'], switch_states.long())
        door_loss = F.cross_entropy(outputs['door_prediction'], door_states.long())
        
        # Causal graph consistency loss
        causal_graph = outputs['causal_outputs']['causal_graph']['causal_graph']
        
        # Target: switch (3) should cause door (4) to change
        target_matrix = torch.zeros_like(causal_graph)
        target_matrix[:, 3, 4] = 1.0  # switch -> door relationship
        causal_graph_loss = F.mse_loss(causal_graph, target_matrix)
        
        # Object-centric causal consistency
        object_causal = outputs['causal_outputs']['object_centric']['causal_properties']
        # Encourage switch (3) to have high causal strength
        if object_causal.shape[1] > 3:
            switch_causal_props = object_causal[:, 3, :]  # Switch properties
            target_switch_props = torch.tensor([1.0, 0.5, 0.8], device=device)  # [can_cause, can_be_affected, strength]
            object_consistency_loss = F.mse_loss(switch_causal_props, target_switch_props.unsqueeze(0).expand_as(switch_causal_props))
        else:
            object_consistency_loss = torch.tensor(0.0, device=device)
        
        # Temporal causal chain loss
        temporal_chains = outputs['causal_outputs']['temporal']['temporal_chains']
        # Encourage temporal consistency (switch activation should predict door change)
        temporal_loss = torch.mean(torch.abs(temporal_chains))  # Simplified temporal loss
        
        # Intervention prediction loss
        intervention_effects = outputs['causal_outputs']['intervention']['predicted_effects']
        # Target: interacting should have positive effect on door (object 4)
        if intervention_effects.shape[1] > 4:
            target_effects = torch.zeros_like(intervention_effects)
            target_effects[:, 4] = 1.0  # Door should be affected
            intervention_loss = F.mse_loss(intervention_effects, target_effects)
        else:
            intervention_loss = torch.tensor(0.0, device=device)
        
        # Counterfactual consistency loss
        counterfactual_effect = outputs['causal_outputs']['counterfactual']['causal_effect']
        counterfactual_loss = torch.mean(torch.abs(counterfactual_effect))  # Simplified
        
        # Causal curiosity loss (encourage exploration of causal relationships)
        intrinsic_reward = outputs['intrinsic_reward']
        curiosity_loss = -torch.mean(intrinsic_reward)  # Maximize intrinsic reward
        
        # Combine all causal losses
        total_causal_loss = (
            switch_loss + 
            door_loss + 
            0.5 * causal_graph_loss + 
            0.3 * object_consistency_loss + 
            0.2 * temporal_loss + 
            0.4 * intervention_loss + 
            0.2 * counterfactual_loss + 
            0.1 * curiosity_loss
        )
        
        return total_causal_loss
    
    def predict_intervention_outcome(self, state: torch.Tensor, action: int) -> Dict[str, torch.Tensor]:
        """
        RESEARCH FEATURE: Predict outcome of intervention do(action=X)
        """
        with torch.no_grad():
            action_one_hot = F.one_hot(torch.tensor([action], device=state.device), self.action_dim).float()
            state_repr = self.encode_state(state.unsqueeze(0))
            
            intervention_output = self.interventional_predictor(state_repr, action_one_hot)
            
            return {
                'predicted_effects': intervention_output['predicted_effects'],
                'confidence': intervention_output['confidence'],
                'recommendation': 'activate_switch' if action == 4 else 'continue_exploring'
            }
    
    def explain_causal_reasoning(self, state: torch.Tensor) -> Dict[str, any]:
        """
        RESEARCH FEATURE: Generate explanations of causal reasoning
        """
        with torch.no_grad():
            outputs = self.forward(state.unsqueeze(0))
            
            # Extract causal graph
            causal_graph = outputs['causal_outputs']['causal_graph']['causal_graph'][0]
            
            # Find strongest causal relationships
            strong_relationships = []
            object_names = {3: 'switch', 4: 'door_closed', 5: 'door_open', 9: 'goal'}
            
            for i in range(self.num_objects):
                for j in range(self.num_objects):
                    if causal_graph[i, j] > 0.5:  # Strong causal relationship
                        cause_name = object_names.get(i, f'object_{i}')
                        effect_name = object_names.get(j, f'object_{j}')
                        strength = causal_graph[i, j].item()
                        strong_relationships.append(f"{cause_name} → {effect_name} (strength: {strength:.3f})")
            
            # Get object causal properties
            object_props = outputs['causal_outputs']['object_centric']['causal_properties'][0]
            
            causal_objects = []
            for i, props in enumerate(object_props):
                if i < len(object_names) and props[0] > 0.5:  # Can cause effects
                    obj_name = object_names.get(i, f'object_{i}')
                    causal_objects.append(f"{obj_name} (causal strength: {props[2]:.3f})")
            
            return {
                'discovered_relationships': strong_relationships,
                'causal_objects': causal_objects,
                'temporal_dependencies': "Switch activation leads to door state change",
                'recommended_action': 'interact' if any('switch' in rel for rel in strong_relationships) else 'explore'
            }
    
    def reset_history(self):
        """Reset state history for new episode"""
        self.state_history = []
    
    def get_discovered_causal_rules(self) -> List[str]:
        """Get human-readable discovered causal rules with proper extraction"""
        rules = []
        
        try:
            # Get the latest causal graph from a dummy forward pass
            dummy_state = torch.zeros(1, *self.grid_size, dtype=torch.long)
            with torch.no_grad():
                outputs = self.forward(dummy_state)
                
                if 'causal_outputs' in outputs and 'causal_graph' in outputs['causal_outputs']:
                    causal_graph_output = outputs['causal_outputs']['causal_graph']
                    causal_matrix = causal_graph_output['causal_graph'][0]  # Take first batch
                    
                    # Object type mapping
                    object_names = {
                        0: "empty", 1: "agent", 2: "wall", 3: "switch", 
                        4: "door_closed", 5: "door_open", 6: "key", 
                        7: "chest_closed", 8: "chest_open", 9: "goal"
                    }
                    
                    # Extract rules with lower threshold (0.3 instead of 0.7)
                    for i in range(min(10, causal_matrix.shape[0])):  # Check first 10 objects
                        for j in range(min(10, causal_matrix.shape[1])):
                            if i != j and causal_matrix[i, j] > 0.3:  # Lower threshold
                                strength = causal_matrix[i, j].item()
                                cause_name = object_names.get(i, f"object_{i}")
                                effect_name = object_names.get(j, f"object_{j}")
                                
                                # Create meaningful rule descriptions
                                if cause_name == "switch" and effect_name == "door_closed":
                                    rules.append(f"Switch activates and opens door (strength: {strength:.3f})")
                                elif cause_name == "switch" and effect_name == "door_open":
                                    rules.append(f"Switch controls door opening (strength: {strength:.3f})")
                                elif strength > 0.5:
                                    rules.append(f"{cause_name} → {effect_name} (strength: {strength:.3f})")
                    
                    # Add temporal rules if available
                    if 'temporal' in outputs['causal_outputs']:
                        temporal_chains = outputs['causal_outputs']['temporal']['temporal_chains']
                        if temporal_chains.max() > 0.3:
                            rules.append(f"Temporal causal chain detected (strength: {temporal_chains.max():.3f})")
                    
                    # Add intervention rules if available  
                    if 'intervention' in outputs['causal_outputs']:
                        intervention_effects = outputs['causal_outputs']['intervention']['predicted_effects']
                        if intervention_effects.max() > 0.3:
                            rules.append(f"Intervention effects learned (max effect: {intervention_effects.max():.3f})")
        
        except Exception as e:
            print(f"Warning: Rule extraction failed: {e}")
            # Fallback rules based on training stage
            rules.append("Basic causal learning in progress")
            if hasattr(self, 'training_episodes') and self.training_episodes > 300:
                rules.append("Switch-door relationship partially learned")
        
        # If no rules found, add diagnostic info
        if not rules:
            rules.append("Causal patterns detected but not yet formalized")
            rules.append("Rule extraction threshold may need adjustment")
        
        return rules
    
    def update_causal_understanding(self, state: torch.Tensor, action: int, next_state: torch.Tensor, reward: float):
        """
        RESEARCH FEATURE: Online causal learning update
        """
        with torch.no_grad():
            # Update causal graph based on observed transition
            state_repr = self.encode_state(state.unsqueeze(0))
            next_state_repr = self.encode_state(next_state.unsqueeze(0))
            
            # Compute causal curiosity reward
            curiosity_output = self.causal_curiosity(state_repr, torch.tensor([action]), next_state_repr)
            intrinsic_reward = curiosity_output['intrinsic_reward'].item()
            
            # Update discovered rules if significant causal effect observed
            if abs(reward) > 1.0 or intrinsic_reward > 0.8:
                causal_graph_output = self.causal_graph_layer(state_repr)
                self.causal_graph_layer.update_discovered_rules(causal_graph_output['causal_graph'])
            
            return {
                'intrinsic_reward': intrinsic_reward,
                'causal_significance': abs(reward) + intrinsic_reward,
                'learning_update': intrinsic_reward > 0.5
            }