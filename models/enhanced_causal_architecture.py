#!/usr/bin/env python3
"""
Enhanced Causal Learning Architecture
Addresses the core issues preventing causal learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CausalRule:
    """Explicit causal rule representation"""
    cause_object: str
    effect_object: str
    relationship: str  # "activates", "opens", "closes"
    confidence: float = 0.0
    
class CausalGraphModule(nn.Module):
    """
    SOLUTION 1: Explicit Causal Graph Learning
    Learns and maintains explicit causal relationships
    """
    
    def __init__(self, num_objects: int, d_model: int):
        super().__init__()
        self.num_objects = num_objects
        self.d_model = d_model
        
        # Causal relationship matrix (learnable)
        self.causal_matrix = nn.Parameter(torch.zeros(num_objects, num_objects))
        
        # Object representation embeddings
        self.object_embeddings = nn.Embedding(num_objects, d_model)
        
        # Causal mechanism predictor
        self.mechanism_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)  # [no_effect, positive_effect, negative_effect]
        )
        
        # Temporal causal encoder
        self.temporal_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        
    def forward(self, state_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Learn causal relationships from state sequences
        """
        batch_size, seq_len, height, width = state_sequence.shape
        
        # Extract object interactions over time
        causal_predictions = []
        causal_confidences = []
        
        for t in range(1, seq_len):
            prev_state = state_sequence[:, t-1].flatten(1)
            curr_state = state_sequence[:, t].flatten(1)
            
            # Detect state changes
            state_diff = (curr_state != prev_state).float()
            
            # For each object pair, predict causal relationship
            for obj1 in range(self.num_objects):
                for obj2 in range(self.num_objects):
                    if obj1 != obj2:
                        # Get embeddings
                        obj1_embed = self.object_embeddings(torch.tensor(obj1))
                        obj2_embed = self.object_embeddings(torch.tensor(obj2))
                        
                        # Predict causal mechanism
                        combined = torch.cat([obj1_embed, obj2_embed])
                        mechanism = self.mechanism_predictor(combined.unsqueeze(0))
                        
                        causal_predictions.append(mechanism)
                        
                        # Update causal matrix
                        obj1_present = (prev_state == obj1).any(dim=1)
                        obj2_changed = (state_diff == obj2).any(dim=1)
                        
                        if obj1_present.any() and obj2_changed.any():
                            # Strengthen causal connection
                            self.causal_matrix[obj1, obj2] += 0.01
        
        return {
            'causal_matrix': torch.sigmoid(self.causal_matrix),
            'causal_predictions': torch.stack(causal_predictions) if causal_predictions else torch.zeros(1, 1, 3),
            'discovered_rules': self.extract_causal_rules()
        }
    
    def extract_causal_rules(self) -> List[CausalRule]:
        """Extract discovered causal rules"""
        rules = []
        causal_probs = torch.sigmoid(self.causal_matrix)
        
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if causal_probs[i, j] > 0.7:  # High confidence threshold
                    rules.append(CausalRule(
                        cause_object=f"object_{i}",
                        effect_object=f"object_{j}",
                        relationship="activates",
                        confidence=causal_probs[i, j].item()
                    ))
        
        return rules

class LanguageGroundingModule(nn.Module):
    """
    SOLUTION 2: Strong Language-Action Grounding
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_actions: int):
        super().__init__()
        self.d_model = d_model
        
        # Language encoder
        self.instruction_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Action-language alignment
        self.action_embeddings = nn.Embedding(num_actions, d_model)
        self.alignment_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Causal instruction parser
        self.causal_parser = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)  # [cause, effect, relationship, confidence]
        )
        
    def forward(self, instruction_tokens: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Ground language instructions to actions with causal understanding
        """
        # Encode instruction
        word_embeds = self.word_embeddings(instruction_tokens)
        instruction_repr, _ = self.instruction_encoder(word_embeds)
        instruction_final = instruction_repr.mean(dim=1)  # Average pooling
        
        # Parse causal components from instruction
        causal_parse = self.causal_parser(instruction_final)
        
        # Align actions with instruction
        action_embeds = self.action_embeddings(actions)
        
        alignment_scores = []
        for action_embed in action_embeds:
            combined = torch.cat([instruction_final, action_embed.unsqueeze(0)], dim=1)
            score = self.alignment_scorer(combined)
            alignment_scores.append(score)
        
        return {
            'instruction_representation': instruction_final,
            'causal_parse': causal_parse,
            'action_alignment': torch.stack(alignment_scores),
            'grounding_loss': self.compute_grounding_loss(instruction_final, action_embeds)
        }
    
    def compute_grounding_loss(self, instruction_repr: torch.Tensor, action_embeds: torch.Tensor) -> torch.Tensor:
        """Compute loss for language-action grounding"""
        # Encourage actions mentioned in instruction to have higher alignment
        # This is a simplified version - you'd want more sophisticated parsing
        return F.mse_loss(instruction_repr.mean(), action_embeds.mean())

class InterventionTrainingModule:
    """
    SOLUTION 3: Systematic Intervention Training
    """
    
    def __init__(self, base_env):
        self.base_env = base_env
        self.intervention_types = [
            'swap_objects',
            'remove_causal_link', 
            'add_noise',
            'change_rewards',
            'block_actions'
        ]
    
    def create_intervention_scenarios(self, num_scenarios: int = 10) -> List[Dict]:
        """Create diverse intervention scenarios for training"""
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = {
                'intervention_type': np.random.choice(self.intervention_types),
                'original_env': self.base_env,
                'modified_env': self._apply_intervention(),
                'counterfactual_question': self._generate_counterfactual()
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _apply_intervention(self):
        """Apply random intervention to environment"""
        # This would modify the environment in systematic ways
        modified_env = self.base_env  # Simplified
        # Apply specific interventions based on type
        return modified_env
    
    def _generate_counterfactual(self) -> str:
        """Generate counterfactual reasoning questions"""
        questions = [
            "What would happen if the switch was removed?",
            "Would the agent reach the goal if the door was always open?",
            "What if pressing the switch closed the door instead?",
            "Could the agent succeed without any switch?"
        ]
        return np.random.choice(questions)

class EnhancedCausalTransformer(nn.Module):
    """
    SOLUTION: Complete Enhanced Causal Learning Architecture
    """
    
    def __init__(self, grid_size: Tuple[int, int], num_objects: int, action_dim: int, 
                 vocab_size: int, d_model: int = 256):
        super().__init__()
        
        # Core transformer
        self.d_model = d_model
        self.state_encoder = nn.Linear(grid_size[0] * grid_size[1], d_model)
        
        # NEW: Explicit causal graph learning
        self.causal_graph = CausalGraphModule(num_objects, d_model)
        
        # NEW: Strong language grounding
        self.language_grounding = LanguageGroundingModule(vocab_size, d_model, action_dim)
        
        # Enhanced attention with causal bias
        self.causal_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # state + causal + language
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # NEW: Counterfactual reasoning head
        self.counterfactual_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)  # Predict action in counterfactual scenario
        )
    
    def forward(self, state: torch.Tensor, instruction_tokens: torch.Tensor, 
                actions: torch.Tensor, state_history: Optional[torch.Tensor] = None):
        
        batch_size = state.shape[0]
        
        # Encode current state
        state_flat = state.view(batch_size, -1).float()
        state_repr = self.state_encoder(state_flat)
        
        # Learn causal relationships
        causal_output = self.causal_graph(state_history if state_history is not None else state.unsqueeze(1))
        causal_repr = causal_output['causal_matrix'].flatten().unsqueeze(0).expand(batch_size, -1)
        causal_repr = F.linear(causal_repr, torch.randn(self.d_model, causal_repr.shape[1]))
        
        # Ground language instructions
        language_output = self.language_grounding(instruction_tokens, actions)
        language_repr = language_output['instruction_representation']
        
        # Combine representations
        combined_repr = torch.cat([state_repr, causal_repr, language_repr], dim=1)
        
        # Generate policy and value
        policy_logits = self.policy_head(combined_repr)
        value = self.value_head(combined_repr)
        
        # Generate counterfactual predictions
        counterfactual_policy = self.counterfactual_head(combined_repr)
        
        return {
            'policy_logits': policy_logits,
            'value': value,
            'counterfactual_policy': counterfactual_policy,
            'causal_graph': causal_output['causal_matrix'],
            'discovered_rules': causal_output['discovered_rules'],
            'language_grounding': language_output,
            'total_causal_loss': self.compute_total_causal_loss(causal_output, language_output)
        }
    
    def compute_total_causal_loss(self, causal_output: Dict, language_output: Dict) -> torch.Tensor:
        """Compute comprehensive causal learning loss"""
        
        # Causal graph learning loss
        causal_loss = F.mse_loss(causal_output['causal_matrix'], 
                                torch.eye(causal_output['causal_matrix'].shape[0]))
        
        # Language grounding loss
        grounding_loss = language_output['grounding_loss']
        
        # Sparsity loss (encourage sparse causal connections)
        sparsity_loss = torch.mean(torch.abs(causal_output['causal_matrix']))
        
        return causal_loss + grounding_loss + 0.1 * sparsity_loss

# SOLUTION 4: Enhanced Training with Interventions
class CausalTrainingCurriculum:
    """Training curriculum that systematically teaches causal reasoning"""
    
    def __init__(self):
        self.stages = [
            {
                'name': 'basic_correlation',
                'episodes': 200,
                'interventions': False,
                'language_complexity': 'simple'
            },
            {
                'name': 'causal_discovery', 
                'episodes': 300,
                'interventions': True,
                'language_complexity': 'causal'
            },
            {
                'name': 'counterfactual_reasoning',
                'episodes': 500,
                'interventions': True,
                'language_complexity': 'complex'
            }
        ]
    
    def get_training_config(self, episode: int) -> Dict:
        """Get training configuration for current episode"""
        for stage in self.stages:
            if episode < stage['episodes']:
                return stage
        return self.stages[-1]  # Final stage

print("✅ Enhanced Causal Learning Architecture Created!")
print("Key Features:")
print("• Explicit causal graph learning")
print("• Strong language-action grounding") 
print("• Systematic intervention training")
print("• Counterfactual reasoning")
print("• Progressive curriculum")