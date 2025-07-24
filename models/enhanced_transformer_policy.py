import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CausalAttentionBlock(nn.Module):
    """Transformer block with causal attention and explicit causal reasoning"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Causal reasoning components
        self.causal_gate = nn.Linear(d_model, d_model)
        self.causal_transform = nn.Linear(d_model, d_model)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Causal reasoning enhancement
        causal_gate = torch.sigmoid(self.causal_gate(src))
        causal_info = self.causal_transform(src)
        src = src + causal_gate * causal_info
        
        # Feed forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class EnhancedTransformerPolicy(nn.Module):
    """
    Enhanced Transformer-based policy with:
    - Causal attention mechanisms
    - Language instruction processing
    - Memory for causal relationships
    - Explicit causal reasoning
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 num_objects: int,
                 action_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_sequence_length: int = 200,
                 vocab_size: int = 1000):
        
        super().__init__()
        
        self.grid_height, self.grid_width = grid_size
        self.num_objects = max(num_objects, 20)  # Ensure minimum vocabulary size
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        # State encoding - use larger vocabulary to handle any object IDs
        self.state_embedding = nn.Embedding(self.num_objects, d_model)
        self.position_embedding = nn.Linear(2, d_model)  # For spatial positions
        
        # Language processing
        self.language_embedding = nn.Embedding(vocab_size, d_model)
        self.language_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        
        # Transformer backbone
        self.pos_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.transformer_layers = nn.ModuleList([
            CausalAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Causal reasoning module
        self.causal_memory = nn.Parameter(torch.randn(100, d_model))  # Learnable causal memory
        self.causal_query = nn.Linear(d_model, d_model)
        self.causal_key = nn.Linear(d_model, d_model)
        self.causal_value = nn.Linear(d_model, d_model)
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )
        
        # Causal prediction head (for auxiliary loss)
        self.causal_prediction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, self.num_objects)
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0.0, 0.02)
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode grid state into sequence of embeddings
        state: (batch_size, height, width) - grid of object IDs
        """
        batch_size, height, width = state.shape
        
        # Flatten spatial dimensions
        state_flat = state.view(batch_size, -1)  # (batch_size, height*width)
        
        # Clamp state values to valid range
        state_flat = torch.clamp(state_flat, 0, self.num_objects - 1)
        
        # Create position encodings
        positions = []
        for i in range(height):
            for j in range(width):
                positions.append([i / height, j / width])  # Normalized positions
        
        positions = torch.tensor(positions, dtype=torch.float, device=state.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, height*width, 2)
        
        # Embed objects and positions
        object_embeds = self.state_embedding(state_flat)  # (batch_size, height*width, d_model)
        position_embeds = self.position_embedding(positions)  # (batch_size, height*width, d_model)
        
        # Combine embeddings
        state_embeds = object_embeds + position_embeds
        
        return state_embeds
    
    def encode_language(self, instruction_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode language instruction
        instruction_tokens: (batch_size, seq_len) - tokenized instruction
        """
        if instruction_tokens is None:
            return None
        
        # Clamp instruction tokens to valid vocabulary range
        vocab_size = self.language_embedding.num_embeddings
        instruction_tokens = torch.clamp(instruction_tokens, 0, vocab_size - 1)
        
        # Embed tokens
        lang_embeds = self.language_embedding(instruction_tokens)
        
        # Encode with LSTM
        lang_encoded, _ = self.language_encoder(lang_embeds)
        
        # Use last hidden state as instruction representation
        instruction_repr = lang_encoded[:, -1, :]  # (batch_size, d_model)
        
        return instruction_repr
    
    def causal_reasoning(self, state_embeds: torch.Tensor, instruction_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply causal reasoning using attention over causal memory
        """
        batch_size, seq_len, d_model = state_embeds.shape
        
        # Create query from current state
        if instruction_repr is not None:
            # Combine state and instruction for query
            pooled_state = state_embeds.mean(dim=1)  # (batch_size, d_model)
            query_input = pooled_state + instruction_repr
        else:
            query_input = state_embeds.mean(dim=1)
        
        query = self.causal_query(query_input).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Create keys and values from causal memory
        causal_memory_expanded = self.causal_memory.unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.causal_key(causal_memory_expanded)  # (batch_size, memory_size, d_model)
        values = self.causal_value(causal_memory_expanded)
        
        # Attention over causal memory
        attention_scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(d_model)
        attention_weights = F.softmax(attention_scores, dim=-1)
        causal_context = torch.bmm(attention_weights, values)  # (batch_size, 1, d_model)
        
        # Broadcast causal context to all sequence positions
        causal_context = causal_context.expand(-1, seq_len, -1)
        
        return causal_context
    
    def forward(self, 
                state: torch.Tensor, 
                instruction_tokens: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: (batch_size, height, width) - current state
            instruction_tokens: (batch_size, seq_len) - tokenized instruction
            return_attention: whether to return attention weights
        
        Returns:
            Dict with 'action_logits', 'value', 'causal_prediction', and optionally 'attention'
        """
        batch_size = state.shape[0]
        
        # Encode state
        state_embeds = self.encode_state(state)  # (batch_size, height*width, d_model)
        
        # Encode instruction
        instruction_repr = self.encode_language(instruction_tokens) if instruction_tokens is not None else None
        
        # Apply causal reasoning
        causal_context = self.causal_reasoning(state_embeds, instruction_repr)
        
        # Combine state and causal context
        enhanced_embeds = state_embeds + causal_context
        
        # Add positional encoding
        enhanced_embeds = self.pos_encoder(enhanced_embeds.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer layers
        hidden = enhanced_embeds
        attention_weights = []
        
        for layer in self.transformer_layers:
            hidden = layer(hidden)
            if return_attention:
                # Note: This is simplified - in practice you'd extract attention from the layer
                attention_weights.append(torch.ones(batch_size, hidden.shape[1], hidden.shape[1]))
        
        # Pool for final representation
        pooled_hidden = hidden.mean(dim=1)  # (batch_size, d_model)
        
        # Generate outputs
        action_logits = self.action_head(pooled_hidden)
        value = self.value_head(pooled_hidden)
        causal_prediction = self.causal_prediction_head(pooled_hidden)
        
        outputs = {
            'action_logits': action_logits,
            'value': value,
            'causal_prediction': causal_prediction
        }
        
        if return_attention:
            outputs['attention'] = attention_weights
        
        return outputs
    
    def get_action_distribution(self, state: torch.Tensor, instruction_tokens: Optional[torch.Tensor] = None) -> Categorical:
        """Get action distribution for sampling"""
        outputs = self.forward(state, instruction_tokens)
        return Categorical(logits=outputs['action_logits'])
    
    def predict_causal_effect(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict the causal effect of taking an action in the current state
        This is used for auxiliary learning to improve causal understanding
        """
        outputs = self.forward(state)
        return outputs['causal_prediction']