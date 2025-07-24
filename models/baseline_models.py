import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional

class LSTMBaseline(nn.Module):
    """
    LSTM-based baseline policy (non-causal)
    This serves as our primary comparison baseline
    """
    
    def __init__(self,
                 grid_size: Tuple[int, int],
                 num_objects: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.grid_height, self.grid_width = grid_size
        self.num_objects = num_objects
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # State encoding
        self.state_embedding = nn.Embedding(num_objects, hidden_dim // 4)
        self.state_encoder = nn.Linear(grid_size[0] * grid_size[1] * (hidden_dim // 4), hidden_dim)
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'bias' in name:
                        torch.nn.init.zeros_(param)
                    elif 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode grid state into feature vector"""
        batch_size = state.shape[0]
        
        # Clamp state values to valid embedding range
        state_clamped = torch.clamp(state, 0, self.num_objects - 1)
        
        # Embed each cell
        state_embeds = self.state_embedding(state_clamped)  # (batch_size, height, width, embed_dim)
        
        # Flatten and encode
        state_flat = state_embeds.view(batch_size, -1)
        state_encoded = self.state_encoder(state_flat)
        
        return state_encoded.unsqueeze(1)  # (batch_size, 1, hidden_dim)
    
    def forward(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode state
        state_encoded = self.encode_state(state)
        
        # LSTM forward
        lstm_out, new_hidden = self.lstm(state_encoded, hidden)
        
        # Get outputs
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        action_logits = self.action_head(last_hidden)
        value = self.value_head(last_hidden)
        
        return {
            'action_logits': action_logits,
            'value': value,
            'hidden': new_hidden
        }
    
    def get_action_distribution(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[Categorical, Tuple[torch.Tensor, torch.Tensor]]:
        """Get action distribution and new hidden state"""
        outputs = self.forward(state, hidden)
        return Categorical(logits=outputs['action_logits']), outputs['hidden']

class CNNBaseline(nn.Module):
    """
    CNN-based baseline policy (spatial but non-causal, non-temporal)
    Good for testing spatial reasoning vs causal reasoning
    """
    
    def __init__(self,
                 grid_size: Tuple[int, int],
                 num_objects: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        
        super().__init__()
        
        self.grid_height, self.grid_width = grid_size
        self.num_objects = num_objects
        self.action_dim = action_dim
        
        # State embedding
        self.state_embedding = nn.Embedding(num_objects, 32)
        
        # CNN backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(256 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output heads
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = state.shape[0]
        
        # Clamp state values to valid embedding range
        state_clamped = torch.clamp(state, 0, self.num_objects - 1)
        
        # Embed state
        state_embeds = self.state_embedding(state_clamped)  # (batch_size, height, width, 32)
        state_embeds = state_embeds.permute(0, 3, 1, 2)  # (batch_size, 32, height, width)
        
        # CNN forward
        conv_features = self.conv_layers(state_embeds)  # (batch_size, 256, 4, 4)
        conv_features = conv_features.reshape(batch_size, -1)  # (batch_size, 256*4*4) - use reshape instead of view
        
        # Process features
        features = self.feature_processor(conv_features)
        
        # Generate outputs
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        return {
            'action_logits': action_logits,
            'value': value
        }
    
    def get_action_distribution(self, state: torch.Tensor) -> Categorical:
        """Get action distribution"""
        outputs = self.forward(state)
        return Categorical(logits=outputs['action_logits'])

class MLPBaseline(nn.Module):
    """
    Simple MLP baseline (no spatial or temporal structure)
    Tests whether structure helps at all
    """
    
    def __init__(self,
                 grid_size: Tuple[int, int],
                 num_objects: int,
                 action_dim: int,
                 hidden_dim: int = 512):
        
        super().__init__()
        
        self.grid_height, self.grid_width = grid_size
        self.input_dim = grid_size[0] * grid_size[1]
        
        # Simple MLP
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output heads
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = state.shape[0]
        
        # Flatten state
        state_flat = state.view(batch_size, -1).float()
        
        # Network forward
        features = self.network(state_flat)
        
        # Generate outputs
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        return {
            'action_logits': action_logits,
            'value': value
        }
    
    def get_action_distribution(self, state: torch.Tensor) -> Categorical:
        """Get action distribution"""
        outputs = self.forward(state)
        return Categorical(logits=outputs['action_logits'])

class RandomBaseline:
    """
    Random action baseline for reference
    """
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    def get_action_distribution(self, state: torch.Tensor) -> Categorical:
        """Get uniform random action distribution"""
        batch_size = state.shape[0]
        logits = torch.ones(batch_size, self.action_dim)
        return Categorical(logits=logits)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass (returns random outputs)"""
        batch_size = state.shape[0]
        return {
            'action_logits': torch.ones(batch_size, self.action_dim),
            'value': torch.zeros(batch_size, 1)
        }

def create_baseline_model(model_type: str, grid_size: Tuple[int, int], num_objects: int, action_dim: int, **kwargs):
    """Factory function for creating baseline models"""
    
    if model_type == "lstm":
        return LSTMBaseline(grid_size, num_objects, action_dim, **kwargs)
    elif model_type == "cnn":
        return CNNBaseline(grid_size, num_objects, action_dim, **kwargs)
    elif model_type == "mlp":
        return MLPBaseline(grid_size, num_objects, action_dim, **kwargs)
    elif model_type == "random":
        return RandomBaseline(action_dim)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")

class BaselineComparison:
    """Utility class for comparing multiple baselines"""
    
    def __init__(self, grid_size: Tuple[int, int], num_objects: int, action_dim: int):
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.action_dim = action_dim
        self.models = {}
    
    def add_baseline(self, name: str, model_type: str, **kwargs):
        """Add a baseline model"""
        self.models[name] = create_baseline_model(model_type, self.grid_size, self.num_objects, self.action_dim, **kwargs)
    
    def get_all_baselines(self) -> Dict[str, nn.Module]:
        """Get all baseline models"""
        return self.models
    
    def create_standard_baselines(self):
        """Create standard set of baselines for comparison"""
        self.add_baseline("LSTM", "lstm", hidden_dim=256, num_layers=2)
        self.add_baseline("CNN", "cnn", hidden_dim=256)
        self.add_baseline("MLP", "mlp", hidden_dim=512)
        self.add_baseline("Random", "random")
        
        return self.models