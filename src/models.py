"""
Neural network models for multi-modal BCI decoding
"""
import numpy as np
from typing import Tuple, Optional, Dict
import logging

# Safe imports for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from config import MODEL_CONFIG, DEVICE

logger = logging.getLogger(__name__)

# Global flag to indicate if module is functional
MODULE_AVAILABLE = TORCH_AVAILABLE


if MODULE_AVAILABLE:
    class EEGCNNLSTM(nn.Module):
        """
        CNN-LSTM model for EEG signal processing
        """
        
        def __init__(self, config: Optional[Dict] = None):
            super(EEGCNNLSTM, self).__init__()
            
            if config is None:
                config = MODEL_CONFIG['eeg_stream']
            
            self.config = config
            self.input_channels = config['input_channels']
            self.input_length = config['input_length']
            self.conv_filters = config['conv_filters']
            self.conv_kernel = config['conv_kernel']
            self.lstm_units = config['lstm_units']
            self.dropout = config['dropout']
            
            # CNN layers for spatial feature extraction
            self.conv_layers = nn.ModuleList()
            in_channels = 1  # Single input channel
            
            for i, n_filters in enumerate(self.conv_filters):
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels, n_filters, self.conv_kernel, padding=1),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                    nn.Dropout2d(self.dropout)
                )
                self.conv_layers.append(conv_block)
                in_channels = n_filters
            
            # Calculate flattened size after CNN
            self._calculate_conv_output_size()
            
            # LSTM layers for temporal dynamics
            self.lstm_layers = nn.ModuleList()
            in_units = self.conv_output_size
            
            for i, units in enumerate(self.lstm_units):
                lstm_layer = nn.LSTM(
                    input_size=in_units,
                    hidden_size=units,
                    num_layers=1,
                    batch_first=True,
                    dropout=self.dropout if i < len(self.lstm_units) - 1 else 0
                )
                self.lstm_layers.append(lstm_layer)
                in_units = units
            
            self.final_units = self.lstm_units[-1]
            
        def _calculate_conv_output_size(self):
            """Calculate output size after CNN layers"""
            # Create dummy input to calculate size
            dummy_input = torch.zeros(1, 1, self.input_channels, self.input_length)

            with torch.no_grad():
                x = dummy_input
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)

            # After CNN: (batch, channels, height, width)
            # We want to preserve time dimension as sequence length
            # Shape becomes (batch, conv_channels, reduced_channels, reduced_time)
            # Flatten spatial dims: (batch, conv_channels * reduced_channels, reduced_time)
            self.conv_output_channels = x.size(1)
            self.conv_spatial_size = x.size(2)
            self.time_steps = x.size(3)
            self.conv_output_size = self.conv_output_channels * self.conv_spatial_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass

            Args:
                x: Input tensor (batch_size, input_channels, input_length)

            Returns:
                LSTM output (batch_size, sequence_length, final_units)
            """
            # Add channel dimension for CNN: (batch_size, 1, channels, time)
            x = x.unsqueeze(1)

            # Apply CNN layers
            for conv_layer in self.conv_layers:
                x = conv_layer(x)

            # Reshape for LSTM: (batch_size, time_steps, features)
            # From (batch, conv_channels, reduced_channels, reduced_time)
            # To (batch, reduced_time, conv_channels * reduced_channels)
            x = x.permute(0, 3, 1, 2)  # (batch, time, conv_channels, spatial)
            x = x.contiguous().view(x.size(0), x.size(1), -1)  # (batch, time, features)

            # Apply LSTM layers
            for lstm_layer in self.lstm_layers:
                x, _ = lstm_layer(x)

            return x


    class fNIRSFCN(nn.Module):
        """
        Fully Connected Network for fNIRS signal processing
        """
        
        def __init__(self, config: Optional[Dict] = None):
            super(fNIRSFCN, self).__init__()
            
            if config is None:
                config = MODEL_CONFIG['fnirs_stream']
            
            self.config = config
            self.input_channels = config['input_channels']
            self.input_length = config['input_length']
            self.dense_units = config['dense_units']
            self.dropout = config['dropout']
            
            # Flatten input size
            input_size = self.input_channels * self.input_length
            
            # Fully connected layers
            self.fc_layers = nn.ModuleList()
            in_units = input_size
            
            for units in self.dense_units:
                fc_block = nn.Sequential(
                    nn.Linear(in_units, units),
                    nn.BatchNorm1d(units),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
                self.fc_layers.append(fc_block)
                in_units = units
            
            self.output_units = in_units
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass
            
            Args:
                x: Input tensor (batch_size, input_channels, input_length)
                
            Returns:
                Output features (batch_size, output_units)
            """
            # Flatten input
            x = x.view(x.size(0), -1)  # (batch_size, input_channels * input_length)
            
            # Apply fully connected layers
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
            
            return x


    class MultiModalFusionModel(nn.Module):
        """
        Multi-modal fusion model with attention mechanism
        """
        
        def __init__(self, config: Optional[Dict] = None):
            super(MultiModalFusionModel, self).__init__()
            
            if config is None:
                config = MODEL_CONFIG
            
            self.eeg_config = config['eeg_stream']
            self.fnirs_config = config['fnirs_stream']
            self.fusion_config = config['fusion']
            self.output_classes = self.fusion_config['output_classes']
            self.fusion_method = self.fusion_config['fusion_method']
            
            # Create individual modality networks
            self.eeg_encoder = EEGCNNLSTM(self.eeg_config)
            self.fnirs_encoder = fNIRSFCN(self.fnirs_config)
            
            # Attention mechanism
            if self.fusion_method == 'attention':
                attention_units = self.fusion_config['attention_units']
                combined_features = self.eeg_encoder.final_units + self.fnirs_encoder.output_units
                
                self.attention = nn.Sequential(
                    nn.Linear(combined_features, attention_units),
                    nn.Tanh(),
                    nn.Linear(attention_units, 2),  # 2 modalities
                    nn.Softmax(dim=1)
                )
                
                fusion_input_size = combined_features
            elif self.fusion_method == 'concat':
                fusion_input_size = self.eeg_encoder.final_units + self.fnirs_encoder.output_units
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(fusion_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, self.output_classes)
            )
        
        def forward(self, eeg_input: torch.Tensor, fnirs_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass
            
            Args:
                eeg_input: EEG input (batch_size, eeg_channels, eeg_length)
                fnirs_input: fNIRS input (batch_size, fnirs_channels, fnirs_length)
                
            Returns:
                Tuple of (logits, attention_weights)
            """
            # Process each modality
            eeg_features = self.eeg_encoder(eeg_input)
            fnirs_features = self.fnirs_encoder(fnirs_input)
            
            # Get the final representations
            eeg_repr = eeg_features[:, -1, :]  # Last time step
            fnirs_repr = fnirs_features
            
            if self.fusion_method == 'attention':
                # Concatenate features for attention
                combined = torch.cat([eeg_repr, fnirs_repr], dim=1)
                
                # Compute attention weights
                attention_weights = self.attention(combined)
                
                # Apply attention weights
                eeg_weighted = eeg_repr * attention_weights[:, 0:1]
                fnirs_weighted = fnirs_repr * attention_weights[:, 1:2]
                
                fused_features = torch.cat([eeg_weighted, fnirs_weighted], dim=1)
            elif self.fusion_method == 'concat':
                fused_features = torch.cat([eeg_repr, fnirs_repr], dim=1)
                attention_weights = torch.zeros(eeg_repr.size(0), 2).to(DEVICE)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            # Classification
            logits = self.classifier(fused_features)
            
            return logits, attention_weights


    class BaselineEEGModel(nn.Module):
        """
        Baseline model using only EEG data
        """
        
        def __init__(self, input_channels: int, input_length: int, output_classes: int):
            super(BaselineEEGModel, self).__init__()
            
            # Create custom config with actual input size
            custom_config = {
                'input_channels': input_channels,
                'input_length': input_length,
                'conv_filters': [32, 64, 128],
                'conv_kernel': (3, 3),
                'lstm_units': [128, 64],
                'dropout': 0.3,
            }
            
            self.eeg_encoder = EEGCNNLSTM(config=custom_config)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(self.eeg_encoder.final_units, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, output_classes)
            )
        
        def forward(self, x_eeg: torch.Tensor, x_fnirs: torch.Tensor = None) -> torch.Tensor:
            """
            Forward pass
            
            Args:
                x_eeg: EEG input (batch_size, channels, time)
                x_fnirs: fNIRS input (ignored for baseline)
                
            Returns:
                Classification logits
            """
            features = self.eeg_encoder(x_eeg)
            # Take last time step
            final_features = features[:, -1, :]
            logits = self.classifier(final_features)
            return logits



    class BaselinefNIRSModel(nn.Module):
        """
        Baseline model using only fNIRS data
        """
        
        def __init__(self, input_channels: int, input_length: int, output_classes: int):
            super(BaselinefNIRSModel, self).__init__()
            
            # Create custom config with actual input size
            custom_config = {
                'input_channels': input_channels,
                'input_length': input_length,
                'dense_units': [128, 64],
                'dropout': 0.2,
            }
            
            self.fnirs_encoder = fNIRSFCN(config=custom_config)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(self.fnirs_encoder.output_units, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, output_classes)
            )
        
        def forward(self, x_eeg: torch.Tensor, x_fnirs: torch.Tensor) -> torch.Tensor:
            """
            Forward pass
            
            Args:
                x_eeg: EEG input (ignored for baseline)
                x_fnirs: fNIRS input (batch_size, channels, time)
                
            Returns:
                Classification logits
            """
            features = self.fnirs_encoder(x_fnirs)
            logits = self.classifier(features)
            return logits
else:
    # Create placeholder classes when torch is not available
    EEGCNNLSTM = None
    fNIRSFCN = None
    MultiModalFusionModel = None
    BaselineEEGModel = None
    BaselinefNIRSModel = None


if MODULE_AVAILABLE:
    def create_model(model_type: str, **kwargs):
        """
        Factory function for creating models
        
        Args:
            model_type: Type of model ('multimodal', 'eeg_baseline', 'fnirs_baseline')
            **kwargs: Additional model parameters
            
        Returns:
            PyTorch model instance
        """
        if model_type.lower() == 'multimodal':
            return MultiModalFusionModel(**kwargs)
        elif model_type.lower() == 'eeg_baseline':
            return BaselineEEGModel(**kwargs)
        elif model_type.lower() == 'fnirs_baseline':
            return BaselinefNIRSModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    def count_parameters(model) -> int:
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def model_summary(model, input_size_eeg: Tuple, input_size_fnirs: Tuple = None):
        """Print model summary"""
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {count_parameters(model):,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            if input_size_fnirs and hasattr(model, 'eeg_encoder'):
                eeg_dummy = torch.randn(input_size_eeg)
                fnirs_dummy = torch.randn(input_size_fnirs)
                output, attention = model(eeg_dummy, fnirs_dummy)
                print(f"Output shape: {output.shape}")
                print(f"Attention weights shape: {attention.shape}")
            else:
                dummy_input = torch.randn(input_size_eeg)
                output = model(dummy_input)
                print(f"Output shape: {output.shape}")
else:
    def create_model(model_type: str, **kwargs):
        """Placeholder function when torch is not available"""
        raise ImportError("PyTorch is not available. Cannot create models.")
    
    def count_parameters(model) -> int:
        """Placeholder function when torch is not available"""
        raise ImportError("PyTorch is not available. Cannot count parameters.")
    
    def model_summary(model, input_size_eeg: Tuple, input_size_fnirs: Tuple = None):
        """Placeholder function when torch is not available"""
        raise ImportError("PyTorch is not available. Cannot summarize model.")


def check_module_availability():
    """Check if all required dependencies are available"""
    return MODULE_AVAILABLE


# Wrap classes to check availability before instantiation
class SafeMultiModalFusionModel:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (torch) not available for models module")
        self.model = MultiModalFusionModel(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


class SafeBaselineEEGModel:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (torch) not available for models module")
        self.model = BaselineEEGModel(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


class SafeBaselinefNIRSModel:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (torch) not available for models module")
        self.model = BaselinefNIRSModel(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


if __name__ == "__main__":
    # Test model creation
    if check_module_availability():
        device = torch.device(DEVICE)
        
        # Create multimodal model
        multimodal_model = MultiModalFusionModel()
        multimodal_model.to(device)
        model_summary(
            multimodal_model,
            input_size_eeg=(2, 64, 1000),
            input_size_fnirs=(2, 10, 200)
        )
        
        # Create baseline models
        eeg_baseline = BaselineEEGModel(64, 1000, 4)
        eeg_baseline.to(device)
        model_summary(eeg_baseline, input_size_eeg=(2, 64, 1000))
        
        fnirs_baseline = BaselinefNIRSModel(10, 200, 4)
        fnirs_baseline.to(device)
        model_summary(fnirs_baseline, input_size_eeg=(2, 10, 200))
    else:
        print("PyTorch not available, skipping model tests")