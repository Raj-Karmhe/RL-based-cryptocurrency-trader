"""
phase3_model.py - Custom TCN Feature Extractor for PPO Trading Policy

This script defines the neural network architecture for our Reinforcement Learning policy.
It subclasses Stable-Baselines3's BaseFeaturesExtractor to extract temporal features
from multi-timeframe market windows using a PyTorch Temporal Convolutional Network (TCN),
combines them with portfolio status variables, and passes them through an MLP bottleneck.
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import os
import sys

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

class Chomp1d(nn.Module):
    """
    Slices off the rightmost padding elements to enforce temporal causality.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Bug #16 fix: guard against chomp_size == 0 (x[:, :, :-0] returns empty tensor)
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalResidualBlock(nn.Module):
    """
    A single residual block containing two dilated causal 1D convolutional layers,
    weight normalization, dropout, and a residual skip connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MarketRecurrentExtractor(BaseFeaturesExtractor):
    """
    Custom PyTorch neural network that extracts temporal market features.
    Processes sequential price windows through Dilated Causal 1D Convolutions (TCN)
    and blends the outcome with portfolio-state features.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        sequence_length: int = config.SEQ_LEN,
        num_features: int = None,
        gru_hidden: int = config.LSTM_HIDDEN_SIZE,  # Kept signature param name to prevent train.py breakdown
        mlp_hidden: int = config.MLP_HIDDEN_SIZE,
        dropout_prob: float = config.DROPOUT_RATE
    ):
        super().__init__(observation_space, features_dim=mlp_hidden)
        
        self.seq_len = sequence_length
        self.gru_hidden = gru_hidden
        self.portfolio_dim = 2
        
        # Calculate market feature count from observation vector
        total_dim = observation_space.shape[0]
        market_total_dim = total_dim - self.portfolio_dim
        
        if num_features is not None:
            self.n_features = num_features
        else:
            self.n_features = market_total_dim // self.seq_len
            
        print(f"[Model Network] Initializing TCN-based MarketRecurrentExtractor:")
        print(f"  Sequence Length    : {self.seq_len}")
        print(f"  Market Feature Dim : {self.n_features}")
        print(f"  TCN Hidden Size    : {self.gru_hidden}")
        print(f"  MLP Bottleneck Size: {mlp_hidden}")
        
        # Build TCN Layers
        kernel_size = 3
        num_channels = [gru_hidden] * config.TCN_N_LAYERS
        
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8
            in_channels = self.n_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalResidualBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, padding=padding,
                dropout=dropout_prob
            ))
            
        self.tcn = nn.Sequential(*layers)
        
        # Combined dimension = TCN outputs + portfolio state (current allocation, unrealized PnL)
        input_mlp_dim = num_channels[-1] + self.portfolio_dim
        self.mlp_head = nn.Sequential(
            nn.Linear(input_mlp_dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Tanh()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        n_feat = self.n_features
        s_len = self.seq_len
        
        # Split features window and portfolio state variables
        market_flat = observations[:, : s_len * n_feat]
        portfolio_state = observations[:, s_len * n_feat :]
        
        # Reshape to (batch_size, channels, seq_len) to match Conv1d requirements
        market_seq = market_flat.view(batch_size, s_len, n_feat).transpose(1, 2)
        
        # Run TCN
        tcn_out = self.tcn(market_seq)  # Shape: (batch_size, gru_hidden, seq_len)
        
        # Extract features at the final time step
        last_tcn_hidden = tcn_out[:, :, -1]  # Shape: (batch_size, gru_hidden)
        
        # Concatenate final hidden state with current portfolio status
        combined_features = torch.cat([last_tcn_hidden, portfolio_state], dim=1)
        
        # Pass through the MLP bottleneck
        features_representation = self.mlp_head(combined_features)
        
        return features_representation

if __name__ == "__main__":
    # Model Standalone Test
    seq_length_test = config.SEQ_LEN
    n_feats_test = 8
    portfolio_state_test = 2
    obs_dimension = seq_length_test * n_feats_test + portfolio_state_test
    
    test_obs_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dimension,),
        dtype=np.float32
    )
    
    network = MarketRecurrentExtractor(
        observation_space=test_obs_space,
        sequence_length=seq_length_test,
        num_features=n_feats_test
    )
    
    print("\nNetwork Architecture:\n", network)
    
    # Run mock batch forward pass
    dummy_input = torch.randn(5, obs_dimension)
    with torch.no_grad():
        output = network(dummy_input)
        
    print(f"\nDummy Input Batch Shape : {dummy_input.shape}")
    print(f"Extractor Output Shape  : {output.shape} (Expected: [5, {config.MLP_HIDDEN_SIZE}])")
    
    assert output.shape == (5, config.MLP_HIDDEN_SIZE), "Output shape validation failed!"
    print("\nCustom Model Feature Extractor standalone test PASSED!")
