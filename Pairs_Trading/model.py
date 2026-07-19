import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import sys

# ==============================================================================
# CLSTM FEATURE EXTRACTOR FOR PAIRS TRADING
# Dual-stream architecture: separate LSTMs for spread, asset A, and asset B
# features, concatenated before the linear layers.
# Falls back to single-stream if DUAL_STREAM = False.
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class PairsLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    CLSTM Feature Extractor for Pairs Trading.

    Dual-Stream Architecture:
        Stream 1 (Spread):  [Spread features × TIME_WINDOW] → LSTM_spread
        Stream 2 (Asset A): [Asset A features × TIME_WINDOW] → LSTM_a
        Stream 3 (Asset B): [Asset B features × TIME_WINDOW] → LSTM_b
        Cross features are concatenated directly after LSTM processing.

        All hidden states → Concat → Linear×3 → Feature Vector

    Single-Stream Architecture (fallback):
        [All features × TIME_WINDOW] → LSTM → Linear×3 → Feature Vector
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        time_window: int = config.TIME_WINDOW,
        n_market_features: int = config.N_FEATURES,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        out_features: int = config.LSTM_OUT_FEATURES,
        n_lstm_layers: int = config.N_LSTM_LAYERS,
        dual_stream: bool = config.DUAL_STREAM,
        # Feature group sizes for dual-stream
        n_spread_features: int = config.N_SPREAD_FEATURES,
        n_asset_a_features: int = config.N_ASSET_A_FEATURES,
        n_asset_b_features: int = config.N_ASSET_B_FEATURES,
        n_cross_features: int = config.N_CROSS_FEATURES,
    ):
        super().__init__(observation_space, features_dim=out_features)

        self.time_window = time_window
        self.n_market_features = n_market_features
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.dual_stream = dual_stream

        # Feature group sizes
        self.n_spread_features = n_spread_features
        self.n_asset_a_features = n_asset_a_features
        self.n_asset_b_features = n_asset_b_features
        self.n_cross_features = n_cross_features

        # Portfolio state dimensions
        portfolio_dim = observation_space.shape[0] - (time_window * n_market_features)
        self.portfolio_dim = max(0, portfolio_dim)

        if dual_stream:
            # ── Dual-Stream: Separate LSTMs per feature group ─────────────
            # Each LSTM processes its own feature subset through time

            # Hidden sizes proportional to feature count
            # (but with a minimum of 64 to ensure sufficient capacity)
            spread_hidden = max(64, hidden_size // 3)
            asset_hidden = max(64, hidden_size // 4)

            self.lstm_spread = nn.LSTM(
                input_size=n_spread_features,
                hidden_size=spread_hidden,
                num_layers=n_lstm_layers,
                batch_first=True,
            )

            self.lstm_a = nn.LSTM(
                input_size=n_asset_a_features,
                hidden_size=asset_hidden,
                num_layers=n_lstm_layers,
                batch_first=True,
            )

            self.lstm_b = nn.LSTM(
                input_size=n_asset_b_features,
                hidden_size=asset_hidden,
                num_layers=n_lstm_layers,
                batch_first=True,
            )

            # Cross features don't have temporal structure that differs from spread,
            # so we just concatenate them from the last timestep
            concat_dim = (spread_hidden + asset_hidden * 2 +
                          n_cross_features + self.portfolio_dim)

        else:
            # ── Single-Stream: One LSTM for everything ────────────────────
            self.lstm = nn.LSTM(
                input_size=n_market_features,
                hidden_size=hidden_size,
                num_layers=n_lstm_layers,
                batch_first=True,
            )
            concat_dim = hidden_size + self.portfolio_dim

        # ── Three Linear Layers with Tanh activation ──────────────────────
        self.linear_layers = nn.Sequential(
            nn.Linear(concat_dim, out_features),
            nn.Tanh(),
            nn.Linear(out_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, out_features),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass: process observation through LSTM(s) and linear layers."""
        batch_size = observations.shape[0]
        seq_len = self.time_window
        n_mf = self.n_market_features

        # Split observation into market sequence and portfolio state
        market_flat = observations[:, :seq_len * n_mf]
        portfolio_st = observations[:, seq_len * n_mf:]

        if self.dual_stream:
            # Reshape to (batch, seq_len, n_features)
            market_seq = market_flat.view(batch_size, seq_len, n_mf)

            # Split features by group
            # Feature order matches config: SPREAD + ASSET_A + ASSET_B + CROSS
            s_end = self.n_spread_features
            a_end = s_end + self.n_asset_a_features
            b_end = a_end + self.n_asset_b_features

            spread_seq = market_seq[:, :, :s_end]
            asset_a_seq = market_seq[:, :, s_end:a_end]
            asset_b_seq = market_seq[:, :, a_end:b_end]
            cross_last = market_seq[:, -1, b_end:]  # Only last timestep

            # Process each stream through its LSTM
            _, (h_spread, _) = self.lstm_spread(spread_seq)
            _, (h_a, _) = self.lstm_a(asset_a_seq)
            _, (h_b, _) = self.lstm_b(asset_b_seq)

            # Take final hidden state from each
            h_spread = h_spread[-1]  # (batch, spread_hidden)
            h_a = h_a[-1]           # (batch, asset_hidden)
            h_b = h_b[-1]           # (batch, asset_hidden)

            # Concatenate all streams + cross features + portfolio state
            parts = [h_spread, h_a, h_b, cross_last]
            if self.portfolio_dim > 0:
                parts.append(portfolio_st)
            combined = torch.cat(parts, dim=1)

        else:
            # Single stream
            market_seq = market_flat.view(batch_size, seq_len, n_mf)
            _, (h_n, _) = self.lstm(market_seq)
            last_hidden = h_n[-1]

            if self.portfolio_dim > 0:
                combined = torch.cat([last_hidden, portfolio_st], dim=1)
            else:
                combined = last_hidden

        # Pass through linear layers
        features = self.linear_layers(combined)
        return features
