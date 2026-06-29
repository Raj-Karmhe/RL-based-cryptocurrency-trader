"""
phase3_model.py — Cascaded LSTM (CLSTM) Feature Extractor for PPO
==================================================================
Phase 3, Component 2 — the Neural Network Architecture.

CLSTM ARCHITECTURE (based on paper 2212.02721v2)
-------------------------------------------------
A standard stacked LSTM feeds the OUTPUT of layer L to the INPUT of layer
L+1.  A Cascaded LSTM (CLSTM) instead concatenates:
    - the hidden state from layer L
    - the original raw input at each timestep
before passing to layer L+1.

This means deeper layers still have direct access to the original features,
preventing the vanishing gradient problem and allowing each layer to learn
complementary temporal abstractions at different scales.

Architecture:
    Input: (batch, SEQ_LEN, N_FEATURES)
        ↓
    LSTM Layer 1    → hidden_state_1   (batch, hidden_size)
        ↓ concatenate with original input
    LSTM Layer 2    → hidden_state_2   (batch, hidden_size)
        ↓ concatenate with portfolio state
    MLP (3 layers)  → feature_vector   (batch, mlp_hidden)
        ↓
    PPO Actor / Critic networks (inside SB3)

The feature_vector is the "compressed market state" that the PPO actor uses
to decide the action and the critic uses to estimate the value function.

INTEGRATION WITH STABLE BASELINES 3
-------------------------------------
We extend SB3's `BaseFeaturesExtractor` so that our CLSTM slots directly
into any SB3 algorithm (PPO in our case) without modifying SB3 internals.
The output `features_dim` is automatically passed to the downstream policy
and value network heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import sys
import os

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ──────────────────────────────────────────────────────────────────────────────
# CASCADED LSTM CELL BLOCK
# ──────────────────────────────────────────────────────────────────────────────

class CascadedLSTMBlock(nn.Module):
    """
    A two-layer Cascaded LSTM (CLSTM) block.

    Unlike a standard stacked LSTM where:
        input_to_L2 = output_of_L1

    A CLSTM concatenates the raw input with the L1 output:
        input_to_L2 = concat(output_of_L1, raw_input)

    This residual information flow allows the second layer to refine the
    first layer's temporal abstraction while retaining direct access to the
    raw signal — a key design choice from the paper that improves gradient
    flow and helps the network learn both short and long-term dependencies.

    Parameters
    ----------
    input_size  : Number of input features at each timestep
    hidden_size : LSTM hidden dimension (same for both layers)
    dropout     : Dropout applied between layers
    """

    def __init__(self,
                 input_size:  int,
                 hidden_size: int,
                 dropout:     float = config.DROPOUT_RATE):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Layer 1: processes the raw input sequence
        self.lstm1 = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,    # (batch, seq, features)
            dropout     = 0.0,
        )

        # Layer 2: processes concatenated (L1 output + raw input)
        # Input dimension is now: hidden_size + input_size  (the CASCADE)
        self.lstm2 = nn.LSTM(
            input_size  = hidden_size + input_size,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
            dropout     = 0.0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        last_hidden : torch.Tensor of shape (batch, hidden_size)
            The final hidden state of LSTM Layer 2 after processing the full sequence.
        """
        # ── LSTM Layer 1 ──────────────────────────────────────────────────
        # out1: (batch, seq_len, hidden_size) — all timestep outputs
        out1, _ = self.lstm1(x)
        out1    = self.dropout(out1)

        # ── CLSTM CASCADE: concatenate L1 output with raw input ──────────
        # This is the defining characteristic of the CLSTM vs standard LSTM.
        # At each timestep t, the input to L2 is:
        #   [lstm1_output_t | raw_input_t]
        cascaded = torch.cat([out1, x], dim=-1)   # (batch, seq_len, hidden+input)

        # ── LSTM Layer 2 ──────────────────────────────────────────────────
        # We only need the final hidden state (summary of the full sequence)
        _, (h2, _) = self.lstm2(cascaded)

        # h2 shape: (num_layers=1, batch, hidden_size) → squeeze to (batch, hidden)
        last_hidden = h2.squeeze(0)
        return last_hidden


# ──────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTOR (integrates with SB3)
# ──────────────────────────────────────────────────────────────────────────────

class CLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Cascaded LSTM feature extractor for Stable Baselines 3.

    This class is passed to PPO via `policy_kwargs["features_extractor_class"]`.
    SB3 calls `forward()` on this class before passing the result to the
    actor and critic MLP heads.

    Observation Layout
    ------------------
    The observation vector from the environment is a flat 1-D array:
        [ market_window_features (SEQ_LEN × N_FEATURES)
          | portfolio_state (position, unrealized_pnl) ]
    This class reshapes the market window, runs it through CLSTM, then
    concatenates the portfolio state before the final MLP.

    Parameters
    ----------
    observation_space  : The Gymnasium observation space (Box)
    seq_len            : Rolling window length (SEQ_LEN)
    n_market_features  : Number of features per timestep (N_FEATURES)
    hidden_size        : LSTM hidden dimension
    mlp_hidden         : Output MLP hidden size (= features_dim for PPO)
    dropout            : LSTM dropout rate
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        seq_len:            int   = config.SEQ_LEN,
        n_market_features:  int   = None,  # inferred if None
        hidden_size:        int   = config.LSTM_HIDDEN_SIZE,
        mlp_hidden:         int   = config.MLP_HIDDEN_SIZE,
        dropout:            float = config.DROPOUT_RATE,
    ):
        # features_dim tells PPO how many outputs this extractor produces
        super().__init__(observation_space, features_dim=mlp_hidden)

        self.seq_len           = seq_len
        self.hidden_size       = hidden_size

        # Infer n_market_features from observation space if not provided
        total_obs = observation_space.shape[0]
        # The last 2 elements are portfolio state (position, unrealized_pnl)
        self.n_portfolio = 2
        market_flat      = total_obs - self.n_portfolio
        self.n_market_features = (n_market_features
                                  if n_market_features is not None
                                  else market_flat // seq_len)

        # ── CLSTM Block ───────────────────────────────────────────────────
        self.clstm = CascadedLSTMBlock(
            input_size  = self.n_market_features,
            hidden_size = hidden_size,
            dropout     = dropout,
        )

        # ── Output MLP ────────────────────────────────────────────────────
        # Input: LSTM hidden output + portfolio state (2 dims)
        mlp_input_dim = hidden_size + self.n_portfolio
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — called by SB3 during both training and inference.

        Parameters
        ----------
        observations : (batch_size, obs_dim) flat tensor from the environment

        Returns
        -------
        features : (batch_size, mlp_hidden) — compressed market + portfolio state
        """
        batch = observations.shape[0]
        n_mf  = self.n_market_features
        sl    = self.seq_len

        # ── 1. Split observation into market window and portfolio state ──
        market_flat  = observations[:, : sl * n_mf]       # (batch, sl*n_mf)
        portfolio_st = observations[:, sl * n_mf :]        # (batch, 2)

        # ── 2. Reshape flat market vector → 3-D sequence ─────────────────
        market_seq = market_flat.view(batch, sl, n_mf)     # (batch, sl, n_mf)

        # ── 3. Run through Cascaded LSTM ─────────────────────────────────
        lstm_out = self.clstm(market_seq)                  # (batch, hidden_size)

        # ── 4. Concatenate portfolio state ────────────────────────────────
        combined = torch.cat([lstm_out, portfolio_st], dim=1)  # (batch, hidden+2)

        # ── 5. MLP to produce the final feature vector ────────────────────
        features = self.mlp(combined)                      # (batch, mlp_hidden)

        return features


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a batch of 4 observations
    SEQ_LEN   = config.SEQ_LEN          # e.g. 24
    N_FEAT    = 8                        # example: 8 golden features
    N_PORTF   = 2
    OBS_DIM   = SEQ_LEN * N_FEAT + N_PORTF

    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(OBS_DIM,), dtype=np.float32
    )

    extractor = CLSTMFeatureExtractor(
        observation_space = obs_space,
        seq_len           = SEQ_LEN,
        n_market_features = N_FEAT,
    )
    print(extractor)

    dummy_obs = torch.randn(4, OBS_DIM)
    with torch.no_grad():
        out = extractor(dummy_obs)

    print(f"\nInput shape : {dummy_obs.shape}")
    print(f"Output shape: {out.shape}   (should be [4, {config.MLP_HIDDEN_SIZE}])")
    assert out.shape == (4, config.MLP_HIDDEN_SIZE), "Shape mismatch!"

    print("\nCLSTM Feature Extractor standalone test PASSED ✓")
