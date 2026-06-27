"""
phase5b_train_hmm.py — Hidden Markov Model Regime Classifier
============================================================
Trains an unsupervised Gaussian HMM on historical returns and volatility
to automatically classify the market into Bull, Bear, or Crab regimes.
"""

import os
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def train_hmm():
    print("\n" + "=" * 70)
    print("  PHASE 5B — TRAINING HIDDEN MARKOV MODEL (HMM)")
    print("=" * 70)

    # 1. Load data (we use BTC as the macro proxy for all crypto)
    data_path = os.path.join(config.DATA_DIR, "BTC_USDT_train_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  [Data] Loaded {len(df)} rows from BTC train set.")

    # 2. Extract features: Daily log return and Daily historical volatility
    if "1d_log_return" not in df.columns or "1d_hvol_20" not in df.columns:
        raise ValueError("Required features '1d_log_return' and '1d_hvol_20' are missing.")

    # Use a rolling window of 24 hours (1 day) to smooth out the features slightly
    # This prevents the HMM from rapidly oscillating every hour.
    df['smooth_return'] = df['1d_log_return'].rolling(24).mean()
    df['smooth_vol'] = df['1d_hvol_20'].rolling(24).mean()
    
    hmm_data = df[['smooth_return', 'smooth_vol']].dropna()
    X = hmm_data.values

    print(f"  [HMM] Training 3-component Gaussian HMM on {len(X)} samples...")
    
    # 3. Fit the HMM
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    hmm_model.fit(X)

    # 4. Predict hidden states on the training data
    hidden_states = hmm_model.predict(X)

    # 5. Map hidden states to Bull, Bear, Crab
    # We calculate the mean return for each state to identify them.
    state_returns = []
    for i in range(3):
        mask = (hidden_states == i)
        state_mean_ret = X[mask, 0].mean()
        state_returns.append((i, state_mean_ret))

    # Sort states by mean return: Lowest -> Highest
    state_returns.sort(key=lambda x: x[1])
    
    # Lowest return = Bear (0)
    # Middle return = Crab (1)
    # Highest return = Bull (2)
    
    bear_state = state_returns[0][0]
    crab_state = state_returns[1][0]
    bull_state = state_returns[2][0]

    state_map = {
        bear_state: "bear",
        crab_state: "crab",
        bull_state: "bull"
    }

    print("\n  [HMM] Regime Mapping Identified:")
    print(f"    State {bull_state}: BULL (Mean smoothed 1d return: {state_returns[2][1]*100:.4f}%)")
    print(f"    State {crab_state}: CRAB (Mean smoothed 1d return: {state_returns[1][1]*100:.4f}%)")
    print(f"    State {bear_state}: BEAR (Mean smoothed 1d return: {state_returns[0][1]*100:.4f}%)")

    # 6. Save the model and mapping
    model_save_path = os.path.join(config.MODELS_DIR, "hmm_regime_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": hmm_model, "state_map": state_map}, f)
    print(f"\n  [Save] HMM model and mapping saved to {model_save_path}")

    # 7. Visualization
    print("  [Plot] Generating HMM regime chart...")
    plot_df = df.loc[hmm_data.index].copy()
    plot_df['regime_id'] = hidden_states
    
    # Map raw integer to standard 0=Bear, 1=Crab, 2=Bull for coloring
    standard_map = {bear_state: 0, crab_state: 1, bull_state: 2}
    plot_df['color_id'] = plot_df['regime_id'].map(standard_map)

    plt.figure(figsize=(15, 7))
    plt.plot(plot_df.index, plot_df['Close'], color='black', lw=0.5, alpha=0.5)
    
    # Scatter plot over the line to show regimes
    cmap = ListedColormap(['#e74c3c', '#95a5a6', '#2ecc71']) # Red, Grey, Green
    scatter = plt.scatter(plot_df.index, plot_df['Close'], c=plot_df['color_id'], cmap=cmap, s=2)
    
    plt.title("BTC/USDT Price Colored by HMM Market Regimes")
    plt.ylabel("Price (USD)")
    
    import matplotlib.lines as mlines
    bear_line = mlines.Line2D([], [], color='#e74c3c', marker='o', linestyle='None', markersize=8, label='Bear')
    crab_line = mlines.Line2D([], [], color='#95a5a6', marker='o', linestyle='None', markersize=8, label='Crab')
    bull_line = mlines.Line2D([], [], color='#2ecc71', marker='o', linestyle='None', markersize=8, label='Bull')
    plt.legend(handles=[bear_line, crab_line, bull_line])
    
    plot_path = os.path.join(config.RESULTS_DIR, "hmm_regimes.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  [Save] Regime plot saved to {plot_path}")

if __name__ == "__main__":
    train_hmm()
