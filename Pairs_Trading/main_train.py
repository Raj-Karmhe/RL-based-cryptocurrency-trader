import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# ==============================================================================
# MAIN TRAINING SCRIPT — PAIRS TRADING RL AGENT
# Orchestrates the full pipeline: data → validation → features → training
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_processing import process_and_split_data
from pair_selection import run_full_validation
from download_data import download_and_align_pair
from environment import PairsTradingEnv
from model import PairsLSTMFeatureExtractor
from utils import PairsTradingCallback, run_agent


def make_env(df_raw, df_scaled, feature_columns):
    """Factory function for DummyVecEnv."""
    def _init():
        return PairsTradingEnv(df=df_raw, df_scaled=df_scaled, feature_columns=feature_columns)
    return _init


def train_and_validate():
    # ── Step 1: Data Download ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: DATA DOWNLOAD")
    print("=" * 70)
    df_a, df_b, df_merged = download_and_align_pair()

    # ── Step 2: Data Processing ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: DATA PROCESSING & FEATURE ENGINEERING")
    print("=" * 70)
    # Pass already-downloaded data to avoid a redundant second API call
    train_s, val_s, test_s, train_r, val_r, test_r = process_and_split_data(
        df_a=df_a, df_b=df_b, df_merged=df_merged
    )

    # ── Step 3: Cointegration Validation (On Train Set Only) ──────────────
    print("\n" + "=" * 70)
    print("  STEP 3: COINTEGRATION VALIDATION")
    print("=" * 70)
    validation = run_full_validation(train_r['Close_A'], train_r['Close_B'])

    if not validation['overall_pass']:
        print("\n  [!] WARNING: Cointegration tests did not all pass.")
        print("  The strategy may have reduced effectiveness.")
        print("  Proceeding with training anyway...\n")

    # ── Step 4: Build Training Environment ────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4: CLSTM-PPO TRAINING")
    print("=" * 70)
    # Spin up 8 parallel environments to feed the GPU 8x faster
    # Using SubprocVecEnv for actual parallel execution across CPU cores
    train_env = SubprocVecEnv([make_env(train_r, train_s, config.FEATURE_COLUMNS) for _ in range(8)])

    # ── Step 5: Create or Load Model ──────────────────────────────────────
    if os.path.exists(config.MODEL_PATH + ".zip") and not config.FORCE_RETRAIN:
        print(f"INFO: Loading pre-trained model from {config.MODEL_PATH}.zip...")
        model = PPO.load(config.MODEL_PATH, env=train_env)
        callback = PairsTradingCallback(verbose=0)
        print("SUCCESS: Model loaded!")
    else:
        print("INFO: Training CLSTM-PPO Pairs Trading agent from scratch...")
        print(f"   Timesteps:     {config.TOTAL_TIMESTEPS:,}")
        print(f"   Architecture:  {'Dual-Stream' if config.DUAL_STREAM else 'Single-Stream'}")
        print(f"   Features:      {config.N_FEATURES}")
        print(f"   Time Window:   {config.TIME_WINDOW}")
        print(f"   LSTM Hidden:   {config.LSTM_HIDDEN_SIZE}")

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            vf_coef=config.VF_COEF,
            max_grad_norm=config.MAX_GRAD_NORM,
            verbose=1,
            seed=42,
            policy_kwargs=dict(
                features_extractor_class=PairsLSTMFeatureExtractor,
                features_extractor_kwargs=dict(
                    time_window=config.TIME_WINDOW,
                    n_market_features=config.N_FEATURES,
                    hidden_size=config.LSTM_HIDDEN_SIZE,
                    out_features=config.LSTM_OUT_FEATURES,
                    n_lstm_layers=config.N_LSTM_LAYERS,
                    dual_stream=config.DUAL_STREAM,
                    n_spread_features=config.N_SPREAD_FEATURES,
                    n_asset_a_features=config.N_ASSET_A_FEATURES,
                    n_asset_b_features=config.N_ASSET_B_FEATURES,
                    n_cross_features=config.N_CROSS_FEATURES,
                ),
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                optimizer_kwargs=dict(eps=1e-8, betas=(0.9, 0.999)),
            ),
        )

        callback = PairsTradingCallback(verbose=1)
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
        )

        model.save(config.MODEL_PATH)
        print(f"\nSUCCESS: Training complete! Model saved to {config.MODEL_PATH}.zip")

    # ── Step 6: Validation Evaluation ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6: VALIDATION EVALUATION")
    print("INFO: Running agent on validation data...")
    val_results = run_agent(model, val_s, val_r, config.FEATURE_COLUMNS)
    vm = val_results['metrics']

    # Baseline comparison
    from utils import compute_spread_baseline, compute_metrics
    baseline_pv = compute_spread_baseline(val_r)
    baseline_metrics = compute_metrics(baseline_pv, config.TIMEFRAME)

    print(f"\n{'-'*75}")
    print(f"  {'Metric':<35} {'CLSTM-PPO':>15} {'CDF Baseline':>18}")
    print(f"{'-'*75}")
    print(f"  {'Total Return (%)':<35} {vm['total_return']*100:>15.2f} "
          f"{baseline_metrics['total_return']*100:>18.2f}")
    print(f"  {'Max Drawdown (%)':<35} {vm['max_drawdown']*100:>15.2f} "
          f"{baseline_metrics['max_drawdown']*100:>18.2f}")
    print(f"  {'Sharpe Ratio':<35} {vm['sharpe_ratio']:>15.4f} "
          f"{baseline_metrics['sharpe_ratio']:>18.4f}")
    print(f"  {'Win Rate (%)':<35} {vm['win_rate']*100:>15.2f}")
    print(f"  {'Final Portfolio ($)':<35} {vm['final_portfolio']:>15,.2f} "
          f"{baseline_metrics['final_portfolio']:>18,.2f}")
    print(f"  {'Total Trades':<35} {val_results['total_trades']:>15}")
    print(f"  {'Stop-Loss Exits':<35} {val_results['sl_triggers']:>15}")
    print(f"  {'Take-Profit Exits':<35} {val_results['tp_triggers']:>15}")
    print(f"  {'Timeout Exits':<35} {val_results['timeout_exits']:>15}")
    print(f"{'-'*75}")

    # ── Step 7: Save Validation Chart ─────────────────────────────────────
    print("\nINFO: Generating validation charts...")
    _save_quick_chart(val_results, baseline_pv, val_r, "validation")

    print(f"\nSUCCESS: Validation complete! Run main_test.py for final backtest.")


def _save_quick_chart(results, baseline_pv, df_raw, label):
    """Generate a quick portfolio comparison chart."""
    dates = pd.DatetimeIndex(results['dates'])
    pv = results['portfolio_values']
    pos = results['positions']
    # Don't let baseline length limit the agent chart
    min_len = min(len(dates), len(pv), len(pos))
    baseline_len = min(len(baseline_pv), min_len)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={'height_ratios': [3, 1]})

    # Portfolio comparison
    axes[0].plot(dates[:min_len], pv[:min_len],
                 color='#2196F3', linewidth=1.5, label='CLSTM-PPO Agent')
    axes[0].plot(dates[:baseline_len], baseline_pv[:baseline_len],
                 color='#FF9800', linewidth=1.0, linestyle='--',
                 label='CDF Baseline')
    axes[0].axhline(y=config.INITIAL_BALANCE, color='gray',
                     linestyle=':', alpha=0.5, label='Starting Capital')
    axes[0].set_title(f'Pairs Trading — {label.title()} Performance')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Position plot
    colors = ['#F44336' if p < -0.05 else '#4CAF50' if p > 0.05 else '#9E9E9E'
              for p in pos[:min_len]]
    axes[1].bar(dates[:min_len], pos[:min_len], color=colors, alpha=0.7, width=0.02)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Spread Position (Red=Short, Green=Long, Gray=Flat)')
    axes[1].set_ylabel('Position')
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f'{label}_portfolio_chart.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


if __name__ == '__main__':
    train_and_validate()
