import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from stable_baselines3 import PPO

# ==============================================================================
# MAIN TESTING SCRIPT — PAIRS TRADING RL AGENT
# Final out-of-sample evaluation with rich multi-panel visualization.
# This data has NEVER been seen during training or validation.
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_processing import process_and_split_data
from utils import run_agent, compute_metrics, compute_spread_baseline


def run_test():
    print("\n" + "=" * 70)
    print("  FINAL OUT-OF-SAMPLE TEST — PAIRS TRADING")
    print(f"  Pair: {config.ASSET_A_LABEL} / {config.ASSET_B_LABEL}")
    print("=" * 70)

    # ── 1. Load Model ─────────────────────────────────────────────────────
    if not os.path.exists(config.MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}.zip. "
            f"Run main_train.py first."
        )

    print(f"INFO: Loading model from {config.MODEL_PATH}.zip...")
    model = PPO.load(config.MODEL_PATH)
    print("SUCCESS: Model loaded!")

    # ── 2. Get Test Data ──────────────────────────────────────────────────
    _, _, test_s, _, _, test_r = process_and_split_data()

    # ── 3. Run Agent ──────────────────────────────────────────────────────
    print("\nINFO: Running CLSTM-PPO agent on UNSEEN test data...")
    test_results = run_agent(model, test_s, test_r, config.FEATURE_COLUMNS)
    tm = test_results['metrics']

    # ── 4. Baseline Comparison ────────────────────────────────────────────
    baseline_pv = compute_spread_baseline(test_r)
    baseline_metrics = compute_metrics(baseline_pv, config.TIMEFRAME)

    # ── 5. Print Results ──────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  {'Metric':<35} {'CLSTM-PPO':>15} {'CDF Baseline':>18}")
    print(f"{'─'*80}")
    print(f"  {'Total Return (%)':<35} {tm['total_return']*100:>15.2f} "
          f"{baseline_metrics['total_return']*100:>18.2f}")
    print(f"  {'Max Drawdown (%)':<35} {tm['max_drawdown']*100:>15.2f} "
          f"{baseline_metrics['max_drawdown']*100:>18.2f}")
    print(f"  {'Sharpe Ratio':<35} {tm['sharpe_ratio']:>15.4f} "
          f"{baseline_metrics['sharpe_ratio']:>18.4f}")
    print(f"  {'Sortino Ratio':<35} {tm['sortino_ratio']:>15.4f}")
    print(f"  {'Calmar Ratio':<35} {tm['calmar_ratio']:>15.4f}")
    print(f"  {'Win Rate (%)':<35} {tm['win_rate']*100:>15.2f}")
    print(f"  {'Final Portfolio ($)':<35} {tm['final_portfolio']:>15,.2f} "
          f"{baseline_metrics['final_portfolio']:>18,.2f}")
    print(f"{'─'*80}")
    print(f"  {'Total Trades':<35} {test_results['total_trades']:>15}")
    print(f"  {'Stop-Loss Exits':<35} {test_results['sl_triggers']:>15}")
    print(f"  {'Take-Profit Exits':<35} {test_results['tp_triggers']:>15}")
    print(f"  {'Timeout Exits':<35} {test_results['timeout_exits']:>15}")
    print(f"  {'Emergency Z-Score Exits':<35} {test_results['emergency_exits']:>15}")
    print(f"  {'Turbulence Exits':<35} {test_results['turb_exits']:>15}")
    print(f"  {'Total Fees Paid ($)':<35} {test_results['total_fees']:>15,.2f}")
    print(f"{'─'*80}")

    # ── 6. Save Metrics ──────────────────────────────────────────────────
    metrics_path = os.path.join(config.RESULTS_DIR, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(tm, f, indent=2)
    print(f"\nSUCCESS: Metrics saved: {metrics_path}")

    # ── 7. Save Trade Log ─────────────────────────────────────────────────
    if test_results['trade_log']:
        trade_log_path = os.path.join(config.RESULTS_DIR, 'test_trade_log.json')
        with open(trade_log_path, 'w', encoding='utf-8') as f:
            json.dump(test_results['trade_log'], f, indent=2, default=str)
        print(f"SUCCESS: Trade log saved: {trade_log_path}")

    # ── 8. Generate Multi-Panel Visualization ─────────────────────────────
    print("\nINFO: Generating multi-panel visualization...")
    generate_multi_panel_chart(test_results, baseline_pv, test_r)

    # ── 9. Generate Trade Annotation Chart ────────────────────────────────
    print("INFO: Generating trade annotation chart...")
    generate_trade_chart(test_results, test_r)
    
    # ── 10. Generate Interactive Plotly Chart ─────────────────────────────
    generate_interactive_chart(test_results, baseline_pv, test_r)

    print(f"\nSUCCESS: All outputs saved to {config.RESULTS_DIR}/")


def generate_multi_panel_chart(results, baseline_pv, df_raw):
    """
    4-panel chart:
    1. Normalized asset prices (both assets on same scale)
    2. Spread with z-score bands (±1, ±2)
    3. Portfolio value (agent vs baseline)
    4. Position over time (color-coded)
    """
    dates = pd.DatetimeIndex(results['dates'])
    pv = results['portfolio_values']
    pos = results['positions']
    min_len = min(len(dates), len(pv), len(pos))

    # Get raw prices for the test period
    start_idx = config.TIME_WINDOW
    close_a = df_raw['Close_A'].values[start_idx:start_idx + min_len]
    close_b = df_raw['Close_B'].values[start_idx:start_idx + min_len]
    spreads_raw = df_raw['Spread'].values[start_idx:start_idx + min_len]

    # Compute z-scores from raw spread
    spread_series = pd.Series(spreads_raw)
    spread_mean = spread_series.rolling(config.SPREAD_ZSCORE_WINDOW).mean()
    spread_std = spread_series.rolling(config.SPREAD_ZSCORE_WINDOW).std()
    zscores_raw = ((spread_series - spread_mean) / (spread_std + 1e-10)).fillna(0).values

    # Normalize prices for comparison
    norm_a = close_a / close_a[0] * 100
    norm_b = close_b / close_b[0] * 100

    # ── Create figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 16),
                              gridspec_kw={'height_ratios': [2.5, 2, 2.5, 1]})
    fig.suptitle(f'Pairs Trading RL Agent — Out-of-Sample Test\n'
                 f'{config.ASSET_A_LABEL} / {config.ASSET_B_LABEL}',
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Panel 1: Normalized Prices ────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates[:min_len], norm_a[:min_len],
             color='#2196F3', linewidth=1.2, label=config.ASSET_A_LABEL, alpha=0.9)
    ax1.plot(dates[:min_len], norm_b[:min_len],
             color='#FF9800', linewidth=1.2, label=config.ASSET_B_LABEL, alpha=0.9)
    ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('Asset Prices (Normalized to 100)')
    ax1.set_ylabel('Price Index')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(dates[0], dates[min_len - 1])

    # ── Panel 2: Spread Z-Score ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(dates[:min_len], zscores_raw[:min_len],
             color='#9C27B0', linewidth=1.0, alpha=0.8, label='Spread Z-Score')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=1, color='#FFC107', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(y=-1, color='#FFC107', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(y=2, color='#F44336', linestyle='--', alpha=0.5, linewidth=0.8,
                 label='±2 σ (Entry Signal)')
    ax2.axhline(y=-2, color='#F44336', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.fill_between(dates[:min_len], -1, 1, alpha=0.05, color='green')
    ax2.set_title('Spread Z-Score with Signal Bands')
    ax2.set_ylabel('Z-Score')
    ax2.set_ylim(-4, 4)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(dates[0], dates[min_len - 1])

    # ── Panel 3: Portfolio Value ──────────────────────────────────────────
    ax3 = axes[2]
    baseline_min = min(len(baseline_pv), min_len)
    ax3.plot(dates[:min_len], pv[:min_len],
             color='#2196F3', linewidth=1.5, label='CLSTM-PPO Agent')
    ax3.plot(dates[:baseline_min], baseline_pv[:baseline_min],
             color='#FF9800', linewidth=1.0, linestyle='--',
             label='CDF Baseline')
    ax3.axhline(y=config.INITIAL_BALANCE, color='gray',
                 linestyle=':', alpha=0.5, label='Starting Capital')
    ax3.set_title('Portfolio Value Comparison')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(dates[0], dates[min_len - 1])

    # ── Panel 4: Position ─────────────────────────────────────────────────
    ax4 = axes[3]
    pos_arr = np.array(pos[:min_len])
    colors = np.where(pos_arr > 0.05, '#4CAF50',
                       np.where(pos_arr < -0.05, '#F44336', '#BDBDBD'))
    ax4.bar(dates[:min_len], pos_arr, color=colors, alpha=0.7, width=0.03)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_title('Agent Position (Green=Long Spread, Red=Short Spread)')
    ax4.set_ylabel('Position')
    ax4.set_ylim(-1.2, 1.2)
    ax4.grid(True, alpha=0.2)
    ax4.set_xlim(dates[0], dates[min_len - 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(config.RESULTS_DIR, 'test_multi_panel_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Multi-panel chart saved: {path}")


def generate_trade_chart(results, df_raw):
    """
    Generates a detailed trade annotation chart showing entry/exit points
    with arrows, P&L, and exit reasons.
    """
    if not results['trade_log']:
        print("  No trades to annotate.")
        return

    dates = pd.DatetimeIndex(results['dates'])
    pv = results['portfolio_values']
    min_len = min(len(dates), len(pv))

    # Get spread data
    start_idx = config.TIME_WINDOW
    spreads_raw = df_raw['Spread'].values[start_idx:start_idx + min_len]

    fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                              gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Trade Annotations — Entry & Exit Points',
                 fontsize=14, fontweight='bold')

    # ── Top: Portfolio with trade markers ─────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates[:min_len], pv[:min_len],
             color='#2196F3', linewidth=1.0, alpha=0.7, label='Portfolio')

    # Annotate trades
    trade_log = results['trade_log']
    max_annotations = 50  # Limit annotations to avoid clutter

    # Filter to significant trades only
    significant_trades = [t for t in trade_log
                          if t.get('exit_reason') is not None
                          or abs(t.get('position_change', 0)) > 0.3]
    trades_to_show = significant_trades[:max_annotations]

    for trade in trades_to_show:
        step = trade['step'] - start_idx
        if step < 0 or step >= min_len:
            continue

        date = dates[step]
        portfolio_val = pv[step] if step < len(pv) else pv[-1]

        # Color by action type
        if trade.get('exit_reason'):
            color = '#F44336'  # Red for forced exits
            marker = 'v'
            label_text = trade['exit_reason']
        elif trade.get('target_position', 0) > 0.05:
            color = '#4CAF50'  # Green for long spread entry
            marker = '^'
            label_text = 'Long Spread'
        elif trade.get('target_position', 0) < -0.05:
            color = '#FF5722'  # Orange for short spread entry
            marker = 'v'
            label_text = 'Short Spread'
        else:
            color = '#9E9E9E'  # Gray for close
            marker = 'x'
            label_text = 'Close'

        ax1.scatter(date, portfolio_val, color=color, marker=marker,
                     s=60, zorder=5, alpha=0.8)

    ax1.set_title('Portfolio Value with Trade Points')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left')

    # ── Bottom: Spread with entry/exit annotations ────────────────────────
    ax2 = axes[1]
    ax2.plot(dates[:min_len], spreads_raw[:min_len],
             color='#9C27B0', linewidth=0.8, alpha=0.7, label='Spread')

    for trade in trades_to_show:
        step = trade['step'] - start_idx
        if step < 0 or step >= min_len:
            continue

        date = dates[step]
        spread_val = trade.get('spread', 0)

        if trade.get('exit_reason'):
            color = '#F44336'
            marker = 'x'
        elif trade.get('target_position', 0) > 0.05:
            color = '#4CAF50'
            marker = '^'
        elif trade.get('target_position', 0) < -0.05:
            color = '#FF5722'
            marker = 'v'
        else:
            color = '#9E9E9E'
            marker = 'o'

        ax2.scatter(date, spread_val, color=color, marker=marker,
                     s=50, zorder=5, alpha=0.8)

    ax2.set_title('Spread with Trade Points (▲=Long, ▼=Short, ✕=Exit)')
    ax2.set_ylabel('Spread Value')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(config.RESULTS_DIR, 'test_trade_annotations.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Trade annotation chart saved: {path}")

    # ── Print Trade Summary ───────────────────────────────────────────────
    print(f"\n  Trade Summary ({len(trade_log)} total trades):")
    exit_reasons = {}
    for t in trade_log:
        reason = t.get('exit_reason', 'agent_decision')
        if reason is None:
            reason = 'agent_decision'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<25} {count:>5} trades")

    # Print sample trades with details
    print(f"\n  Sample Trades (first 10):")
    print(f"  {'Step':>6} | {'Z-Score':>8} | {'Action':>7} | {'Position':>8} | "
          f"{'Cost':>10} | {'Exit Reason':<20}")
    print(f"  {'-'*75}")
    for t in trade_log[:10]:
        z = t.get('zscore', 0.0)
        z = float(z) if z is not None else 0.0
        reason = t.get('exit_reason')
        reason = reason if reason is not None else 'entry'
        print(f"  {t['step']:>6} | {z:>8.3f} | "
              f"{t['action']:>7.3f} | {t['target_position']:>8.3f} | "
              f"${t['trade_cost']:>9.2f} | "
              f"{reason:<20}")

def generate_interactive_chart(results, baseline_pv, df_raw):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  Plotly not installed. Skipping interactive chart.")
        return
        
    print("\nINFO: Generating Plotly interactive HTML chart...")
    dates = pd.DatetimeIndex(results['dates'])
    pv = results['portfolio_values']
    pos = results['positions']
    min_len = min(len(dates), len(pv), len(pos))
    
    start_idx = config.TIME_WINDOW
    close_a = df_raw['Close_A'].values[start_idx:start_idx + min_len]
    close_b = df_raw['Close_B'].values[start_idx:start_idx + min_len]
    spreads = df_raw['Spread'].values[start_idx:start_idx + min_len]
    
    coint_4h = df_raw['Cointegration_P_Value_4h'].values[start_idx:start_idx + min_len]
    coint_1d = df_raw['Cointegration_P_Value_1d'].values[start_idx:start_idx + min_len]
    vol_corr = df_raw['Volume_Corr'].values[start_idx:start_idx + min_len]
    
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03,
                        subplot_titles=('Normalized Asset Prices', 'Spread Value & Trades', 'Portfolio Performance', 'Cointegration (ADF P-Value)', 'Volume Correlation'),
                        row_heights=[0.25, 0.25, 0.2, 0.15, 0.15])
                        
    norm_a = close_a / close_a[0] * 100
    norm_b = close_b / close_b[0] * 100
    
    # 1. Prices
    fig.add_trace(go.Scatter(x=dates[:min_len], y=norm_a[:min_len], name=config.ASSET_A_LABEL, line=dict(color='#2196F3')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates[:min_len], y=norm_b[:min_len], name=config.ASSET_B_LABEL, line=dict(color='#FF9800')), row=1, col=1)
    
    # 2. Spread
    fig.add_trace(go.Scatter(x=dates[:min_len], y=spreads[:min_len], name='Spread', line=dict(color='#9C27B0')), row=2, col=1)
    
    # Add trade markers
    trade_log = results['trade_log']
    long_dates, long_spreads = [], []
    short_dates, short_spreads = [], []
    exit_dates, exit_spreads = [], []
    
    for t in trade_log:
        step = t['step'] - start_idx
        if 0 <= step < min_len:
            d = dates[step]
            s = t.get('spread', spreads[step])
            if t.get('exit_reason'):
                exit_dates.append(d)
                exit_spreads.append(s)
            elif t.get('target_position', 0) > 0.05:
                long_dates.append(d)
                long_spreads.append(s)
            elif t.get('target_position', 0) < -0.05:
                short_dates.append(d)
                short_spreads.append(s)
            else:
                exit_dates.append(d)
                exit_spreads.append(s)

    fig.add_trace(go.Scatter(x=long_dates, y=long_spreads, mode='markers', name='Long Entry', marker=dict(symbol='triangle-up', size=10, color='#4CAF50')), row=2, col=1)
    fig.add_trace(go.Scatter(x=short_dates, y=short_spreads, mode='markers', name='Short Entry', marker=dict(symbol='triangle-down', size=10, color='#FF5722')), row=2, col=1)
    fig.add_trace(go.Scatter(x=exit_dates, y=exit_spreads, mode='markers', name='Exit', marker=dict(symbol='x', size=8, color='#F44336')), row=2, col=1)

    # 3. Portfolio
    baseline_min = min(len(baseline_pv), min_len)
    fig.add_trace(go.Scatter(x=dates[:min_len], y=pv[:min_len], name='CLSTM-PPO', line=dict(color='#4CAF50')), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates[:baseline_min], y=baseline_pv[:baseline_min], name='Baseline', line=dict(color='#FF9800', dash='dash')), row=3, col=1)
    
    # 4. Cointegration
    fig.add_trace(go.Scatter(x=dates[:min_len], y=coint_4h[:min_len], name='P-Value (4h)', line=dict(color='#00BCD4')), row=4, col=1)
    fig.add_trace(go.Scatter(x=dates[:min_len], y=coint_1d[:min_len], name='P-Value (1d)', line=dict(color='#8BC34A')), row=4, col=1)
    # Highlight area where p > 0.05 (Null hypothesis not rejected = blocked/unsafe)
    fig.add_hrect(y0=0.05, y1=1.0, fillcolor="red", opacity=0.2, line_width=0, row=4, col=1, annotation_text="Not Cointegrated (Blocked)", annotation_position="top left")
    
    # 5. Correlation
    fig.add_trace(go.Scatter(x=dates[:min_len], y=vol_corr[:min_len], name='Volume Correlation', line=dict(color='#E91E63')), row=5, col=1)
    
    fig.update_layout(title=f"Interactive Pairs Trading Backtest: {config.ASSET_A_LABEL} vs {config.ASSET_B_LABEL}",
                      height=1200, template="plotly_dark", hovermode="x unified")
                      
    path = os.path.join(config.RESULTS_DIR, 'interactive_backtest.html')
    fig.write_html(path)
    print(f"  Interactive HTML chart saved: {path}")


if __name__ == '__main__':
    run_test()
