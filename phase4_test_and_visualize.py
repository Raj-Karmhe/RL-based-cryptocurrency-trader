"""
phase4_test_and_visualize.py — Out-of-Sample Testing & Interactive Visualization
==================================================================================
Phase 4 — Final Evaluation.

WHAT THIS DOES
--------------
1.  Loads the trained CLSTM-PPO model.
2.  Runs a full backtest on the UNSEEN test split.
3.  Computes all required metrics: Return, Drawdown, Sharpe, Win Rate.
4.  Generates an INTERACTIVE Plotly HTML dashboard with synchronized crosshair:

    ┌──────────────────────────────────────────────────────────────┐
    │  Panel 1: BTC/USDT Price Chart                               │
    │    • Actual close price line                                  │
    │    • Fibonacci retracement levels (from recent swing H/L)    │
    │    • Buy annotations (▲ green) at entry prices               │
    │    • Sell/Short annotations (▼ red) at exit prices           │
    │    • Hover tooltip: Date, Price, Action Reason               │
    ├──────────────────────────────────────────────────────────────┤
    │  Panel 2: Agent Position Over Time (−1 to +1)                │
    │    • Filled area chart: green=long, red=short, grey=flat     │
    │    • Hover crosshair synced with Panel 1                     │
    ├──────────────────────────────────────────────────────────────┤
    │  Panel 3: Portfolio Value vs Buy & Hold                      │
    │    • Agent portfolio value (blue)                            │
    │    • Buy & Hold baseline (orange dashed)                     │
    │    • Hover crosshair synced with Panels 1 & 2               │
    └──────────────────────────────────────────────────────────────┘

    Moving the mouse over ANY panel shows a vertical line on ALL panels
    simultaneously, with values at that timestamp.

5.  Saves:
    • results/interactive_backtest.html  ← open in any browser
    • results/test_metrics.json
    • results/test_trades.csv
    • results/test_summary_chart.png     ← static matplotlib fallback

FIBONACCI RETRACEMENT
----------------------
Fibonacci retracement levels (23.6%, 38.2%, 50.0%, 61.8%, 78.6%) are
computed dynamically from the MOST RECENT significant swing high and low
within a rolling 200-bar window.  They are displayed as horizontal dashed
lines on the price chart — useful as reference levels for understanding
where the agent is taking positions relative to key Fibonacci zones.

Note: The ATR-based stop-loss / take-profit in the RL environment are kept
as the primary risk management mechanism because they adapt to current
volatility.  Fibonacci levels serve as complementary context.
"""

import json
import os
import sys
import io
import warnings

# Force UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv
from phase3_train import run_backtest, compute_turbulence
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ──────────────────────────────────────────────────────────────────────────────
# FIBONACCI RETRACEMENT HELPER
# ──────────────────────────────────────────────────────────────────────────────

FIBO_LEVELS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
FIBO_COLORS = [
    "#e74c3c",   # 0.0   — swing low / red
    "#e67e22",   # 23.6%
    "#f1c40f",   # 38.2% — key level
    "#2ecc71",   # 50.0% — key level
    "#3498db",   # 61.8% — key level (golden ratio)
    "#9b59b6",   # 78.6%
    "#1abc9c",   # 1.0   — swing high / teal
]
FIBO_LABELS = ["0% (Swing Low)", "23.6%", "38.2%",
               "50%", "61.8% (Golden)", "78.6%", "100% (Swing High)"]


def compute_fibonacci_levels(prices: np.ndarray,
                              window: int = 200) -> dict:
    """
    Computes dynamic Fibonacci retracement levels from the most recent
    swing high and low within a rolling `window`-bar lookback.

    Parameters
    ----------
    prices : Array of close prices
    window : Lookback window for swing detection (default 200 bars)

    Returns
    -------
    dict  {level_pct: price_level}
    """
    lookback = prices[-window:]
    swing_high = np.max(lookback)
    swing_low  = np.min(lookback)
    price_range = swing_high - swing_low

    levels = {}
    for lvl in FIBO_LEVELS:
        # Retracement from high: swing_high - lvl * range
        levels[lvl] = swing_high - lvl * price_range
    return levels, swing_high, swing_low


# ──────────────────────────────────────────────────────────────────────────────
# TRADE LOG PARSER
# ──────────────────────────────────────────────────────────────────────────────

def parse_trades(trade_log: list, dates: list, prices: list) -> pd.DataFrame:
    """
    Converts the raw trade_log list into a clean annotated DataFrame.

    Each row represents a significant position change, annotated with:
    - Timestamp, Price, Action type (Buy / Sell / Short / Cover)
    - Exit reason (if force-exit: stop_loss / take_profit / turbulence)
    - Position size change
    """
    def _to_naive(val):
        """Convert any date-like value to a tz-naive pd.Timestamp."""
        if isinstance(val, str):
            val = pd.Timestamp(val)
        if hasattr(val, 'tzinfo') and val.tzinfo is not None:
            return val.tz_localize(None)
        return val

    records = []
    for t in trade_log:
        # Bug 1 fix: prefer the timestamp stored directly in the trade_log entry
        # (set in env.step()), falling back to index math for legacy logs.
        stored_date = t.get("date")
        date = None
        if stored_date is not None:
            try:
                date = _to_naive(stored_date)
            except ValueError:
                # If stored_date is something like "24" (integer string), parsing will fail.
                pass
        
        if date is None:
            idx = t.get("step", 0) - config.SEQ_LEN
            if idx >= 0 and idx < len(dates):
                date = _to_naive(dates[idx])
            else:
                continue

        prev_pos   = t.get("prev_position", 0.0)
        new_pos    = t.get("target_position", 0.0)
        delta      = t.get("position_delta", new_pos - prev_pos)
        force_exit = t.get("force_exit", False)
        reason     = t.get("exit_reason", None)

        # Determine action label
        if force_exit and reason:
            action_label = {
                "stop_loss":   "🛑 Stop-Loss Exit",
                "take_profit": "✅ Take-Profit Exit",
                "turbulence":  "⚡ Turbulence Exit",
            }.get(reason, f"Exit ({reason})")
        elif abs(new_pos) > abs(prev_pos):
            if new_pos > 0:
                action_label = "🟢 Long Entry"
            else:
                action_label = "🔴 Short Entry"
        else:
            if new_pos == 0:
                action_label = "⬜ Close Position"
            elif new_pos > 0:
                action_label = "🔼 Increase Long"
            else:
                action_label = "🔽 Increase Short"

        records.append({
            "date":           date,
            "price":          t.get("price", 0),
            "prev_position":  prev_pos,
            "new_position":   new_pos,
            "position_delta": delta,
            "action":         action_label,
            "exit_reason":    reason,
            "trade_cost":     t.get("trade_cost", 0),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# INTERACTIVE PLOTLY DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

def build_interactive_chart(
    dates:            list,
    prices:           list,
    portfolio_values: list,
    positions:        list,
    trade_log:        list,
    bh_portfolio:     np.ndarray,
    metrics:          dict,
    save_path:        str,
    features_dict:    dict = None,
):
    """
    Builds a 3-panel interactive Plotly dashboard with synchronized crosshair.

    Panels
    ------
    1. Price chart + Fibonacci levels + trade entry/exit markers
    2. Agent position size over time (filled area, −1 to +1)
    3. Portfolio value vs Buy & Hold
    4. Golden Features (Agent's input observations)

    Features
    --------
    • Synchronized vertical crosshair across all 3 panels
    • Hover tooltips showing Date, Price, Position, Portfolio Value
    • Fibonacci retracement reference levels on price panel
    • Annotated trade markers (entries, exits, forced exits)
    """
    dates_pd = pd.to_datetime(dates)
    n        = min(len(dates), len(prices), len(portfolio_values), len(positions))
    dates_pd = dates_pd[:n]
    prices   = np.array(prices[:n])
    pv       = np.array(portfolio_values[:n])
    pos      = np.array(positions[:n])
    bh_pv    = np.array(bh_portfolio[:n])

    # ── Parse trade annotations ────────────────────────────────────────────
    trades_df = parse_trades(trade_log, list(dates_pd), list(prices))

    # ── Fibonacci Levels (from last 200 bars of test set) ─────────────────
    fibo_levels, swing_high, swing_low = compute_fibonacci_levels(prices)

    # ── Colour-code positions (green=long, red=short, grey=flat) ──────────
    pos_colors = np.where(pos > 0.02, "rgba(46,204,113,0.4)",
                 np.where(pos < -0.02, "rgba(231,76,60,0.4)",
                 "rgba(149,165,166,0.2)"))

    # ── Create 4-row subplot figure ───────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,          # KEY: shared x-axis = synced crosshair
        vertical_spacing=0.04,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        subplot_titles=[
            "📈 BTC/USDT Price + Fibonacci Levels + Trade Annotations",
            "📊 Agent Position (−1=Full Short | 0=Flat | +1=Full Long)",
            "💼 Portfolio Value: CLSTM-PPO vs Buy & Hold",
            "🔍 Agent Observations (Golden Features)",
        ],
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 1: Price Chart
    # ═══════════════════════════════════════════════════════════════════════
    fig.add_trace(
        go.Scatter(
            x=dates_pd, y=prices,
            mode="lines",
            name="BTC/USDT Close",
            line=dict(color="#2c3e50", width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                          "Price: $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1
    )

    # ── Fibonacci retracement levels ───────────────────────────────────────
    for lvl, price_val, color, label in zip(
        FIBO_LEVELS, fibo_levels.values(), FIBO_COLORS, FIBO_LABELS
    ):
        fig.add_hline(
            y=price_val, line_dash="dot", line_color=color,
            line_width=1.2, opacity=0.7,
            annotation_text=f"Fib {label}: ${price_val:,.0f}",
            annotation_position="right",
            annotation_font_size=9,
            row=1, col=1,
        )

    # ── Trade annotations ──────────────────────────────────────────────────
    if not trades_df.empty:
        # Long entries (green up triangles)
        long_entries = trades_df[trades_df["action"].str.contains("Long Entry|Increase Long", na=False)]
        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(long_entries["date"]),
                    y=long_entries["price"],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", size=10,
                                color="#27ae60", line=dict(color="white", width=1)),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                                  "LONG ENTRY<br>Price: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1
            )

        # Short entries (red down triangles)
        short_entries = trades_df[trades_df["action"].str.contains("Short Entry|Increase Short", na=False)]
        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(short_entries["date"]),
                    y=short_entries["price"],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", size=10,
                                color="#e74c3c", line=dict(color="white", width=1)),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                                  "SHORT ENTRY<br>Price: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1
            )

        # Stop-loss exits (red X)
        sl_exits = trades_df[trades_df["exit_reason"] == "stop_loss"]
        if not sl_exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sl_exits["date"]),
                    y=sl_exits["price"],
                    mode="markers+text",
                    name="Stop-Loss Hit",
                    text=["SL"] * len(sl_exits),
                    textposition="top center",
                    textfont=dict(size=8, color="#c0392b"),
                    marker=dict(symbol="x", size=12, color="#c0392b",
                                line=dict(color="white", width=1.5)),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                                  "🛑 STOP-LOSS EXIT<br>Price: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1
            )

        # Take-profit exits (green star)
        tp_exits = trades_df[trades_df["exit_reason"] == "take_profit"]
        if not tp_exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(tp_exits["date"]),
                    y=tp_exits["price"],
                    mode="markers+text",
                    name="Take-Profit Hit",
                    text=["TP"] * len(tp_exits),
                    textposition="top center",
                    textfont=dict(size=8, color="#27ae60"),
                    marker=dict(symbol="star", size=12, color="#27ae60",
                                line=dict(color="white", width=1)),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                                  "✅ TAKE-PROFIT EXIT<br>Price: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1
            )

        # Turbulence exits (orange lightning)
        turb_exits = trades_df[trades_df["exit_reason"] == "turbulence"]
        if not turb_exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(turb_exits["date"]),
                    y=turb_exits["price"],
                    mode="markers",
                    name="Turbulence Exit",
                    marker=dict(symbol="diamond", size=10, color="#f39c12",
                                line=dict(color="white", width=1)),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                                  "⚡ TURBULENCE EXIT<br>Price: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1
            )

        # Close position markers (grey square)
        close_pos = trades_df[trades_df["action"].str.contains("Close", na=False)]
        if not close_pos.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(close_pos["date"]),
                    y=close_pos["price"],
                    mode="markers",
                    name="Close Position",
                    marker=dict(symbol="square", size=7, color="#95a5a6",
                                line=dict(color="white", width=1)),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                                  "⬜ CLOSE POSITION<br>Price: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1
            )

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 2: Agent Position Over Time (filled area)
    # ═══════════════════════════════════════════════════════════════════════
    # Split position into long and short components for separate coloring
    pos_long  = np.where(pos >= 0, pos,  0.0)
    pos_short = np.where(pos <  0, pos,  0.0)

    fig.add_trace(
        go.Scatter(
            x=dates_pd, y=pos_long,
            fill="tozeroy",
            mode="none",
            name="Long Allocation",
            fillcolor="rgba(46,204,113,0.45)",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                          "Long: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates_pd, y=pos_short,
            fill="tozeroy",
            mode="none",
            name="Short Allocation",
            fillcolor="rgba(231,76,60,0.45)",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                          "Short: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1
    )
    # Zero line for reference
    fig.add_hline(y=0, line_dash="solid", line_color="#7f8c8d",
                  line_width=0.8, row=2, col=1)

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 3: Portfolio Value vs Buy & Hold
    # ═══════════════════════════════════════════════════════════════════════
    fig.add_trace(
        go.Scatter(
            x=dates_pd, y=pv,
            mode="lines",
            name="CLSTM-PPO Agent",
            line=dict(color="#3498db", width=2),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                          "Agent: $%{y:,.0f}<extra></extra>",
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates_pd, y=bh_pv,
            mode="lines",
            name="Buy & Hold (BTC)",
            line=dict(color="#e67e22", width=2, dash="dot"),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                          "Buy & Hold: $%{y:,.0f}<extra></extra>",
        ),
        row=3, col=1
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 4: Golden Features (Agent Observations)
    # ═══════════════════════════════════════════════════════════════════════
    if features_dict:
        for feat_name, feat_values in features_dict.items():
            feat_vals = np.array(feat_values[:n])
            fig.add_trace(
                go.Scatter(
                    x=dates_pd, y=feat_vals,
                    mode="lines",
                    name=feat_name,
                    line=dict(width=1),
                    hovertemplate=f"<b>%{{x|%Y-%m-%d %H:%M}}</b><br>"
                                  f"{feat_name}: %{{y:.4f}}<extra></extra>",
                ),
                row=4, col=1
            )

    # ── Metrics annotation box ─────────────────────────────────────────────
    agent_return = metrics["total_return"] * 100
    bh_return    = (bh_pv[-1] / bh_pv[0] - 1) * 100
    sharpe       = metrics["sharpe_ratio"]
    max_dd       = metrics["max_drawdown"] * 100
    win_rate     = metrics["win_rate"] * 100

    annotation_text = (
        f"<b>Agent:</b> Return={agent_return:+.1f}%  |  "
        f"Sharpe={sharpe:.2f}  |  MaxDD={max_dd:.1f}%  |  WinRate={win_rate:.1f}%<br>"
        f"<b>B&H:</b>  Return={bh_return:+.1f}%"
    )
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        xanchor="left", yanchor="bottom",
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#bdc3c7",
        borderwidth=1,
        borderpad=6,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # LAYOUT — synchronized crosshair & interactive settings
    # ═══════════════════════════════════════════════════════════════════════
    fig.update_layout(
        title=dict(
            text="<b>CLSTM-PPO Cryptocurrency Trading Agent — Backtest Analysis</b>",
            font=dict(size=16),
        ),
        height=950,
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", size=11, color="#2c3e50"),
        hovermode="x unified",      # ← KEY: shows all panel values at the same x
        hoverdistance=1,
        spikedistance=-1,           # show spike line across all subplots
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            font=dict(size=10),
        ),
        xaxis=dict(
            showspikes=True,        # vertical spike line
            spikemode="across",     # extends across all panels
            spikesnap="cursor",
            spikecolor="#7f8c8d",
            spikedash="dot",
            spikethickness=1.5,
            showline=True, linecolor="#bdc3c7",
            gridcolor="#ecf0f1",
        ),
        xaxis2=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#7f8c8d",
            spikedash="dot",
            spikethickness=1.5,
            showline=True, linecolor="#bdc3c7",
            gridcolor="#ecf0f1",
        ),
        xaxis3=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#7f8c8d",
            spikedash="dot",
            spikethickness=1.5,
            showline=True, linecolor="#bdc3c7",
            gridcolor="#ecf0f1",
        ),
        xaxis4=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#7f8c8d",
            spikedash="dot",
            spikethickness=1.5,
            showline=True, linecolor="#bdc3c7",
            gridcolor="#ecf0f1",
        ),
    )

    # Axis labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                     gridcolor="#ecf0f1", zeroline=False)
    fig.update_yaxes(title_text="Position", row=2, col=1,
                     range=[-1.1, 1.1], gridcolor="#ecf0f1",
                     zeroline=True, zerolinecolor="#7f8c8d")
    fig.update_yaxes(title_text="Portfolio ($)", row=3, col=1,
                     gridcolor="#ecf0f1", zeroline=False)
    fig.update_yaxes(title_text="Feature Value", row=4, col=1,
                     gridcolor="#ecf0f1", zeroline=True)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    # Save to HTML
    pio.write_html(fig, file=save_path, include_plotlyjs="cdn",
                   config={'scrollZoom': True})
    try:
        pio.write_image(fig, file=save_path.replace(".html", ".png"), width=1600, height=1000)
    except Exception as e:
        print(f"Warning: Could not save static image. {e}")
    print(f"  [Plot] Interactive dashboard saved → {save_path}")
    print(f"         Open in any browser — move mouse to see synced crosshair!")


# ──────────────────────────────────────────────────────────────────────────────
# STATIC MATPLOTLIB FALLBACK CHART
# ──────────────────────────────────────────────────────────────────────────────

def save_static_chart(dates, prices, portfolio_values, positions,
                      bh_portfolio, metrics, save_path):
    """
    Saves a static 3-panel matplotlib figure as a PNG fallback.
    """
    dates_pd = pd.to_datetime(dates)
    n        = min(len(dates), len(prices), len(portfolio_values), len(positions))
    dates_pd = dates_pd[:n]
    prices   = np.array(prices[:n])
    pv       = np.array(portfolio_values[:n])
    pos      = np.array(positions[:n])
    bh_pv    = np.array(bh_portfolio[:n])

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
                             sharex=True)

    # Panel 1: Price
    axes[0].plot(dates_pd, prices, color="#2c3e50", lw=1, label="BTC/USDT")
    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title("CLSTM-PPO Backtest — Price + Position + Portfolio",
                      fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # Panel 2: Position
    axes[1].fill_between(dates_pd, pos, 0,
                         where=(pos >= 0), alpha=0.55, color="#27ae60", label="Long")
    axes[1].fill_between(dates_pd, pos, 0,
                         where=(pos < 0),  alpha=0.55, color="#e74c3c", label="Short")
    axes[1].axhline(0, color="#7f8c8d", lw=0.8)
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].set_ylabel("Position")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # Panel 3: Portfolio
    axes[2].plot(dates_pd, pv,    color="#3498db", lw=2, label="CLSTM-PPO")
    axes[2].plot(dates_pd, bh_pv, color="#e67e22", lw=2, ls="--", label="Buy & Hold")
    axes[2].set_ylabel("Portfolio ($)")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Static fallback chart saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TESTING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def run_test_and_visualize(override_fee: float = None, override_slippage: float = None, model_override: str = None, coin_override: str = None, eval_set: str = "test"):
    """
    Main Phase 4 function.
    """
    print("\n" + "=" * 70)
    print(f"  PHASE 4 — OUT-OF-SAMPLE TEST & VISUALIZATION ({eval_set.upper()} SET)")
    print("=" * 70)

    # Override config if requested
    if override_fee is not None:
        config.TRANSACTION_FEE = override_fee
        print(f"  [Override] Transaction Fee set to: {override_fee}")
    if override_slippage is not None:
        config.SLIPPAGE = override_slippage
        print(f"  [Override] Slippage set to: {override_slippage}")

    coins_to_test = [coin_override] if coin_override else config.MULTI_COINS

    model_path = model_override if model_override else config.MODEL_PATH
    model_zip = model_path if model_path.endswith(".zip") else model_path + ".zip"

    if not os.path.exists(model_zip):
        raise FileNotFoundError(f"{model_zip} not found. Run Phase 3 first.")

    # Auto-detect the correct feature list from the model's stored observation space.
    # This allows old 22-feature models and new 10-feature models to both work correctly.
    _probe = PPO.load(model_zip[:-4])
    _obs_dim = _probe.observation_space.shape[0]
    _n_feat  = (_obs_dim - 2) // config.SEQ_LEN   # subtract 2 portfolio state dims

    registry = getattr(config, "FEATURE_REGISTRY", {})
    if _n_feat in registry and os.path.exists(registry[_n_feat]):
        features_path = registry[_n_feat]
    else:
        features_path = config.GOLDEN_FEATURES_PATH   # fallback to default

    with open(features_path) as fp:
        golden_features = json.load(fp)

    print(f"  [Features] Auto-detected {_n_feat} features (obs dim={_obs_dim}) "
          f"-> loaded from {os.path.basename(features_path)}")
    del _probe


    all_portfolio_values = []
    bh_portfolio_values = []
    all_dates = []
    summary_metrics = []
    
    # ── Loop over all coins ───────────────────────────────────────────────
    for target_coin in coins_to_test:
        sym_file = target_coin.replace('/', '_')
        print("\n" + "─" * 70)
        print(f"  [Target Coin] Evaluating on: {target_coin} ({eval_set} set)")
        print("─" * 70)

        train_feat_path = os.path.join(config.DATA_DIR, f"{sym_file}_train_features.csv")
        full_train_df   = pd.read_csv(train_feat_path, index_col=0, parse_dates=True)

        if eval_set == "val":
            split_idx = int(len(full_train_df) * 0.7)
            test_df   = full_train_df.iloc[split_idx:].copy()
            train_df  = full_train_df.iloc[:split_idx]
        else:
            test_feat_path = os.path.join(config.DATA_DIR, f"{sym_file}_test_features.csv")
            if not os.path.exists(test_feat_path):
                print(f"  [Warning] {test_feat_path} not found. Skipping.")
                continue
            test_df  = pd.read_csv(test_feat_path, index_col=0, parse_dates=True)
            train_df = full_train_df

        train_turb = compute_turbulence(train_df['Close'])
        turb_threshold = float(np.nanpercentile(train_turb[train_turb > 0], config.TURBULENCE_PERCENTILE))
        test_df['Turbulence'] = compute_turbulence(test_df['Close']).values

        # ── Regime Volatility Leakage Fix (Bug 2) ──
        from reward_functions import compute_rolling_std
        train_sigmas = compute_rolling_std(train_df['Close'].values)
        config.MEAN_VOL_TRAINING = float(np.nanmean(train_sigmas))

        # Probe model first (no env) to read its obs + action spaces.
        # This lets old long-only 22-feat models and new long+short 10-feat models
        # both load correctly without touching config.py.
        try:
            model_no_env      = PPO.load(model_zip[:-4])
            model_obs_dim     = model_no_env.observation_space.shape[0]
            model_act_low     = float(model_no_env.action_space.low[0])
            model_allow_short = (model_act_low < 0.0)   # True if trained with shorts
            del model_no_env

            dummy_env = DummyVecEnv([lambda: CryptoTradingEnv(
                test_df, golden_features,
                initial_balance  = config.INITIAL_BALANCE,
                turb_threshold   = turb_threshold,
                allow_short      = model_allow_short,
            )])

            env_obs_dim = dummy_env.observation_space.shape[0]

            if model_obs_dim != env_obs_dim:
                seq_len   = config.SEQ_LEN
                model_n   = (model_obs_dim - 2) // seq_len
                current_n = len(golden_features)
                print(f"\n  [ERROR] Observation space mismatch for {target_coin}!")
                print(f"    Model expects : {model_n} features (obs dim = {model_obs_dim})")
                print(f"    Env provides  : {current_n} features (obs dim = {env_obs_dim})")
                regime = "bear" if "bear" in model_zip else "bull" if "bull" in model_zip else "crab"
                print(f"  --> Retrain: python phase3_train.py --regime {regime}")
                continue

            # Use custom_objects to override stored spaces — this bypasses SB3's
            # strict space check, allowing old long-only models to run in any env.
            model = PPO.load(
                model_zip[:-4],
                env=dummy_env,
                custom_objects={
                    "observation_space": dummy_env.observation_space,
                    "action_space":      dummy_env.action_space,
                },
            )

        except Exception as e:
            print(f"\n  [ERROR] Could not load model: {e}")
            print(f"  --> Retrain with: python phase3_train.py --regime <regime>")
            continue


        results  = run_backtest(model, test_df, golden_features, turb_threshold=turb_threshold)
        metrics  = results["metrics"]
        dates    = results["dates"]
        prices   = results["prices"]
        pv       = results["portfolio_values"]
        pos      = results["positions"]
        trades   = results["trade_log"]

        all_portfolio_values.append(pv)
        all_dates.append(dates)
        
        test_prices = np.array(prices)
        bh_pv       = (test_prices / test_prices[0]) * config.INITIAL_BALANCE if len(test_prices) > 0 else np.array([config.INITIAL_BALANCE])
        bh_portfolio_values.append(bh_pv)
        bh_return   = float((bh_pv[-1] / bh_pv[0]) - 1) if len(bh_pv) > 1 else 0.0

        bh_rets_arr = np.diff(bh_pv) / (bh_pv[:-1] + 1e-8)
        bh_sharpe = float((np.mean(bh_rets_arr) / (np.std(bh_rets_arr) + 1e-8)) * np.sqrt(8760)) if len(bh_rets_arr) > 1 else 0.0
        bh_peak = np.maximum.accumulate(bh_pv)
        bh_max_dd = float(np.max((bh_peak - bh_pv) / (bh_peak + 1e-8)))

        summary_metrics.append({
            'Asset': sym_file,
            'Agent Return (%)': metrics['total_return'] * 100,
            'B&H Return (%)': bh_return * 100,
            'Agent Max DD (%)': metrics['max_drawdown'] * 100,
            'B&H Max DD (%)': bh_max_dd * 100,
            'Agent Sharpe': metrics['sharpe_ratio'],
            'B&H Sharpe': bh_sharpe,
            'Win Rate (%)': metrics['win_rate'] * 100,
            'Trades': results['total_trades']
        })
        
        html_path = os.path.join(config.RESULTS_DIR, f"interactive_backtest_{sym_file}_{eval_set}.html")
        features_dict = {feat: test_df[feat].iloc[config.SEQ_LEN:config.SEQ_LEN + len(dates)].tolist() for feat in golden_features}
        build_interactive_chart(dates, list(prices), pv, pos, trades, bh_pv, metrics, html_path, features_dict)
        print(f"\n  [Plot] Interactive dashboard → {html_path}")

    # ── Combined Portfolio Metrics ─────────────────────────────────────────
    if len(all_portfolio_values) > 1:
        print("\n" + "═" * 70)
        print("  WHOLE PORTFOLIO METRICS (COMBINED ASSETS)")
        print("═" * 70)
        
        min_len = min(len(p) for p in all_portfolio_values)
        
        # Flatten to ensure 1D arrays and strictly enforce min_len
        combined_pv = np.sum([np.array(p).flatten()[:min_len] for p in all_portfolio_values], axis=0)
        combined_bh = np.sum([np.array(p).flatten()[:min_len] for p in bh_portfolio_values], axis=0)
        
        combined_rets = np.diff(combined_pv) / (combined_pv[:-1] + 1e-8)
        combined_bh_return = float((combined_bh[-1] / combined_bh[0]) - 1) if len(combined_bh) > 1 else 0.0
        combined_return = float((combined_pv[-1] / combined_pv[0]) - 1) if len(combined_pv) > 1 else 0.0
        
        ann_factor = np.sqrt(8760)
        combined_sharpe = float((np.mean(combined_rets) / (np.std(combined_rets) + 1e-8)) * ann_factor) if len(combined_rets) > 1 else 0.0
        
        peak = np.maximum.accumulate(combined_pv)
        drawdown = (peak - combined_pv) / (peak + 1e-8)
        combined_max_dd = float(np.max(drawdown))
        
        combined_bh_rets_arr = np.diff(combined_bh) / (combined_bh[:-1] + 1e-8)
        combined_bh_sharpe = float((np.mean(combined_bh_rets_arr) / (np.std(combined_bh_rets_arr) + 1e-8)) * ann_factor) if len(combined_bh_rets_arr) > 1 else 0.0
        combined_bh_peak = np.maximum.accumulate(combined_bh)
        combined_bh_max_dd = float(np.max((combined_bh_peak - combined_bh) / (combined_bh_peak + 1e-8)))

        summary_metrics.append({
            'Asset': 'COMBINED',
            'Agent Return (%)': combined_return * 100,
            'B&H Return (%)': combined_bh_return * 100,
            'Agent Max DD (%)': combined_max_dd * 100,
            'B&H Max DD (%)': combined_bh_max_dd * 100,
            'Agent Sharpe': combined_sharpe,
            'B&H Sharpe': combined_bh_sharpe,
            'Win Rate (%)': float('nan'),
            'Trades': float('nan')
        })
        
    print("\n" + "═" * 105)
    print("  SUMMARY METRICS TABLE")
    print("═" * 105)
    print(f"  {'Asset':<12} | {'A-Ret(%)':>10} | {'B&H-Ret(%)':>10} | {'A-MaxDD(%)':>10} | {'B&H-MaxDD':>10} | {'A-Sharpe':>9} | {'B&H-Sharpe':>10} | {'WinRate(%)':>10} | {'Trades':>6}")
    print("  " + "-" * 103)
    for row in summary_metrics:
        win_rate_str = f"{row['Win Rate (%)']:.2f}" if not np.isnan(row['Win Rate (%)']) else "-"
        trades_str = f"{row['Trades']}" if not np.isnan(row['Trades']) else "-"
        print(f"  {row['Asset']:<12} | {row['Agent Return (%)']:>10.2f} | {row['B&H Return (%)']:>10.2f} | {row['Agent Max DD (%)']:>10.2f} | {row['B&H Max DD (%)']:>10.2f} | {row['Agent Sharpe']:>9.2f} | {row['B&H Sharpe']:>10.2f} | {win_rate_str:>10} | {trades_str:>6}")
    print("═" * 105 + "\n")

    print("\n  PHASE 4 COMPLETE ✓")


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Test & Visualize")
    parser.add_argument("--fee", type=float, default=None,
                        help="Override default transaction fee (e.g., 0.001 for 0.1 percent)")
    parser.add_argument("--slippage", type=float, default=None,
                        help="Override default slippage (e.g., 0.0005 for 0.05 percent)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a specific model .zip file to load")
    parser.add_argument("--coin", type=str, default=None,
                        help="Coin to evaluate on (e.g., 'ETH/USDT'). Defaults to all coins.")
    parser.add_argument("--dataset", type=str, default="test", choices=["test", "val"],
                        help="Dataset to evaluate on: 'test' (default, out-of-sample) or 'val' (validation split).")
    args = parser.parse_args()

    run_test_and_visualize(override_fee=args.fee, override_slippage=args.slippage, model_override=args.model, coin_override=args.coin, eval_set=args.dataset)
