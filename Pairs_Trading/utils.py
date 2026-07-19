import numpy as np
import pandas as pd
import os
import sys
from stable_baselines3.common.callbacks import BaseCallback

# ==============================================================================
# UTILITY FUNCTIONS & CALLBACKS FOR PAIRS TRADING
# Contains the training callback, backtest runner, and metric computation.
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from environment import PairsTradingEnv


class PairsTradingCallback(BaseCallback):
    """
    Training callback that monitors episode-level pairs trading performance.
    Prints spread-specific metrics like position balance and spread P&L.
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self._ep_reward = 0.0
        self._ep_length = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals['rewards'][0]
        self._ep_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)

            infos = self.locals.get('infos', [{}])
            if infos and 'portfolio_value' in infos[0]:
                self.portfolio_values.append(infos[0]['portfolio_value'])

            self._ep_reward = 0.0
            self._ep_length = 0

            n = len(self.episode_rewards)
            if self.verbose and n % 5 == 0:
                avg_r = np.mean(self.episode_rewards[-5:])
                last_p = self.portfolio_values[-1] if self.portfolio_values else 0
                trades = infos[0].get('total_trades', 0) if infos else 0
                print(f"   Ep {n:4d} | Avg Reward: {avg_r:8.5f} | "
                      f"Portfolio: ${last_p:12,.2f} | Trades: {trades}")

        return True


def run_agent(model, df_scaled, df_original, feature_columns,
              initial_balance=config.INITIAL_BALANCE):
    """
    Runs the trained pairs trading agent through a dataset.

    Args:
        model: Trained PPO model
        df_scaled: Scaled DataFrame (features are normalized)
        df_original: Unscaled DataFrame (for display prices)
        feature_columns: List of feature column names
        initial_balance: Starting capital

    Returns:
        dict with portfolio values, positions, actions, trade log, and metrics
    """
    env = PairsTradingEnv(
        df=df_scaled,
        feature_columns=feature_columns,
        initial_balance=initial_balance,
    )
    obs, _ = env.reset()

    portfolio_values = [initial_balance]
    positions = [0.0]
    actions_taken = []
    spreads = []
    zscores = []

    start_idx = config.TIME_WINDOW
    dates = list(df_original.index)[start_idx:]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        portfolio_values.append(info['portfolio_value'])
        positions.append(info['position'])
        actions_taken.append(float(action[0]))
        spreads.append(info.get('spread', 0.0))
        zscores.append(info.get('zscore', 0.0))

        if terminated or truncated:
            break

    # Align lengths
    min_len = min(len(portfolio_values), len(dates))
    portfolio_values = portfolio_values[:min_len]
    dates = dates[:min_len]
    positions = positions[:min_len]

    # Compute metrics
    metrics = compute_metrics(
        portfolio_values, config.TIMEFRAME, initial_balance
    )

    return {
        'portfolio_values': portfolio_values,
        'positions': positions,
        'actions': actions_taken,
        'dates': dates,
        'spreads': spreads[:min_len] if len(spreads) >= min_len else spreads,
        'zscores': zscores[:min_len] if len(zscores) >= min_len else zscores,
        'trade_log': env.trade_log,
        'total_trades': env.total_trades,
        'total_fees': env.total_fees_paid,
        'sl_triggers': env.risk_manager.sl_count,
        'tp_triggers': env.risk_manager.tp_count,
        'timeout_exits': env.risk_manager.timeout_count,
        'emergency_exits': env.risk_manager.emergency_count,
        'turb_exits': env.turb_exits,
        'metrics': metrics,
    }


def compute_metrics(portfolio_values, timeframe='1h', initial_balance=100_000):
    """
    Computes standard trading performance metrics.

    Returns:
        dict with total_return, sharpe_ratio, max_drawdown, win_rate,
             sortino_ratio, final_portfolio, calmar_ratio
    """
    pv = np.array(portfolio_values)

    # Returns
    rets = np.diff(pv) / (pv[:-1] + 1e-8)
    rets = rets[np.isfinite(rets)]

    # Total return
    total_return = (pv[-1] / pv[0]) - 1

    # Annualization factor
    ann_factor = np.sqrt(8760) if timeframe == '1h' else np.sqrt(365)

    # Sharpe ratio
    sharpe = (np.mean(rets) / (np.std(rets) + 1e-8)) * ann_factor \
        if len(rets) > 1 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / (peak + 1e-8)
    max_drawdown = float(np.max(drawdown))

    # Win rate
    win_rate = float(np.sum(rets > 0) / len(rets)) if len(rets) > 0 else 0.0

    # Sortino ratio (downside risk only)
    down_rets = rets[rets < 0]
    sortino = (np.mean(rets) / (np.std(down_rets) + 1e-8)) * ann_factor \
        if len(down_rets) > 0 else float('inf')

    # Calmar ratio (annualized return / max drawdown)
    n_hours = len(pv)
    ann_return = (pv[-1] / pv[0]) ** (8760 / max(n_hours, 1)) - 1 \
        if timeframe == '1h' else (pv[-1] / pv[0]) ** (365 / max(n_hours, 1)) - 1
    calmar = ann_return / (max_drawdown + 1e-8)

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'final_portfolio': float(pv[-1]),
    }


def compute_spread_baseline(df, initial_balance=config.INITIAL_BALANCE):
    """
    Computes a KDE CDF threshold baseline strategy for comparison.
    Rules:
        - Long spread when CDF < 0.05  (spread unusually low)
        - Short spread when CDF > 0.95 (spread unusually high)
        - Close when 0.4 < CDF < 0.6   (spread near mean)

    Returns:
        list of portfolio values
    """
    close_a = df['Close_A'].values
    close_b = df['Close_B'].values
    
    # Use CDF instead of Z-scores if available
    if 'Spread_CDF_KDE' in df.columns:
        cdfs = df['Spread_CDF_KDE'].values
    else:
        # Fallback to Z-score mapped to CDF proxy using normal CDF
        from scipy.stats import norm
        spread_series = pd.Series(df['Spread'].values)
        spread_mean = spread_series.rolling(config.SPREAD_ZSCORE_WINDOW).mean()
        spread_std = spread_series.rolling(config.SPREAD_ZSCORE_WINDOW).std()
        zscores = ((spread_series - spread_mean) / (spread_std + 1e-10)).fillna(0).values
        cdfs = norm.cdf(zscores)

    funding_a = df['Funding_Rate_A'].values if 'Funding_Rate_A' in df.columns else np.zeros(len(close_a))
    funding_b = df['Funding_Rate_B'].values if 'Funding_Rate_B' in df.columns else np.zeros(len(close_a))

    balance = initial_balance
    position = 0  # -1, 0, +1
    entry_price_a = 0.0
    entry_price_b = 0.0
    units_a = 0.0
    units_b = 0.0
    notional = 0.0

    portfolio_values = [initial_balance]

    for i in range(1, len(close_a)):
        cdf = cdfs[i]

        # Entry signals
        if position == 0:
            if cdf < 0.05:  # Spread too low → long spread
                position = 1
                notional = balance * 0.4
                units_a = notional / close_a[i]
                units_b = notional / close_b[i]
                entry_price_a = close_a[i]
                entry_price_b = close_b[i]
                fee = notional * 2 * (config.TRANSACTION_FEE + config.SLIPPAGE)
                balance -= (notional * 2 + fee)
            elif cdf > 0.95:  # Spread too high → short spread
                position = -1
                notional = balance * 0.4
                units_a = notional / close_a[i]
                units_b = notional / close_b[i]
                entry_price_a = close_a[i]
                entry_price_b = close_b[i]
                fee = notional * 2 * (config.TRANSACTION_FEE + config.SLIPPAGE)
                balance -= (notional * 2 + fee)

        # Exit signals
        elif 0.4 < cdf < 0.6:
            if position == 1:
                pnl = (units_a * (close_a[i] - entry_price_a) +
                       units_b * (entry_price_b - close_b[i]))
            else:
                pnl = (units_a * (entry_price_a - close_a[i]) +
                       units_b * (close_b[i] - entry_price_b))
            fee = notional * 2 * (config.TRANSACTION_FEE + config.SLIPPAGE)
            balance += notional * 2 + pnl - fee
            position = 0
            units_a = units_b = 0.0

        # Mark to market and funding
        if position != 0:
            hourly_fund_a = funding_a[i] / 8.0
            hourly_fund_b = funding_b[i] / 8.0
            notional_a = units_a * close_a[i]
            notional_b = units_b * close_b[i]

            if position == 1:
                pnl = (units_a * (close_a[i] - entry_price_a) +
                       units_b * (entry_price_b - close_b[i]))
                funding_cost = (notional_a * hourly_fund_a) + (notional_b * -hourly_fund_b)
            else:
                pnl = (units_a * (entry_price_a - close_a[i]) +
                       units_b * (close_b[i] - entry_price_b))
                funding_cost = (notional_a * -hourly_fund_a) + (notional_b * hourly_fund_b)
            
            balance -= funding_cost
            pv = balance + notional * 2 + pnl
        else:
            pv = balance

        portfolio_values.append(max(0.0, pv))

    return portfolio_values
