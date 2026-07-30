import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import deque

def compute_moving_average(close_prices, window=480):
    return pd.Series(close_prices).rolling(window=window, min_periods=window).mean().values

def compute_rolling_std(close_prices, window=480):
    return pd.Series(close_prices).rolling(window=window, min_periods=window).std().values

def classify_regimes(df: pd.DataFrame, mean_vol_training: float = None):
    """
    Classify each observation into a market regime.
    Run this over your entire dataset before training, and pass the array to your env.
    If mean_vol_training is None, it computes the mean over the entire dataframe.
    """
    close_col = 'close' if 'close' in df.columns else 'Close'
    close = df[close_col].values
    n = len(close)
    
    ma = compute_moving_average(close, 480)
    sigma = compute_rolling_std(close, 480)
    
    # Calculate slope: 24h average hourly change
    slope = np.full(n, np.nan)
    for t in range(24, n):
        if not np.isnan(ma[t]) and not np.isnan(ma[t-24]):
            slope[t] = (ma[t] - ma[t-24]) / 24

    if mean_vol_training is None:
        mean_vol_training = np.nanmean(sigma)
    vol_threshold = 2.0 * mean_vol_training

    regimes = np.full(n, "unclassified", dtype=object)
    
    for t in range(480, n):
        if np.isnan(sigma[t]) or np.isnan(slope[t]):
            continue
            
        # 1. High Volatility (Priority 1)
        if sigma[t] > vol_threshold:
            regimes[t] = "high_volatility"
            continue
            
        # Calculate 24-hour total change for trend rules
        total_24h_change = slope[t] * 24
        theta = 0.001 * ma[t] # 0.1% threshold
        
        # 2. Bull/Bear/Neutral
        if total_24h_change > theta:
            regimes[t] = "bull"
        elif total_24h_change < -theta:
            regimes[t] = "bear"
        else:
            regimes[t] = "neutral"
            
    return regimes, sigma


class RewardFunction(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, pnl: float, action: float, **kwargs) -> float:
        pass
        
    def reset(self):
        pass

class AdaptiveRiskControlReward(RewardFunction):
    """
    R_t = PnL_t - delta_t * sigma(PnL_{t-24:t})
    Penalizes portfolio volatility. The penalty scales automatically with broader market volatility.

    Regime-aware extensions (all applied as additive terms on top of the base reward):
    - Bull market, agent is FLAT:
        A linearly-growing "cash-drag" penalty that increases by BULL_CASH_DRAG_RATE
        for every consecutive step the agent holds zero/flat during a bull market.
        This forces the agent to eventually take a position or suffer escalating pain.
    - Bull market, agent ENTERS a long position (position_delta > 0):
        A one-time BULL_ENTRY_BONUS reward to reinforce good timing.
    - Bear market, agent ENTERS any position (|position_delta| > threshold):
        A BEAR_ENTRY_PENALTY (half the magnitude of the original bear penalty to
        be less draconian while still discouraging reckless entries).
    """
    def __init__(self,
                 delta_min=0.01, delta_max=0.5, n=24,
                 sigma_min=0.0, sigma_max=2000.0, alpha=1.0,
                 bull_cash_drag_rate: float = 0.0002,
                 bull_entry_bonus:    float = 0.005,
                 bear_entry_penalty:  float = 0.005,
                 bull_hold_bonus_max: float = 0.002):
        super().__init__("adaptive_risk")
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.n = n
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.alpha = alpha
        self.pnl_history = deque(maxlen=n)

        # Regime-aware parameters
        self.bull_cash_drag_rate = bull_cash_drag_rate   # Penalty added per step in bull while flat
        self.bull_entry_bonus    = bull_entry_bonus       # One-time bonus for entering during bull
        self.bear_entry_penalty  = bear_entry_penalty     # One-time penalty for entering during bear
        self.bull_hold_bonus_max = bull_hold_bonus_max    # Max hold bonus that decays over 48 hours

        # Tracks consecutive flat/long steps during a bull regime
        self._bull_flat_steps = 0
        self._bull_long_hold_steps = 0

    def compute(self, pnl: float, action: float, **kwargs) -> float:
        scaled_pnl = self.alpha * pnl
        self.pnl_history.append(scaled_pnl)

        # Calculate recent portfolio volatility
        pnl_volatility = np.std(self.pnl_history) if len(self.pnl_history) > 1 else 0.0

        # Current market volatility from kwargs
        sigma_480 = kwargs.get("sigma_480", 0.0)

        # Scale delta based on market conditions
        sigma_clamped = np.clip(sigma_480, self.sigma_min, self.sigma_max)
        if self.sigma_max > self.sigma_min:
            ratio = (sigma_clamped - self.sigma_min) / (self.sigma_max - self.sigma_min)
        else:
            ratio = 0.0

        delta_t = self.delta_min + (self.delta_max - self.delta_min) * ratio

        # Base reward: scaled PnL minus volatility risk penalty
        risk_penalty = delta_t * pnl_volatility
        reward = scaled_pnl - risk_penalty

        # ── Regime-Aware Extensions ──────────────────────────────────────────
        regime         = kwargs.get("regime", "neutral")
        position       = kwargs.get("position", 0.0)       # current position (after this step)
        position_delta = kwargs.get("position_delta", 0.0) # change in position this step
        cooldown       = kwargs.get("cooldown_counter", 0) # if >0, agent is locked out
        entry_threshold = 0.05  # Minimum |delta| to count as a meaningful entry

        if regime == "bull":
            # ── Bull: cash-drag penalty (linear escalation) ──────────────────
            if abs(position) < 0.05:   # Agent is effectively flat
                self._bull_long_hold_steps = 0
                if cooldown == 0:
                    self._bull_flat_steps += 1
                    # Penalty grows linearly: 0 at step 0, grows by drag_rate per step
                    cash_drag = self.bull_cash_drag_rate * self._bull_flat_steps
                    reward -= cash_drag
            else:
                # Agent has a position — reset the flat counter
                self._bull_flat_steps = 0
                # ── Bull: decreasing hold bonus ──────────────────────────────
                if position > 0.05: # Long position
                    self._bull_long_hold_steps += 1
                    # Bonus decreases linearly to 0 over 48 hours (2 days)
                    hold_bonus = max(0.0, self.bull_hold_bonus_max * (1.0 - self._bull_long_hold_steps / 48.0))
                    reward += hold_bonus
                else:
                    self._bull_long_hold_steps = 0

            # ── Bull: one-time entry bonus ────────────────────────────────────
            if position_delta > entry_threshold:
                reward += self.bull_entry_bonus

        elif regime == "bear":
            # ── Bear: halved entry penalty ────────────────────────────────────
            # Original full penalty would be 2 × bear_entry_penalty; here we
            # use 0.5 × to be less draconian (as requested).
            if abs(position_delta) > entry_threshold:
                reward -= self.bear_entry_penalty   # already at 0.5× intended level

            # Reset bull counters when not in bull
            self._bull_flat_steps = 0
            self._bull_long_hold_steps = 0

        else:
            # Neutral / high_volatility — reset bull counters
            self._bull_flat_steps = 0
            self._bull_long_hold_steps = 0

        return reward

    def reset(self):
        self.pnl_history.clear()
        self._bull_flat_steps = 0
        self._bull_long_hold_steps = 0


class AsymmetricMarketReward(RewardFunction):
    """
    Changes reward logic based on current market regime.

    Bull market:
      • Bonus for positive PnL (existing).
      • Linearly-growing cash-drag penalty if agent stays flat (NEW).
      • One-time entry bonus when agent opens a long (NEW).

    Bear market:
      • Penalty for negative PnL (existing, but now HALVED compared to original).
      • One-time entry penalty when agent opens any position (halved, NEW).

    High volatility:
      • Extra penalty for negative PnL (existing).
    """
    def __init__(self,
                 gamma: float = 0.1,
                 phi:   float = 0.05,    # Halved from original 0.1
                 alpha: float = 1.0,
                 bull_cash_drag_rate: float = 0.0002,
                 bull_entry_bonus:    float = 0.005,
                 bear_entry_penalty:  float = 0.005,
                 bull_hold_bonus_max: float = 0.002):
        super().__init__("asymmetric_market")
        self.gamma = gamma
        self.phi   = phi    # Bear penalty scale (halved)
        self.alpha = alpha
        self.bull_cash_drag_rate = bull_cash_drag_rate
        self.bull_entry_bonus    = bull_entry_bonus
        self.bear_entry_penalty  = bear_entry_penalty
        self.bull_hold_bonus_max = bull_hold_bonus_max
        self._bull_flat_steps    = 0
        self._bull_long_hold_steps = 0

    def compute(self, pnl: float, action: float, **kwargs) -> float:
        scaled_pnl     = self.alpha * pnl
        regime         = kwargs.get("regime", "neutral")
        position       = kwargs.get("position", 0.0)
        position_delta = kwargs.get("position_delta", 0.0)
        cooldown       = kwargs.get("cooldown_counter", 0)
        entry_threshold = 0.05

        reward = scaled_pnl

        if regime == "bull":
            # Existing: reward profitable trades extra
            if scaled_pnl > 0:
                reward += self.gamma * scaled_pnl

            # NEW: linearly-growing cash-drag penalty for staying flat (skipped if cooling down)
            if abs(position) < 0.05:
                self._bull_long_hold_steps = 0
                if cooldown == 0:
                    self._bull_flat_steps += 1
                    cash_drag = self.bull_cash_drag_rate * self._bull_flat_steps
                    reward -= cash_drag
            else:
                self._bull_flat_steps = 0
                # NEW: linearly decreasing bonus for holding long
                if position > 0.05:
                    self._bull_long_hold_steps += 1
                    hold_bonus = max(0.0, self.bull_hold_bonus_max * (1.0 - self._bull_long_hold_steps / 48.0))
                    reward += hold_bonus
                else:
                    self._bull_long_hold_steps = 0

            # NEW: one-time entry bonus for opening a long
            if position_delta > entry_threshold:
                reward += self.bull_entry_bonus

        elif regime == "bear":
            # Existing: penalize losses in bear (now halved via phi=0.05)
            if scaled_pnl < 0:
                reward -= self.phi * abs(scaled_pnl)

            # NEW: halved one-time entry penalty for opening any position in bear
            if abs(position_delta) > entry_threshold:
                reward -= self.bear_entry_penalty

            self._bull_flat_steps = 0
            self._bull_long_hold_steps = 0

        elif regime == "high_volatility":
            if scaled_pnl < 0:
                reward -= self.phi * abs(scaled_pnl) * 1.5
            self._bull_flat_steps = 0
            self._bull_long_hold_steps = 0

        else:
            self._bull_flat_steps = 0
            self._bull_long_hold_steps = 0

        return reward

    def reset(self):
        self._bull_flat_steps = 0
        self._bull_long_hold_steps = 0
