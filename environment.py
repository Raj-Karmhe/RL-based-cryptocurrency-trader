"""
environment.py

Regime-Aware Trading Environment
for PPO Agent
"""

# -------------------------------------------------------
# Imports
# -------------------------------------------------------

import gymnasium as gym
from gymnasium import spaces

import numpy as np

from portfolio import Portfolio


# -------------------------------------------------------
# Trading Environment
# -------------------------------------------------------

class TradingEnvironment(gym.Env):

    def __init__(self, df):

        super().__init__()

        # ------------------------------------------------
        # Dataset
        # ------------------------------------------------

        self.df = df.reset_index(drop=True)

        # ------------------------------------------------
        # Feature Columns
        # ------------------------------------------------

        self.feature_columns = [

            "Open",
            "High",
            "Low",
            "Close",
            "Volume",

            "SMA20",
            "SMA50",
            "EMA20",

            "RSI",

            "MACD",
            "MACD_SIGNAL",

            "BB_UPPER",
            "BB_LOWER",

            "ATR",

            "VOLUME_MA20",

            "Regime_0",
            "Regime_1",
            "Regime_2",
            "Regime_3"

        ]

        # ------------------------------------------------
        # Observation Settings
        # ------------------------------------------------

        self.window_size = 20

        self.observation_space = spaces.Box(

            low=-np.inf,

            high=np.inf,

            shape=(

                self.window_size,

                len(self.feature_columns)

            ),

            dtype=np.float32

        )

        # ------------------------------------------------
        # Continuous Action Space
        #
        # -1  -> 100% Cash
        #  0  -> 50% BTC
        # +1  -> 100% BTC
        # ------------------------------------------------

        self.action_space = spaces.Box(

            low=-1.0,

            high=1.0,

            shape=(1,),

            dtype=np.float32

        )

        # ------------------------------------------------
        # Portfolio
        # ------------------------------------------------

        self.portfolio = Portfolio(

            initial_balance=100000,

            transaction_fee=0.001,

            slippage=0.0005

        )

        # ------------------------------------------------
        # Environment Variables
        # ------------------------------------------------

        self.current_step = self.window_size

        self.previous_portfolio_value = (

            self.portfolio.portfolio_value

        )

        self.max_portfolio_value = (

            self.portfolio.portfolio_value

        )

        # ------------------------------------------------
    # Get Observation
    # ------------------------------------------------

    def _get_observation(self):

        observation = self.df.loc[
            self.current_step - self.window_size:
            self.current_step - 1,
            self.feature_columns
        ].to_numpy(dtype=np.float32)

        return observation


    # ------------------------------------------------
    # Reset Environment
    # ------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Reset portfolio
        self.portfolio.reset()

        # Reset environment variables
        self.current_step = self.window_size

        self.previous_portfolio_value = (

            self.portfolio.portfolio_value

        )

        self.max_portfolio_value = (

            self.portfolio.portfolio_value

        )

        observation = self._get_observation()

        info = {

            "portfolio_value": float(
                self.portfolio.portfolio_value
            ),

            "balance": float(
                self.portfolio.balance
            ),

            "btc": float(
                self.portfolio.btc_held
            )

        }

        return observation, info

        # ------------------------------------------------
    # Step
    # ------------------------------------------------

    def step(self, action):

        # --------------------------------------------
        # Current Market Price
        # --------------------------------------------

        current_price = float(
            self.df.loc[self.current_step, "Close"]
        )

        # --------------------------------------------
        # Convert Action
        # PPO Output : [-1, 1]
        # Target Allocation : [0, 1]
        # --------------------------------------------

        action = float(action[0])

        # Dead Zone (Hold)

        if abs(action) < 0.05:
            action = 0.0

        target_fraction = (action + 1.0) / 2.0

        # --------------------------------------------
        # Portfolio Before Trade
        # --------------------------------------------

        previous_value = self.portfolio.update_value(
            current_price
        )

        # --------------------------------------------
        # Execute Trade
        # --------------------------------------------

        self.portfolio.rebalance(
            target_fraction,
            current_price
        )

        current_value = self.portfolio.update_value(
            current_price
        )

        # --------------------------------------------
        # Reward
        # --------------------------------------------

        portfolio_return = (

            current_value - previous_value

        ) / max(previous_value, 1e-8)

        trade_penalty = 0.0005 * abs(action)

        reward = (

            portfolio_return

            -

            trade_penalty

        )

        # --------------------------------------------
        # Track Maximum Portfolio
        # --------------------------------------------

        self.max_portfolio_value = max(

            self.max_portfolio_value,

            current_value

        )

        # --------------------------------------------
        # Next Step
        # --------------------------------------------

        self.current_step += 1

        terminated = (

            self.current_step >= len(self.df)

        )

        truncated = False

        # --------------------------------------------
        # Observation
        # --------------------------------------------

        if not terminated:

            observation = self._get_observation()

        else:

            observation = np.zeros(

                self.observation_space.shape,

                dtype=np.float32

            )

        # --------------------------------------------
        # Info
        # --------------------------------------------

        info = {

            "portfolio_value": float(
                current_value
            ),

            "balance": float(
                self.portfolio.balance
            ),

            "btc_held": float(
                self.portfolio.btc_held
            ),

            "target_fraction": float(
                target_fraction
            ),

            "executed_action": float(
                action
            ),

            "step": int(
                self.current_step
            )

        }

        return (

            observation,

            reward,

            terminated,

            truncated,

            info

        )

        # ------------------------------------------------
    # Render
    # ------------------------------------------------

    def render(self):

        current_price = float(
            self.df.loc[
                min(self.current_step, len(self.df) - 1),
                "Close"
            ]
        )

        portfolio_value = self.portfolio.update_value(
            current_price
        )

        print("-" * 60)

        print(f"Step            : {self.current_step}")

        print(f"BTC Price       : {current_price:.2f}")

        print(f"Cash Balance    : {self.portfolio.balance:.2f}")

        print(f"BTC Held        : {self.portfolio.btc_held:.6f}")

        print(f"Portfolio Value : {portfolio_value:.2f}")

        print("-" * 60)
