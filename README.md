# Hidden Markov Model (HMM) for Market Regime Detection

## Overview

Financial markets continuously transition between different market conditions such as bullish trends, bearish trends, highly volatile periods, and sideways movements. These market conditions are not explicitly available in historical price data, making them **latent (hidden)** variables.

To capture these hidden dynamics, this project integrates a **Hidden Markov Model (HMM)** before the Reinforcement Learning (RL) agent. The HMM identifies statistically similar market behaviors and classifies each trading period into one of several hidden market states. These inferred states are then incorporated into the observation space of the PPO agent, allowing it to learn trading policies that adapt to different market conditions.

Unlike supervised classification algorithms, the HMM performs **unsupervised learning**, discovering hidden market structures directly from historical data without requiring manually labeled bull or bear markets.

---

# Why Hidden Markov Model?

Traditional Reinforcement Learning agents rely only on raw market information such as:

- Open
- High
- Low
- Close
- Volume
- Technical Indicators

Although these features contain valuable information, they do not explicitly describe the underlying market condition.

For example, an RSI value of **70** may indicate different trading opportunities depending on whether the market is:

- Strongly Bullish
- Highly Volatile
- Bearish
- Sideways

Instead of forcing the RL agent to infer these market conditions by itself, an HMM is introduced to identify the **hidden market regime** and provide this additional context to the trading agent.

---

# What is a Hidden Markov Model?

A Hidden Markov Model (HMM) is a probabilistic model designed for sequential data where the underlying system cannot be directly observed.

An HMM consists of two components:

## Observed Variables

These are directly available from historical market data.

In this project they include:

- Daily Returns
- Rolling Volatility
- Average True Range (ATR)
- Volume Change

## Hidden Variables

These represent the true market condition.

Examples include:

- Bull Market
- Bear Market
- Sideways Market
- High Volatility Market

These market conditions are **never explicitly provided** within the dataset.

The objective of the HMM is to infer these hidden market states from the observable market statistics.

---

# Why are Market Regimes Hidden?

The dataset only contains numerical information such as:

- Price
- Volume
- Returns
- Technical Indicators

There is no column stating

```
Bull Market
```

or

```
Bear Market
```

Therefore, market conditions must be inferred indirectly.

This makes Hidden Markov Models particularly suitable because they estimate hidden states from observable sequential data.

---

# Markov Assumption

The Hidden Markov Model is based on the **Markov Property**, which states:

> The probability of the current hidden state depends only on the previous hidden state and not on the complete historical sequence.

Mathematically,

\[
P(S_t|S_{t-1},S_{t-2},...,S_1)=P(S_t|S_{t-1})
\]

where

- \(S_t\) denotes the hidden state at time \(t\).

This assumption allows the model to efficiently capture transitions between different market regimes.

---

# Feature Engineering

The HMM is trained using four carefully selected statistical features.

These features describe market direction, uncertainty, momentum, and trading activity.

---

## 1. Daily Return

\[
Return_t=\frac{Close_t-Close_{t-1}}{Close_{t-1}}
\]

### Purpose

- Captures market direction
- Positive values indicate upward movement
- Negative values indicate downward movement

---

## 2. Rolling Volatility

Rolling standard deviation of daily returns over a fixed window.

### Purpose

- Measures market uncertainty
- Identifies stable and unstable market periods

---

## 3. Normalized Average True Range (ATR)

\[
ATR_{normalized}=\frac{ATR}{Close}
\]

### Purpose

- Measures relative price movement
- Removes dependence on absolute price

---

## 4. Percentage Change in Volume

\[
VolumeChange=\frac{Volume_t-Volume_{t-1}}{Volume_{t-1}}
\]

### Purpose

- Captures changes in market participation
- Identifies unusually active trading periods

---

# Feature Standardization

The selected features exist on completely different numerical scales.

For example,

| Feature | Typical Value |
|----------|---------------|
| Return | 0.01 |
| ATR | 250 |
| Volume Change | 0.20 |

Without preprocessing, ATR would dominate the learning process.

Therefore, all features are standardized using **StandardScaler**:

\[
x'=\frac{x-\mu}{\sigma}
\]

where

- \(\mu\) is the feature mean
- \(\sigma\) is the feature standard deviation

This ensures every feature contributes equally during HMM training.

---

# Gaussian Hidden Markov Model

This implementation uses a **Gaussian Hidden Markov Model**.

Each hidden state is modeled as a multivariate Gaussian distribution.

Rather than manually defining market regimes, the model automatically discovers groups of statistically similar observations.

For every hidden state, the HMM learns:

- Mean Vector
- Covariance Matrix
- Transition Probabilities

The implementation uses

```python
covariance_type = "diag"
```

Diagonal covariance assumes conditional independence among features inside each hidden state.

This significantly improves numerical stability while reducing computational complexity.

---

# Number of Hidden States

The HMM is configured with

```python
n_components = 4
```

This means the model automatically discovers **four hidden market states**.

It is important to note that these states are **not predefined**.

The model is **never told** what constitutes

- Bull Market
- Bear Market
- Sideways Market

Instead, it groups observations according to their statistical similarity.

---

# Hidden States vs Market Regimes

A common misconception is that the HMM directly classifies markets into

- Bullish
- Bearish
- Sideways

This is **incorrect**.

The HMM only predicts hidden state labels.

For example,

```
State 0
State 1
State 2
State 3
```

These labels have **no intrinsic meaning**.

After training, we analyze the statistical characteristics of each state.

Example:

| Hidden State | Avg Return | Avg Volatility | Possible Interpretation |
|--------------|------------|----------------|--------------------------|
| State 0 | Positive | Low | Bullish Low Volatility |
| State 1 | Negative | High | Bearish High Volatility |
| State 2 | Near Zero | Low | Sideways Market |
| State 3 | Positive | High | Volatile Bull Market |

These names are assigned **after training** based on observed statistics.

---

# How Does the HMM Classify Regimes?

Every trading day is represented as a four-dimensional feature vector:

```
[
    Daily Return,
    Rolling Volatility,
    ATR / Close,
    Volume Change
]
```

The Gaussian HMM models each hidden state as a probability distribution in this feature space.

During training, the model identifies groups of observations with similar statistical characteristics.

For example,

| Return | Volatility | ATR | Volume |
|---------|------------|------|---------|
| +1.2% | Low | Low | Normal |
| +0.9% | Low | Low | High |
| +1.5% | Low | Low | Normal |

may be grouped into one hidden state.

Similarly,

| Return | Volatility | ATR | Volume |
|---------|------------|------|---------|
| -2.8% | High | High | High |
| -1.9% | High | High | High |

may be assigned to another hidden state.

The classification is therefore based entirely on **statistical similarity**, not manually defined labels.

---

# Hidden State Prediction

Once trained, the HMM predicts a hidden state for every trading day.

Example:

| Day | Hidden State |
|-----|--------------|
| Day 1 | State 0 |
| Day 2 | State 0 |
| Day 3 | State 2 |
| Day 4 | State 3 |
| Day 5 | State 1 |

This produces a sequential description of changing market conditions.

---

# One-Hot Encoding

The hidden states are categorical variables.

Using integer labels directly would incorrectly imply

```
State 3 > State 2
```

which has no real interpretation.

Therefore, every hidden state is converted into one-hot encoded features.

Example:

| Hidden State | Regime_0 | Regime_1 | Regime_2 | Regime_3 |
|--------------|----------|----------|----------|----------|
| State 0 | 1 | 0 | 0 | 0 |
| State 2 | 0 | 0 | 1 | 0 |
| State 1 | 0 | 1 | 0 | 0 |

These binary regime features are appended to the processed dataset.

---

# Integration into the Reinforcement Learning Pipeline

Before HMM integration, the PPO agent received observations containing

- OHLCV Data
- Technical Indicators

After HMM integration, the observation vector becomes

- OHLCV Data
- Technical Indicators
- Regime_0
- Regime_1
- Regime_2
- Regime_3

The PPO agent therefore receives explicit information about the inferred market regime in addition to traditional technical indicators.

Instead of independently discovering market conditions, the RL agent learns **regime-dependent trading policies**.

For example,

- Increase exposure during one regime.
- Reduce exposure during another regime.
- Hold cash during highly volatile conditions.

These policies are learned entirely through reinforcement learning.

---

# Advantages of HMM Integration

The integration of HMM provides several advantages:

- Identifies hidden market structures without requiring labeled data.
- Captures sequential market behavior through transition probabilities.
- Provides additional contextual information to the RL agent.
- Improves market state representation.
- Enables regime-specific trading strategies.
- Enhances decision-making under varying market conditions.

---

# Complete HMM Pipeline

```text
Historical BTC OHLCV Data
          │
          ▼
Technical Indicator Generation
          │
          ▼
Feature Engineering
(Return, Volatility, ATR, Volume Change)
          │
          ▼
Feature Standardization
(StandardScaler)
          │
          ▼
Gaussian Hidden Markov Model
          │
          ▼
Hidden State Prediction
(State 0, State 1, State 2, State 3)
          │
          ▼
One-Hot Encoding
(Regime_0, Regime_1, Regime_2, Regime_3)
          │
          ▼
Augmented Dataset
          │
          ▼
Trading Environment
          │
          ▼
PPO Reinforcement Learning Agent
          │
          ▼
Adaptive Trading Decisions
```
---

# Experimental Observations

The HMM module was integrated into the reinforcement learning pipeline as an additional feature extraction stage.

Two experimental configurations were evaluated.

## Configuration 1

- Technical Indicators
- PPO Trading Agent

## Configuration 2

- Technical Indicators
- Hidden Markov Model
- PPO Trading Agent

The HMM successfully identified statistically distinct market regimes and augmented the observation space with four additional regime features.

While the HMM consistently produced meaningful regime segmentation, the improvement in overall trading profitability was **not consistent across all experiments**. During different debugging iterations and environment modifications, overall returns varied, indicating that the final trading performance depends not only on regime detection but also on several interconnected factors such as:

- Reward function design
- Portfolio management strategy
- Action execution logic
- Transaction cost modeling
- PPO hyperparameter tuning
- Market data characteristics

These observations suggest that market regime detection alone is insufficient to guarantee higher returns, but it provides valuable contextual information that can improve the learning capability of downstream reinforcement learning models when combined with appropriate environment design and training procedures.

---

# Limitations

The current implementation uses four statistical features:

- Daily Return
- Rolling Volatility
- ATR / Close
- Volume Change

Although these features capture short-term market behavior, they do not explicitly model long-term trends.

Future improvements may include additional features such as:

- RSI
- MACD
- ADX
- Moving Average Slope
- Bollinger Band Width
- Momentum Indicators

to produce richer and more informative market regimes.

---

# Conclusion

The Hidden Markov Model serves as an unsupervised market regime detection module that enriches the state representation used by the PPO trading agent. Rather than predicting future prices directly, the HMM discovers latent statistical structures within historical market data and provides regime information as additional context. These inferred regimes enable the PPO agent to learn adaptive trading policies that respond to varying market conditions, resulting in a more informed, context-aware, and robust reinforcement learning framework.
