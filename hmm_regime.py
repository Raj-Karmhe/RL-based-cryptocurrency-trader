"""
hmm_regime.py

Train Hidden Markov Model
to detect market regimes.
"""

# -----------------------------------
# Imports
# -----------------------------------

from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM



# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("processed_data.csv")

# ---------------------------------------------------
# Feature Engineering for HMM
# ---------------------------------------------------

# Daily Return
df["Return"] = df["Close"].pct_change()

# Rolling Volatility
df["Volatility"] = df["Return"].rolling(20).std()

# ATR already exists
# Normalize it relative to price
df["ATR_Normalized"] = df["ATR"] / df["Close"]

# Volume Change
df["Volume_Change"] = df["Volume"].pct_change()

# Remove NaNs
df = df.dropna().reset_index(drop=True)

# -----------------------------------
# Train Hidden Markov Model
# -----------------------------------

# Features used by HMM
features = df[
    [
        "Return",
        "Volatility",
        "ATR_Normalized",
        "Volume_Change"
    ]
].copy()

# Standardize features
scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)

# Create HMM
model = GaussianHMM(

    n_components=4,

    covariance_type="diag",

    n_iter=500,

    random_state=42

)

print("\nTraining HMM...")

# Train model
model.fit(features_scaled)

print("Training Complete!")

# Predict Hidden States
hidden_states = model.predict(features_scaled)

# Store in dataframe
df["Regime"] = hidden_states

# ----------------------------------------------------
# One-Hot Encode Regimes
# ----------------------------------------------------

regime_dummies = pd.get_dummies(
    df["Regime"],
    prefix="Regime",
    dtype=int
).reindex(
    columns=["Regime_0", "Regime_1", "Regime_2", "Regime_3"],
    fill_value=0
)

df = pd.concat(
    [df, regime_dummies],
    axis=1
)

print()

print(df[
    [
        "Regime",
        "Regime_0",
        "Regime_1",
        "Regime_2",
        "Regime_3"
    ]
].head(10))

# -----------------------------------
# Analyze Regimes
# -----------------------------------

print("\nNumber of days in each regime:\n")

print(df["Regime"].value_counts().sort_index())

print("\n")

for regime in sorted(df["Regime"].unique()):

    regime_data = df[df["Regime"] == regime]

    print("=" * 50)

    print(f"Regime {regime}")

    print(f"Days              : {len(regime_data)}")

    print(f"Average Return    : {regime_data['Return'].mean():.5f}")

    print(f"Average Volatility: {regime_data['Volatility'].mean():.5f}")

    print("=" * 50)

print()

# -----------------------------------
# Visualize Market Regimes
# -----------------------------------

colors = {

    0:"#4CAF50",

    1:"#F44336",

    2:"#2196F3",

    3:"#FFC107"

}

plt.figure(figsize=(18, 8))

plt.plot(
    df.index,
    df["Close"],
    color="black",
    linewidth=1.3,
    label="BTC Price"
)

for regime in sorted(df["Regime"].unique()):

    mask = df["Regime"] == regime

    plt.scatter(

        df.index[mask],

        df["Close"][mask],

        s=12,

        color=colors[regime],

        label=f"Regime {regime}"

    )

plt.title("Bitcoin Price with Hidden Market Regimes")

plt.xlabel("Trading Days")

plt.ylabel("BTC Price")

plt.legend()

plt.grid(alpha=0.3)

plt.show()

# ----------------------------------------------------
# Save
# ----------------------------------------------------

df.to_csv(

    "processed_data_with_regime.csv",

    index=False

)

print()

print("="*60)

print("Dataset saved successfully!")

print("File : processed_data_with_regime.csv")

print("="*60)

print("\nTransition Matrix\n")

print(model.transmat_)

print("\nState Means\n")

print(model.means_)