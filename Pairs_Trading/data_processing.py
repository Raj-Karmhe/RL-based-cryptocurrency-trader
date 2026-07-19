import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller

# ==============================================================================
# DATA PROCESSING & FEATURE ENGINEERING FOR PAIRS TRADING (ENHANCED)
# Computes spread-based, per-asset, cross-asset, KDE, OU, and multi-TF features.
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ──────────────────────────────────────────────────────────────────────────────
# SPREAD CONSTRUCTION & OU ADAPTIVE WINDOW
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_hedge_ratio(log_a: pd.Series, log_b: pd.Series,
                                 window: int = None) -> pd.Series:
    if window is None:
        window = config.HEDGE_RATIO_WINDOW

    hedge_ratios = pd.Series(index=log_a.index, dtype=np.float64)

    for i in range(window, len(log_a)):
        la = log_a.iloc[i - window:i].values
        lb = log_b.iloc[i - window:i].values

        cov = np.cov(la, lb)
        beta = cov[0, 1] / (cov[1, 1] + 1e-12)
        hedge_ratios.iloc[i] = beta

    # Let NaNs remain, they will be dropped during the warmup phase
    return hedge_ratios


def compute_spread(log_a: pd.Series, log_b: pd.Series,
                    hedge_ratio: pd.Series) -> pd.Series:
    return log_a - hedge_ratio * log_b


def compute_half_life_rolling(spread: pd.Series, window: int = None) -> pd.Series:
    if window is None:
        window = config.HALF_LIFE_WINDOW

    half_lives = pd.Series(index=spread.index, dtype=np.float64)

    for i in range(window, len(spread)):
        s = spread.iloc[i - window:i].values
        s_lag = s[:-1]
        s_diff = np.diff(s)

        if np.std(s_lag) < 1e-10:
            half_lives.iloc[i] = config.HALF_LIFE_WINDOW
            continue

        theta = np.sum(s_diff * s_lag) / (np.sum(s_lag ** 2) + 1e-12)

        if theta >= 0:
            half_lives.iloc[i] = config.HALF_LIFE_WINDOW
        else:
            hl = -np.log(2) / theta
            half_lives.iloc[i] = np.clip(hl, 1.0, config.HALF_LIFE_WINDOW * 2)

    half_lives = half_lives.fillna(config.HALF_LIFE_WINDOW)
    return half_lives


def compute_adaptive_window(half_lives: pd.Series) -> pd.Series:
    """Calculates adaptive window = 2 * half_life bounded by MIN/MAX constants."""
    windows = (half_lives * 2).astype(int)
    return windows.clip(lower=config.MIN_ADAPTIVE_WINDOW, upper=config.MAX_ADAPTIVE_WINDOW)


def compute_rolling_kde_cdf(spread: pd.Series, adaptive_windows: pd.Series) -> pd.Series:
    """Computes Gaussian KDE CDF probability of the current spread."""
    print("  Calculating KDE CDF (this may take a minute)...")
    cdfs = pd.Series(index=spread.index, dtype=np.float64)
    spread_vals = spread.values
    win_vals = adaptive_windows.values
    
    for i in range(len(spread)):
        w = int(win_vals[i])
        if i < w:
            cdfs.iloc[i] = 0.5
            continue
            
        window_data = spread_vals[i-w:i]
        curr_val = spread_vals[i]
        
        if np.std(window_data) < 1e-6:
            cdfs.iloc[i] = 0.5
            continue
            
        try:
            kde = gaussian_kde(window_data)
            cdf = kde.integrate_box_1d(-np.inf, curr_val)
            cdfs.iloc[i] = cdf
        except:
            cdfs.iloc[i] = 0.5
            
    return cdfs.fillna(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def rolling_adf_pvalue(spread: pd.Series, window: int) -> pd.Series:
    pvalues = pd.Series(index=spread.index, dtype=np.float64)
    vals = spread.values
    for i in range(window, len(vals)):
        if np.std(vals[i-window:i]) < 1e-6:
            pvalues.iloc[i] = 1.0
            continue
        try:
            res = adfuller(vals[i-window:i], maxlag=1, autolag=None)
            pvalues.iloc[i] = res[1]
        except:
            pvalues.iloc[i] = 1.0
    return pvalues.fillna(1.0)


def add_multi_tf_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    print("  Calculating multi-timeframe features (4h, 1d)...")
    # Downsample with right label/closed to prevent lookahead bias (data leakage)
    df_4h = df_merged.resample('4h', label='right', closed='right').agg({'Close_A': 'last', 'Close_B': 'last'}).dropna()
    df_1d = df_merged.resample('1d', label='right', closed='right').agg({'Close_A': 'last', 'Close_B': 'last'}).dropna()
    
    # 4h features
    log_a_4h = np.log(df_4h['Close_A'])
    log_b_4h = np.log(df_4h['Close_B'])
    hr_4h = compute_rolling_hedge_ratio(log_a_4h, log_b_4h, window=90)
    spread_4h = compute_spread(log_a_4h, log_b_4h, hr_4h)
    z_4h = (spread_4h - spread_4h.rolling(90).mean()) / (spread_4h.rolling(90).std() + 1e-10)
    pval_4h = rolling_adf_pvalue(spread_4h, 90)
    
    # 1d features
    log_a_1d = np.log(df_1d['Close_A'])
    log_b_1d = np.log(df_1d['Close_B'])
    hr_1d = compute_rolling_hedge_ratio(log_a_1d, log_b_1d, window=30)
    spread_1d = compute_spread(log_a_1d, log_b_1d, hr_1d)
    z_1d = (spread_1d - spread_1d.rolling(30).mean()) / (spread_1d.rolling(30).std() + 1e-10)
    pval_1d = rolling_adf_pvalue(spread_1d, 30)
    
    # Reindex back to 1h
    df_tf = pd.DataFrame(index=df_merged.index)
    df_tf['Hedge_Ratio_4h'] = hr_4h.reindex(df_tf.index).ffill()
    df_tf['Spread_ZScore_4h'] = z_4h.reindex(df_tf.index).ffill()
    df_tf['Cointegration_P_Value_4h'] = pval_4h.reindex(df_tf.index).ffill()
    
    df_tf['Hedge_Ratio_1d'] = hr_1d.reindex(df_tf.index).ffill()
    df_tf['Spread_ZScore_1d'] = z_1d.reindex(df_tf.index).ffill()
    df_tf['Cointegration_P_Value_1d'] = pval_1d.reindex(df_tf.index).ffill()
    
    return df_tf


# ──────────────────────────────────────────────────────────────────────────────
# PER-ASSET FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def add_asset_features(df_merged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange

    close = df_merged[f'Close{suffix}']
    high = df_merged[f'High{suffix}']
    low = df_merged[f'Low{suffix}']
    volume = df_merged[f'Volume{suffix}']
    
    df = pd.DataFrame(index=close.index)

    df[f'Log_Return{suffix}'] = np.log(close / close.shift(1))
    df[f'RSI{suffix}'] = RSIIndicator(close=close, window=config.RSI_PERIOD).rsi()
    
    atr = AverageTrueRange(high=high, low=low, close=close, window=config.ATR_PERIOD).average_true_range()
    df[f'ATR_Pct{suffix}'] = atr / close
    df[f'ATR{suffix}'] = atr

    vol_sma = volume.rolling(20).mean()
    df[f'Volume_Ratio{suffix}'] = volume / (vol_sma + 1e-8)

    log_ret = np.log(close / close.shift(1))
    df[f'Volatility_20d{suffix}'] = log_ret.rolling(config.VOLATILITY_WINDOW).std() * np.sqrt(8760)

    bb = BollingerBands(close=close, window=config.BB_PERIOD, window_dev=config.BB_STD)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    df[f'BB_Position{suffix}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SPREAD FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def add_spread_features(spread: pd.Series, hedge_ratio: pd.Series) -> pd.DataFrame:
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    from ta.trend import MACD

    df = pd.DataFrame(index=spread.index)
    window = config.SPREAD_ZSCORE_WINDOW

    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    df['Spread_ZScore'] = (spread - spread_mean) / (spread_std + 1e-10)

    df['Spread_ZScore_Velocity'] = df['Spread_ZScore'].diff()
    df['Spread_ZScore_Accel'] = df['Spread_ZScore_Velocity'].diff()

    # Half-life and adaptive window
    half_lives = compute_half_life_rolling(spread)
    adaptive_windows = compute_adaptive_window(half_lives)
    
    df['Half_Life'] = half_lives / config.HALF_LIFE_WINDOW
    df['Hedge_Ratio_Change'] = hedge_ratio.pct_change(periods=24)
    
    spread_returns = spread.diff()
    df['Spread_Volatility'] = spread_returns.rolling(window).std()

    spread_shifted = spread - spread.min() + 1
    df['Spread_RSI'] = RSIIndicator(close=spread_shifted, window=config.RSI_PERIOD).rsi()

    bb = BollingerBands(close=spread, window=config.BB_PERIOD, window_dev=config.BB_STD)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    df['Spread_BB_Position'] = (spread - bb_lower) / (bb_upper - bb_lower + 1e-8)

    macd = MACD(close=spread_shifted, window_slow=config.MACD_SLOW, window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
    df['Spread_MACD'] = macd.macd()
    df['Spread_MACD_Signal'] = macd.macd_signal()
    
    # KDE CDF based on OU adaptive window
    df['Spread_CDF_KDE'] = compute_rolling_kde_cdf(spread, adaptive_windows)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-ASSET FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def add_cross_features(close_a: pd.Series, close_b: pd.Series,
                        volume_a: pd.Series, volume_b: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=close_a.index)
    ret_a = np.log(close_a / close_a.shift(1))
    ret_b = np.log(close_b / close_b.shift(1))
    df['Return_Diff'] = ret_a - ret_b
    vol_a = ret_a.rolling(config.VOLATILITY_WINDOW).std()
    vol_b = ret_b.rolling(config.VOLATILITY_WINDOW).std()
    df['Volatility_Ratio'] = vol_a / (vol_b + 1e-10)
    df['Volume_Corr'] = volume_a.rolling(72).corr(volume_b)
    price_ratio = close_a / close_b
    ratio_mean = price_ratio.rolling(config.SPREAD_ZSCORE_WINDOW).mean()
    ratio_std = price_ratio.rolling(config.SPREAD_ZSCORE_WINDOW).std()
    df['Price_Ratio_ZScore'] = (price_ratio - ratio_mean) / (ratio_std + 1e-10)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# TURBULENCE INDEX
# ──────────────────────────────────────────────────────────────────────────────

def compute_turbulence(close_a: pd.Series, close_b: pd.Series,
                        lookback: int = None) -> pd.Series:
    if lookback is None:
        lookback = config.TURBULENCE_LOOKBACK
    ret_a = np.log(close_a / close_a.shift(1)).fillna(0).values
    ret_b = np.log(close_b / close_b.shift(1)).fillna(0).values
    turb = np.zeros(len(ret_a))
    for i in range(lookback, len(ret_a)):
        hist_a = ret_a[i - lookback:i]
        hist_b = ret_b[i - lookback:i]
        mu = np.array([hist_a.mean(), hist_b.mean()])
        cov = np.cov(hist_a, hist_b)
        y = np.array([ret_a[i], ret_b[i]])
        diff = y - mu
        try:
            cov_inv = np.linalg.inv(cov + np.eye(2) * 1e-10)
            turb[i] = float(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            turb[i] = 0.0
    return pd.Series(turb, index=close_a.index)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def process_and_split_data():
    from download_data import download_and_align_pair

    print("\n" + "=" * 70)
    print("  DATA PROCESSING PIPELINE")
    print("=" * 70)

    df_a, df_b, df_merged = download_and_align_pair()

    close_a = df_merged['Close_A']
    close_b = df_merged['Close_B']
    log_a = np.log(close_a)
    log_b = np.log(close_b)

    print("\nComputing rolling hedge ratio and spread...")
    hedge_ratio = compute_rolling_hedge_ratio(log_a, log_b)
    spread = compute_spread(log_a, log_b, hedge_ratio)

    print("\nComputing features...")
    print("  -> Asset A features...")
    feat_a = add_asset_features(df_merged, '_A')
    
    print("  -> Asset B features...")
    feat_b = add_asset_features(df_merged, '_B')

    print("  -> Spread & KDE features...")
    feat_spread = add_spread_features(spread, hedge_ratio)

    print("  -> Cross-asset features...")
    feat_cross = add_cross_features(close_a, close_b, df_merged['Volume_A'], df_merged['Volume_B'])

    print("  -> Multi-timeframe features...")
    feat_tf = add_multi_tf_features(df_merged)

    print("  -> Turbulence index...")
    turbulence = compute_turbulence(close_a, close_b)

    print("\nMerging all features...")
    df_full = pd.concat([
        df_merged,
        feat_a,
        feat_b,
        feat_spread,
        feat_cross,
        feat_tf,
    ], axis=1)

    df_full['Spread'] = spread
    df_full['Hedge_Ratio'] = hedge_ratio
    df_full['Turbulence'] = turbulence

    n_before = len(df_full)
    df_full = df_full.dropna(subset=config.FEATURE_COLUMNS)
    n_after = len(df_full)
    print(f"\n  Dropped {n_before - n_after} NaN warmup rows ({n_after} remaining)")

    n = len(df_full)
    train_end = int(config.TRAIN_RATIO * n)
    val_end = int((config.TRAIN_RATIO + config.VAL_RATIO) * n)

    df_train = df_full.iloc[:train_end].copy()
    df_val = df_full.iloc[train_end:val_end].copy()
    df_test = df_full.iloc[val_end:].copy()

    print(f"\n  Data Split (chronological, no overlap):")
    print(f"    Train:      {len(df_train):>8,} periods")
    print(f"    Validation: {len(df_val):>8,} periods")
    print(f"    Test:       {len(df_test):>8,} periods")

    train_turb = df_train['Turbulence']
    config.TURBULENCE_THRESHOLD = float(np.nanpercentile(
        train_turb[train_turb > 0], config.TURBULENCE_PERCENTILE
    ))
    
    print("\nScaling features...")
    if os.path.exists(config.SCALER_PATH) and not config.FORCE_RETRAIN:
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler()
        scaler.fit(df_train[config.FEATURE_COLUMNS])
        with open(config.SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)

    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    df_train_scaled[config.FEATURE_COLUMNS] = scaler.transform(df_train[config.FEATURE_COLUMNS])
    df_val_scaled[config.FEATURE_COLUMNS] = scaler.transform(df_val[config.FEATURE_COLUMNS])
    df_test_scaled[config.FEATURE_COLUMNS] = scaler.transform(df_test[config.FEATURE_COLUMNS])

    for df in [df_train_scaled, df_val_scaled, df_test_scaled]:
        df[config.FEATURE_COLUMNS] = df[config.FEATURE_COLUMNS].fillna(0)

    print(f"\nSUCCESS: Data processing pipeline complete!")
    return (df_train_scaled, df_val_scaled, df_test_scaled, df_train, df_val, df_test)


if __name__ == '__main__':
    train_s, val_s, test_s, train_r, val_r, test_r = process_and_split_data()
