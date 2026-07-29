"""
phase1_feature_engineering.py - Technical Indicator Generator and Stationarity Enforcer

This script computes a comprehensive set of technical indicators on the multi-timeframe OHLCV data.
It enforces stationarity for trending features using percentage distances and rolling z-scores,
splits features, scales them using a StandardScaler fitted only on the training split, and saves them.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator, IchimokuIndicator, CCIIndicator, PSARIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator, MFIIndicator
except ImportError:
    raise ImportError("Please install the 'ta' library to compute technical indicators: pip install ta")

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def get_rolling_zscore(series: pd.Series, window: int = config.Z_SCORE_WINDOW) -> pd.Series:
    """
    Computes a rolling z-score of a series to make it mean-reverting and stationary:
    (value - rolling_mean) / (rolling_std + epsilon)
    """
    rolling_mean = series.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window=window, min_periods=window // 2).std()
    return (series - rolling_mean) / (rolling_std + 1e-8)

def get_pct_distance(series: pd.Series, baseline: pd.Series) -> pd.Series:
    """
    Calculates the percentage deviation of a series from a baseline series:
    (series - baseline) / (baseline + epsilon)
    """
    return (series - baseline) / (baseline.abs() + 1e-8)

def compute_supertrend_custom(high: pd.Series, low: pd.Series, close: pd.Series, period=10, multiplier=3.0) -> pd.Series:
    """
    Computes the Supertrend direction (+1 for bullish, -1 for bearish).
    """
    atr = AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
    hl2 = (high + low) / 2
    
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    
    close_np = close.values
    basic_upper_np = basic_upper.values
    basic_lower_np = basic_lower.values
    
    final_upper_np = np.zeros(len(close))
    final_lower_np = np.zeros(len(close))
    supertrend_dir_np = np.ones(len(close))
    supertrend_val_np = np.zeros(len(close))
    
    final_upper_np[0] = basic_upper_np[0]
    final_lower_np[0] = basic_lower_np[0]
    
    for i in range(1, len(close)):
        # Calculate final upper band
        if basic_upper_np[i] < final_upper_np[i-1] or close_np[i-1] > final_upper_np[i-1]:
            final_upper_np[i] = basic_upper_np[i]
        else:
            final_upper_np[i] = final_upper_np[i-1]
            
        # Calculate final lower band
        if basic_lower_np[i] > final_lower_np[i-1] or close_np[i-1] < final_lower_np[i-1]:
            final_lower_np[i] = basic_lower_np[i]
        else:
            final_lower_np[i] = final_lower_np[i-1]
            
    for i in range(1, len(close)):
        if close_np[i] > final_upper_np[i-1]:
            supertrend_dir_np[i] = 1.0
        elif close_np[i] < final_lower_np[i-1]:
            supertrend_dir_np[i] = -1.0
        else:
            supertrend_dir_np[i] = supertrend_dir_np[i-1]
            
        if supertrend_dir_np[i] == 1.0:
            supertrend_val_np[i] = final_lower_np[i]
        else:
            supertrend_val_np[i] = final_upper_np[i]
            
    supertrend_dir = pd.Series(supertrend_dir_np, index=close.index)
    supertrend_val = pd.Series(supertrend_val_np, index=close.index)
    
    # Bug #13 fix: return both direction and value (price level) for distance feature
    return supertrend_dir, supertrend_val

def generate_timeframe_features(df_tf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Generates all base and stationary indicators for a single timeframe's raw data.
    """
    high = df_tf["High"]
    low = df_tf["Low"]
    close = df_tf["Close"]
    open_p = df_tf["Open"]
    vol = df_tf["Volume"]
    
    features = pd.DataFrame(index=df_tf.index)
    
    # 1. Base Features
    features[f"{prefix}log_return"] = np.log(close / close.shift(1))
    features[f"{prefix}return_5"] = close.pct_change(5)
    features[f"{prefix}return_20"] = close.pct_change(20)
    features[f"{prefix}range_pct"] = (high - low) / (close + 1e-8)
    features[f"{prefix}volume_log"] = np.log(vol + 1.0)
    
    # 2. Trend Indicators
    macd_ind = MACD(close=close, window_fast=config.MACD_FAST, window_slow=config.MACD_SLOW, window_sign=config.MACD_SIGNAL)
    features[f"{prefix}macd"] = macd_ind.macd()
    features[f"{prefix}macd_sig"] = macd_ind.macd_signal()
    features[f"{prefix}macd_diff"] = macd_ind.macd_diff()
    
    psar_ind = PSARIndicator(high=high, low=low, close=close, step=config.SAR_STEP, max_step=config.SAR_MAX)
    sar_vals = psar_ind.psar_up().combine_first(psar_ind.psar_down())
    features[f"{prefix}sar_distance"] = get_pct_distance(close, sar_vals)
    
    adx_ind = ADXIndicator(high=high, low=low, close=close, window=config.ADX_PERIOD)
    features[f"{prefix}adx"] = adx_ind.adx()
    features[f"{prefix}adx_pos"] = adx_ind.adx_pos()
    features[f"{prefix}adx_neg"] = adx_ind.adx_neg()
    features[f"{prefix}adx_diff"] = features[f"{prefix}adx_pos"] - features[f"{prefix}adx_neg"]
    
    supertrend_dir, supertrend_val = compute_supertrend_custom(high, low, close, config.SUPERTREND_PERIOD, config.SUPERTREND_MULT)
    features[f"{prefix}supertrend_direction"] = supertrend_dir
    # Bug #13 fix: add supertrend distance as a valuable trend-strength feature
    features[f"{prefix}supertrend_distance"] = get_pct_distance(close, supertrend_val)
    
    for sma_p in config.SMA_PERIODS:
        sma_val = SMAIndicator(close=close, window=sma_p).sma_indicator()
        features[f"{prefix}sma_{sma_p}_distance"] = get_pct_distance(close, sma_val)
        
    for ema_p in config.EMA_PERIODS:
        ema_val = EMAIndicator(close=close, window=ema_p).ema_indicator()
        features[f"{prefix}ema_{ema_p}_distance"] = get_pct_distance(close, ema_val)
        
    ichimoku = IchimokuIndicator(high=high, low=low)
    features[f"{prefix}ichimoku_tenkan_distance"] = get_pct_distance(close, ichimoku.ichimoku_conversion_line())
    features[f"{prefix}ichimoku_kijun_distance"] = get_pct_distance(close, ichimoku.ichimoku_base_line())
    # NOTE: ichimoku_a() and ichimoku_b() (Senkou Span A/B) are intentionally excluded.
    # The ta library forward-shifts these by 26 periods for charting, which introduces
    # lookahead bias when used as input features for a causal model.
    
    # 3. Momentum Indicators
    features[f"{prefix}rsi"] = RSIIndicator(close=close, window=config.RSI_PERIOD).rsi()
    
    stoch = StochasticOscillator(high=high, low=low, close=close, window=config.STOCH_K, smooth_window=config.STOCH_D)
    features[f"{prefix}stoch_k"] = stoch.stoch()
    features[f"{prefix}stoch_d"] = stoch.stoch_signal()
    
    cci = CCIIndicator(high=high, low=low, close=close, window=config.CCI_PERIOD).cci()
    features[f"{prefix}cci_zscore"] = get_rolling_zscore(cci)
    
    ao = AwesomeOscillatorIndicator(high=high, low=low, window1=config.AO_SHORT, window2=config.AO_LONG).awesome_oscillator()
    features[f"{prefix}ao_zscore"] = get_rolling_zscore(ao)
    
    features[f"{prefix}mfi"] = MFIIndicator(high=high, low=low, close=close, volume=vol, window=config.MFI_PERIOD).money_flow_index()
    
    # 4. Volatility Indicators
    bb = BollingerBands(close=close, window=config.BB_PERIOD, window_dev=config.BB_STD)
    features[f"{prefix}bb_pct"] = bb.bollinger_pband()
    features[f"{prefix}bb_bandwidth_zscore"] = get_rolling_zscore((bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-8))
    
    atr = AverageTrueRange(high=high, low=low, close=close, window=config.ATR_PERIOD).average_true_range()
    features[f"{prefix}atr_pct"] = atr / (close + 1e-8)
    features[f"{prefix}historical_volatility_20"] = features[f"{prefix}log_return"].rolling(20).std()
    
    kc = KeltnerChannel(high=high, low=low, close=close, window=config.KELTNER_PERIOD, window_atr=config.KELTNER_PERIOD, multiplier=config.KELTNER_ATR_MULT)
    kc_high, kc_low, kc_mid = kc.keltner_channel_hband(), kc.keltner_channel_lband(), kc.keltner_channel_mband()
    features[f"{prefix}keltner_pct"] = (close - kc_low) / (kc_high - kc_low + 1e-8)
    features[f"{prefix}keltner_bandwidth_zscore"] = get_rolling_zscore((kc_high - kc_low) / (kc_mid + 1e-8))
    
    dc = DonchianChannel(high=high, low=low, close=close, window=config.DONCHIAN_PERIOD)
    dc_high, dc_low = dc.donchian_channel_hband(), dc.donchian_channel_lband()
    features[f"{prefix}donchian_pct"] = (close - dc_low) / (dc_high - dc_low + 1e-8)
    features[f"{prefix}donchian_bandwidth_zscore"] = get_rolling_zscore((dc_high - dc_low) / (close + 1e-8))
    
    # 5. Volume Indicators
    obv = OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    features[f"{prefix}obv_zscore"] = get_rolling_zscore(obv)
    
    adl = AccDistIndexIndicator(high=high, low=low, close=close, volume=vol).acc_dist_index()
    features[f"{prefix}adl_zscore"] = get_rolling_zscore(adl)
    
    features[f"{prefix}cmf"] = ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=vol, window=20).chaikin_money_flow()
    
    typical_price = (high + low + close) / 3.0
    tp_volume = typical_price * vol
    vwap = tp_volume.rolling(config.VWAP_PERIOD).sum() / (vol.rolling(config.VWAP_PERIOD).sum() + 1e-8)
    features[f"{prefix}vwap_distance"] = get_pct_distance(close, vwap)
    
    features[f"{prefix}volume_ratio"] = vol / (vol.rolling(20).mean() + 1e-8)
    
    # Keep only light features if configured
    if config.INDICATOR_SET == 10:
        light_cols = [
            f"{prefix}log_return", f"{prefix}volume_log", f"{prefix}macd", f"{prefix}macd_sig", f"{prefix}macd_diff",
            f"{prefix}supertrend_direction", f"{prefix}sma_200_distance", f"{prefix}ema_50_distance", f"{prefix}rsi",
            f"{prefix}mfi", f"{prefix}bb_pct", f"{prefix}bb_bandwidth_zscore", f"{prefix}atr_pct", f"{prefix}cmf"
        ]
        features = features[[c for c in light_cols if c in features.columns]]
        
    return features

def process_features_for_coin(symbol_file: str) -> tuple:
    """
    Loads raw multi-timeframe data for a coin, computes indicators, merges timeframes
    correctly shifting higher timeframes to prevent lookahead bias, splits chronologically,
    and returns features.
    """
    base_tf = config.BASE_TF
    base_raw_path = os.path.join(config.DATA_DIR, f"{symbol_file}_{base_tf}_raw.csv")
    
    if not os.path.exists(base_raw_path):
        raise FileNotFoundError(f"Raw data file {base_raw_path} not found. Run phase1_data_extraction.py first.")
        
    base_df = pd.read_csv(base_raw_path, index_col="Date", parse_dates=True)
    # Ensure base index is tz-aware (UTC) to match HTF indices after localization
    if base_df.index.tzinfo is None:
        base_df.index = base_df.index.tz_localize("UTC")
    base_index = base_df.index
    
    all_timeframe_features = []
    
    for tf in config.TIMEFRAMES:
        raw_path = os.path.join(config.DATA_DIR, f"{symbol_file}_{tf}_raw.csv")
        df_raw = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
        
        # Standardize indices to UTC
        if df_raw.index.tzinfo is None:
            df_raw.index = df_raw.index.tz_localize("UTC")
            
        print(f"  Generating indicators for {tf} timeframe...")
        tf_feats = generate_timeframe_features(df_raw, f"{tf}_")
        
        if tf != base_tf:
            # Shift by 1 period before mapping to prevent lookahead bias
            tf_feats = tf_feats.shift(1)
            
        # Reindex and forward-fill to 1h base timeline
        tf_feats = tf_feats.reindex(base_index, method='ffill')
        all_timeframe_features.append(tf_feats)
        
    # Concatenate columns
    all_feats_df = pd.concat(all_timeframe_features, axis=1)
    
    # Maintain target close price and ATR (unscaled) for environment risk management
    all_feats_df["Close"] = base_df["Close"]
    all_feats_df["ATR"] = AverageTrueRange(high=base_df["High"], low=base_df["Low"], close=base_df["Close"], window=config.ATR_PERIOD).average_true_range()
    
    # Drop rows at start where indicators are still warming up (NaNs)
    clean_feats_df = all_feats_df.dropna()
    print(f"  Feature generation done. Shape: {clean_feats_df.shape}")
    
    return clean_feats_df
def scale_and_save_splits(symbol_file: str, full_df: pd.DataFrame) -> tuple:
    """
    Splits the features chronologically, fits StandardScaler only on the train set,
    and applies it to all splits. Saves scaled sets to CSV.
    """
    max_date = full_df.index.max()
    test_start = max_date - pd.Timedelta(days=180)
    val_start = test_start - pd.Timedelta(days=180)
    train_start = val_start - pd.Timedelta(days=4*365)
    
    train_feat = full_df[(full_df.index >= train_start) & (full_df.index < val_start)].copy()
    val_feat = full_df[(full_df.index >= val_start) & (full_df.index < test_start)].copy()
    test_feat = full_df[full_df.index >= test_start].copy()
    
    # Identify feature columns to scale (exclude Close and ATR)
    feature_cols = [col for col in full_df.columns if col not in ("Close", "ATR")]
    
    print("  Fitting StandardScaler on training split only...")
    scaler = StandardScaler()
    scaler.fit(train_feat[feature_cols])
    
    # Cache scaler
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    # Apply transformation to splits
    def apply_scaling(df_split):
        scaled = df_split.copy()
        scaled[feature_cols] = scaler.transform(df_split[feature_cols])
        # Safe replacements for infinites and NaNs
        scaled[feature_cols] = scaled[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return scaled
        
    train_scaled = apply_scaling(train_feat)
    val_scaled = apply_scaling(val_feat)
    test_scaled = apply_scaling(test_feat)
    
    # Save to CSV files
    train_scaled.to_csv(os.path.join(config.DATA_DIR, f"{symbol_file}_train_features.csv"))
    val_scaled.to_csv(os.path.join(config.DATA_DIR, f"{symbol_file}_val_features.csv"))
    test_scaled.to_csv(os.path.join(config.DATA_DIR, f"{symbol_file}_test_features.csv"))
    
    print(f"  Saved scaled features for {symbol_file}")
    return train_scaled, val_scaled, test_scaled, feature_cols

def run_feature_engineering_pipeline():
    """
    Main orchestration function for feature engineering.
    """
    print("\n" + "=" * 60)
    print("Starting Feature Engineering and Scaling Pipeline")
    print("=" * 60)
    
    symbols = config.MULTI_COINS if config.MULTI_COIN_MODE else [config.SYMBOL]
    primary_coin = config.SYMBOL
    
    results = None
    
    for symbol in symbols:
        print(f"\nProcessing Feature Generation for: {symbol}")
        symbol_file = symbol.replace("/", "_")
        
        full_df = process_features_for_coin(symbol_file)
        train_s, val_s, test_s, feature_cols = scale_and_save_splits(symbol_file, full_df)
        
        if symbol == primary_coin:
            # Save primary splits to general paths
            train_s.to_csv(config.TRAIN_FEAT_PATH)
            val_s.to_csv(config.VAL_FEAT_PATH)
            test_s.to_csv(config.TEST_FEAT_PATH)
            results = (train_s, val_s, test_s, feature_cols)
            
    print("\nFeature engineering and scaling pipeline completed successfully.")
    return results

if __name__ == "__main__":
    run_feature_engineering_pipeline()
