"""
phase1_feature_engineering.py — 25-50 Indicator Generation + Stationarity
==========================================================================
Phase 1, Step 2 of the pipeline.

What this script does
---------------------
1.  Takes the merged multi-timeframe DataFrame (output of phase1_data_extraction).
2.  Computes 25–50 technical indicators per timeframe using the `ta` library.
    Categories:
        Trend     : MACD, Parabolic SAR, Ichimoku, ADX, Supertrend,
                    SMA/EMA pct-distance
        Momentum  : RSI, Stochastic, CCI, Awesome Oscillator, MFI
        Volatility: Bollinger Bands, ATR%, Keltner Channels, Donchian width
        Volume    : OBV z-score, ADL z-score, CMF, VWAP pct-distance
        Base      : OHLCV (raw), returns, log returns
3.  Enforces stationarity:
        • Bounded oscillators (RSI, Stochastic %K, CCI, MFI) → kept as-is
        • Unbounded / trending series (price levels, OBV, Ichimoku lines,
          raw SMA/EMA values) → rolling z-score (50-period window)
          OR percentage distance from a moving average
4.  Fits a StandardScaler ONLY on the training split and transforms
    val/test with the SAME fitted scaler (no lookahead bias in scaling).
5.  Saves the three feature-engineered splits to CSV.

STATIONARITY RATIONALE
-----------------------
RL agents and LSTMs are extremely sensitive to non-stationary inputs.
A trending feature (e.g. raw Close price going from $10 k to $60 k) has
a very different scale in the test set than in training, causing the agent
to effectively see "unseen" input magnitudes and fail to generalise.

Rolling z-scores convert an unbounded series into a distribution centred
around 0 with unit variance, making it stationary and comparable across
all market regimes.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler

try:
    # ta library — pip install ta
    from ta.trend import (MACD, ADXIndicator, EMAIndicator, SMAIndicator,
                          IchimokuIndicator, CCIIndicator, PSARIndicator)
    from ta.momentum import (RSIIndicator, StochasticOscillator,
                             AwesomeOscillatorIndicator)
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import (OnBalanceVolumeIndicator, AccDistIndexIndicator,
                           ChaikinMoneyFlowIndicator, MFIIndicator)
except ImportError:
    raise ImportError("ta library not found. Install with: pip install ta")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — Rolling Z-Score  (key stationarity transformation)
# ──────────────────────────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = config.Z_SCORE_WINDOW
                   ) -> pd.Series:
    """
    Transforms an unbounded series into a rolling z-score:
        z_t = (x_t - μ_{t-window:t}) / σ_{t-window:t}

    This makes previously trending features (like OBV, price levels) into
    mean-reverting, unit-variance series suitable for an LSTM.
    """
    roll = series.rolling(window=window, min_periods=window // 2)
    return (series - roll.mean()) / (roll.std() + 1e-8)


def pct_distance(series: pd.Series, moving_avg: pd.Series) -> pd.Series:
    """
    Percentage distance of `series` from `moving_avg`:
        d_t = (x_t - MA_t) / MA_t
    This is stationary because it captures relative deviation, not absolute level.
    """
    return (series - moving_avg) / (moving_avg.abs() + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# CORE INDICATOR FUNCTION  (applied independently per timeframe column set)
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(o: pd.Series, h: pd.Series, l: pd.Series,
                       c: pd.Series, v: pd.Series,
                       prefix: str) -> pd.DataFrame:
    """
    Computes all 25–50 technical indicators for ONE timeframe's OHLCV data.

    Parameters
    ----------
    o, h, l, c, v : Open/High/Low/Close/Volume series (same index)
    prefix        : Column name prefix, e.g. '1h_' or '4h_'

    Returns
    -------
    pd.DataFrame  containing only the new indicator columns (with prefix)
    """
    feat = pd.DataFrame(index=c.index)
    p    = prefix   # shorthand

    # ──────────────────────────────────────────────────────────────────────────
    # BASE FEATURES — already stationary or directly useful
    # ──────────────────────────────────────────────────────────────────────────
    feat[f"{p}log_return"]    = np.log(c / c.shift(1))           # per-bar log return
    feat[f"{p}return_5"]      = c.pct_change(5)                  # 5-bar return
    feat[f"{p}return_20"]     = c.pct_change(20)                 # 20-bar return
    feat[f"{p}high_low_pct"]  = (h - l) / (c + 1e-8)            # bar range as % of close
    feat[f"{p}vol_log"]       = np.log(v + 1)                    # log-volume (reduces skew)

    # ──────────────────────────────────────────────────────────────────────────
    # TREND INDICATORS
    # ──────────────────────────────────────────────────────────────────────────

    # 1. MACD — already stationary (difference of EMAs)
    macd_ind = MACD(close=c, window_slow=config.MACD_SLOW,
                    window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
    feat[f"{p}macd"]        = macd_ind.macd()
    feat[f"{p}macd_signal"] = macd_ind.macd_signal()
    feat[f"{p}macd_hist"]   = macd_ind.macd_diff()   # histogram (momentum of momentum)

    # 2. Parabolic SAR — native PSARIndicator from ta.trend
    #    Converts to pct-distance from close (stationary)
    psar_ind = PSARIndicator(high=h, low=l, close=c,
                             step=config.SAR_STEP, max_step=config.SAR_MAX)
    # combine bullish and bearish SAR into a single series
    sar_vals = psar_ind.psar_up().combine_first(psar_ind.psar_down())
    feat[f"{p}sar_dist"] = pct_distance(c, sar_vals)

    # 3. ADX — bounded [0, 100], keep as-is
    adx_ind = ADXIndicator(high=h, low=l, close=c, window=config.ADX_PERIOD)
    feat[f"{p}adx"]     = adx_ind.adx()                       # trend strength
    feat[f"{p}dmp"]     = adx_ind.adx_pos()                   # +DI
    feat[f"{p}dmn"]     = adx_ind.adx_neg()                   # −DI
    feat[f"{p}di_diff"] = feat[f"{p}dmp"] - feat[f"{p}dmn"]  # directional bias

    # 4. Supertrend — pure-pandas implementation (+1 / -1 signal, stationary)
    def _supertrend_dir(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = config.SUPERTREND_PERIOD,
                        mult: float = config.SUPERTREND_MULT) -> pd.Series:
        """Returns +1.0 (bullish) / -1.0 (bearish) Supertrend direction."""
        atr_s = AverageTrueRange(high=high, low=low, close=close,
                                 window=period).average_true_range()
        hl2   = (high + low) / 2
        upper_band = hl2 + mult * atr_s
        lower_band = hl2 - mult * atr_s
        direction  = pd.Series(np.nan, index=close.index)
        final_ub   = upper_band.copy()
        final_lb   = lower_band.copy()
        for i in range(1, len(close)):
            final_ub.iat[i] = (upper_band.iat[i]
                               if upper_band.iat[i] < final_ub.iat[i - 1]
                                  or close.iat[i - 1] > final_ub.iat[i - 1]
                               else final_ub.iat[i - 1])
            final_lb.iat[i] = (lower_band.iat[i]
                               if lower_band.iat[i] > final_lb.iat[i - 1]
                                  or close.iat[i - 1] < final_lb.iat[i - 1]
                               else final_lb.iat[i - 1])
        supertrend = final_ub.copy()
        for i in range(1, len(close)):
            if close.iat[i] > supertrend.iat[i - 1]:
                supertrend.iat[i] = final_lb.iat[i]
            elif close.iat[i] < supertrend.iat[i - 1]:
                supertrend.iat[i] = final_ub.iat[i]
            else:
                supertrend.iat[i] = supertrend.iat[i - 1]
        direction = np.where(close > supertrend, 1.0, -1.0)
        return pd.Series(direction, index=close.index)
    feat[f"{p}supertrend_dir"] = _supertrend_dir(h, l, c)

    # 5. SMA pct-distances — stationary via pct_distance()
    for period in config.SMA_PERIODS:
        sma = SMAIndicator(close=c, window=period).sma_indicator()
        feat[f"{p}sma{period}_dist"] = pct_distance(c, sma)

    # 6. EMA pct-distances — stationary via pct_distance()
    for period in config.EMA_PERIODS:
        ema = EMAIndicator(close=c, window=period).ema_indicator()
        feat[f"{p}ema{period}_dist"] = pct_distance(c, ema)

    # 7. Ichimoku — convert all lines to pct-distance from close
    #    Ichimoku lines are price levels (non-stationary) so we must transform.
    ichi_ind = IchimokuIndicator(high=h, low=l)
    ichi_lines = {
        "tenkan":  ichi_ind.ichimoku_conversion_line(),   # Tenkan-sen
        "kijun":   ichi_ind.ichimoku_base_line(),         # Kijun-sen
        "spana":   ichi_ind.ichimoku_a(),                 # Senkou Span A
        "spanb":   ichi_ind.ichimoku_b(),                 # Senkou Span B
    }
    for name, series in ichi_lines.items():
        feat[f"{p}ichi_{name}_dist"] = pct_distance(c, series)

    # ──────────────────────────────────────────────────────────────────────────
    # MOMENTUM INDICATORS
    # ──────────────────────────────────────────────────────────────────────────

    # 8. RSI — bounded [0, 100], keep as-is
    feat[f"{p}rsi"] = RSIIndicator(close=c, window=config.RSI_PERIOD).rsi()

    # 9. Stochastic %K / %D — bounded [0, 100], keep as-is
    stoch_ind = StochasticOscillator(
        high=h, low=l, close=c,
        window=config.STOCH_K, smooth_window=config.STOCH_D
    )
    feat[f"{p}stoch_k"] = stoch_ind.stoch()
    feat[f"{p}stoch_d"] = stoch_ind.stoch_signal()

    # 10. CCI — unbounded in theory but oscillates; use rolling z-score
    cci_vals = CCIIndicator(high=h, low=l, close=c, window=config.CCI_PERIOD).cci()
    feat[f"{p}cci"] = rolling_zscore(cci_vals)

    # 11. Awesome Oscillator — native AwesomeOscillatorIndicator from ta.momentum
    #     apply rolling z-score for scale stability
    ao_vals = AwesomeOscillatorIndicator(
        high=h, low=l, window1=config.AO_SHORT, window2=config.AO_LONG
    ).awesome_oscillator()
    feat[f"{p}ao"] = rolling_zscore(ao_vals)

    # 12. MFI (Money Flow Index) — bounded [0, 100], keep as-is
    feat[f"{p}mfi"] = MFIIndicator(
        high=h, low=l, close=c, volume=v, window=config.MFI_PERIOD
    ).money_flow_index()

    # ──────────────────────────────────────────────────────────────────────────
    # VOLATILITY INDICATORS
    # ──────────────────────────────────────────────────────────────────────────

    # 13. Bollinger Bands — use %B (position within bands) and bandwidth
    bb_ind = BollingerBands(close=c, window=config.BB_PERIOD, window_dev=config.BB_STD)
    feat[f"{p}bb_pct"]   = bb_ind.bollinger_pband()        # %B — bounded [0,1]
    bb_upper = bb_ind.bollinger_hband()
    bb_lower = bb_ind.bollinger_lband()
    bb_mid   = bb_ind.bollinger_mavg()
    feat[f"{p}bb_width"] = rolling_zscore(
        (bb_upper - bb_lower) / (bb_mid + 1e-8)            # normalised bandwidth
    )

    # 14. ATR as percentage of close — already stationary (ratio)
    atr_vals = AverageTrueRange(
        high=h, low=l, close=c, window=config.ATR_PERIOD
    ).average_true_range()
    feat[f"{p}atr_pct"] = atr_vals / (c + 1e-8)

    # 15. Historical volatility — rolling std of log returns
    feat[f"{p}hvol_20"] = feat[f"{p}log_return"].rolling(20).std()

    # 16. Keltner Channel — use position within channel (similar to %B)
    kc_ind = KeltnerChannel(
        high=h, low=l, close=c,
        window=config.KELTNER_PERIOD,
        window_atr=config.KELTNER_PERIOD,
        multiplier=config.KELTNER_ATR_MULT
    )
    kc_upper = kc_ind.keltner_channel_hband()
    kc_lower = kc_ind.keltner_channel_lband()
    kc_mid   = kc_ind.keltner_channel_mband()
    feat[f"{p}kc_pos"]   = (c - kc_lower) / (kc_upper - kc_lower + 1e-8)
    feat[f"{p}kc_width"] = rolling_zscore(
        (kc_upper - kc_lower) / (kc_mid + 1e-8)
    )

    # 17. Donchian Channels — native DonchianChannel from ta.volatility
    dc_ind   = DonchianChannel(high=h, low=l, close=c, window=config.DONCHIAN_PERIOD)
    dc_upper = dc_ind.donchian_channel_hband()
    dc_lower = dc_ind.donchian_channel_lband()
    feat[f"{p}dc_width"] = rolling_zscore((dc_upper - dc_lower) / (c + 1e-8))
    feat[f"{p}dc_pos"]   = (c - dc_lower) / (dc_upper - dc_lower + 1e-8)

    # ──────────────────────────────────────────────────────────────────────────
    # VOLUME INDICATORS
    # ──────────────────────────────────────────────────────────────────────────

    # 18. OBV — cumulative, trending → MUST apply rolling z-score
    obv_vals = OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
    feat[f"{p}obv_zscore"] = rolling_zscore(obv_vals)

    # 19. Accumulation / Distribution Line (ADL) — trending → rolling z-score
    ad_vals = AccDistIndexIndicator(high=h, low=l, close=c, volume=v).acc_dist_index()
    feat[f"{p}ad_zscore"] = rolling_zscore(ad_vals)

    # 20. Chaikin Money Flow (CMF) — bounded-ish oscillator
    feat[f"{p}cmf"] = ChaikinMoneyFlowIndicator(
        high=h, low=l, close=c, volume=v, window=20
    ).chaikin_money_flow()

    # 21. VWAP pct-distance — rolling VWAP (price level → pct-distance)
    #     (ta library VWAP requires DatetimeIndex; use pure-pandas rolling)
    typical = (h + l + c) / 3
    tp_vol  = typical * v
    vwap    = tp_vol.rolling(config.VWAP_PERIOD).sum() / (
                  v.rolling(config.VWAP_PERIOD).sum() + 1e-8)
    feat[f"{p}vwap_dist"] = pct_distance(c, vwap)

    # 22. Volume ratio — current volume vs 20-bar rolling average
    vol_ma = v.rolling(20).mean()
    feat[f"{p}vol_ratio"] = v / (vol_ma + 1e-8)
    # Filter based on config.INDICATOR_SET
    if getattr(config, "INDICATOR_SET", 22) == 10:
        light_cols = [
            f"{p}log_return", f"{p}vol_log",
            f"{p}macd", f"{p}macd_signal", f"{p}macd_hist",
            f"{p}supertrend_dir",
            f"{p}sma200_dist",
            f"{p}ema50_dist",
            f"{p}rsi",
            f"{p}mfi",
            f"{p}bb_pct", f"{p}bb_width",
            f"{p}atr_pct",
            f"{p}cmf"
        ]
        # Keep only columns that successfully generated and are in the light set
        feat = feat[[c for c in light_cols if c in feat.columns]]

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# APPLY TO ALL TIMEFRAMES IN THE MERGED DATAFRAME
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features_for_symbol(sym_file: str) -> pd.DataFrame:
    """
    Extracts the OHLCV columns for each timeframe from the raw DataFrames,
    computes indicators on their native timeframes, and aligns them to the 
    base timeframe (1h).

    Parameters
    ----------
    sym_file : The filesystem-safe symbol string (e.g. 'BTC_USDT')

    Returns
    -------
    pd.DataFrame  containing only stationary indicator columns and Close/ATR
    """
    all_feats = []
    
    base_raw_path = os.path.join(config.DATA_DIR, f"{sym_file}_{config.BASE_TF}_raw.csv")
    if not os.path.exists(base_raw_path):
        raise FileNotFoundError(f"Missing base data {base_raw_path}")
        
    base_df = pd.read_csv(base_raw_path, index_col=0, parse_dates=True)
    base_index = base_df.index

    for tf in config.TIMEFRAMES:
        prefix = f"{tf}_"
        raw_path = os.path.join(config.DATA_DIR, f"{sym_file}_{tf}_raw.csv")
        try:
            df_raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        except Exception:
            print(f"  [Warning] Raw file for timeframe '{tf}' not found. Skipping.")
            continue

        o = df_raw["Open"]
        h = df_raw["High"]
        l = df_raw["Low"]
        c = df_raw["Close"]
        v = df_raw["Volume"]

        print(f"  [Features] Computing indicators natively for {tf} …")
        feat_df = compute_indicators(o, h, l, c, v, prefix)
        
        if tf != config.BASE_TF:
            # ANTI-LOOKAHEAD: Shift by 1 native candle before broadcasting to the 1h index.
            # This ensures that indicators computed using a candle's close price
            # are only visible after the candle has fully closed.
            feat_df = feat_df.shift(1)
            
        # Align to the 1h index by forward filling the computed indicators
        feat_df = feat_df.reindex(base_index, method='ffill')
        
        all_feats.append(feat_df)
        print(f"    -> {feat_df.shape[1]} raw feature columns generated & aligned.")

    # Horizontally concatenate all timeframe feature columns
    full_feat = pd.concat(all_feats, axis=1)

    # ── Alternative Data Integration ──
    if getattr(config, "USE_ALTERNATIVE_DATA", False):
        merged_path = os.path.join(config.DATA_DIR, f"{sym_file}_merged_all_tfs.csv")
        if os.path.exists(merged_path):
            merged_df = pd.read_csv(merged_path, index_col=0, parse_dates=True)
            
            # 1. Fear and Greed Z-Score (14 days = 336 hours)
            if 'fng_index' in merged_df.columns:
                full_feat['alt_fng_zscore'] = rolling_zscore(merged_df['fng_index']).fillna(0.0)
            else:
                full_feat['alt_fng_zscore'] = 0.0
                
            # 2. Funding Rate Smoothing (3 days = 72 hours)
            if 'funding_rate' in merged_df.columns:
                full_feat['alt_funding_sma'] = merged_df['funding_rate'].rolling(72).mean().fillna(0.0)
            else:
                full_feat['alt_funding_sma'] = 0.0
                
            # 3. Open Interest Change % (24 hours)
            if 'open_interest' in merged_df.columns:
                full_feat['alt_oi_change_24h'] = merged_df['open_interest'].pct_change(24).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            else:
                full_feat['alt_oi_change_24h'] = 0.0
                
            # 4. L2 Imbalance
            if 'l2_imbalance' in merged_df.columns:
                full_feat['alt_l2_imbalance'] = merged_df['l2_imbalance'].rolling(24).mean().fillna(0.0)
            else:
                full_feat['alt_l2_imbalance'] = 0.0
        else:
            full_feat['alt_fng_zscore'] = 0.0
            full_feat['alt_funding_sma'] = 0.0
            full_feat['alt_oi_change_24h'] = 0.0
            full_feat['alt_l2_imbalance'] = 0.0

    # Also keep the base-timeframe Close and ATR columns — needed by the
    # environment for price simulation and stop-loss calculation.
    full_feat["Close"] = base_df["Close"]
    full_feat["ATR"]   = AverageTrueRange(
        high=base_df["High"],
        low=base_df["Low"],
        close=base_df["Close"],
        window=config.ATR_PERIOD,
    ).average_true_range()

    # Drop rows with any NaN values caused by indicator warm-up periods
    before = len(full_feat)
    full_feat.dropna(inplace=True)
    print(f"\n  [Features] Dropped {before - len(full_feat)} NaN rows "
          f"(indicator warm-up). Remaining: {len(full_feat):,} rows.")
    print(f"  [Features] Total feature columns: {full_feat.shape[1]}")

    return full_feat


# ──────────────────────────────────────────────────────────────────────────────
# SCALING — fit on train only, transform all splits (no lookahead)
# ──────────────────────────────────────────────────────────────────────────────

def scale_splits(train_feat: pd.DataFrame,
                 val_feat:   pd.DataFrame,
                 test_feat:  pd.DataFrame,
                 feature_cols: list,
                 force_refit: bool = True) -> tuple:
    """
    Fits a StandardScaler on the TRAINING split only, then applies it to
    all three splits.  This prevents any information about future data from
    leaking into the scaling (a common but subtle form of lookahead bias).

    Parameters
    ----------
    train_feat, val_feat, test_feat : Feature DataFrames for each split
    feature_cols                    : List of columns to scale
    force_refit                     : If False, loads saved scaler if available

    Returns
    -------
    (train_scaled, val_scaled, test_scaled)  — same structure with scaled values
    """
    if os.path.exists(config.SCALER_PATH) and not force_refit:
        print(f"  [Scaler] Loading cached scaler from {config.SCALER_PATH}")
        with open(config.SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    else:
        print(f"  [Scaler] Fitting StandardScaler on TRAINING data only …")
        scaler = StandardScaler()
        scaler.fit(train_feat[feature_cols])
        with open(config.SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  [Scaler] Saved to {config.SCALER_PATH}")

    # Apply transform to all three splits (using the SAME fitted scaler)
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[feature_cols] = scaler.transform(df[feature_cols])
        # Replace any residual NaN / Inf from scaling with 0
        out[feature_cols] = out[feature_cols].replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    train_s = _transform(train_feat)
    val_s   = _transform(val_feat)
    test_s  = _transform(test_feat)

    return train_s, val_s, test_s


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_engineering() -> tuple:
    print("\n" + "=" * 70)
    print("  PHASE 1 — FEATURE ENGINEERING & STATIONARITY ENFORCEMENT")
    print("=" * 70)

    symbols_to_process = config.MULTI_COINS if config.MULTI_COIN_MODE else [config.SYMBOL]
    
    # We will return the primary coin's features
    primary_train_s, primary_val_s, primary_test_s, primary_cols = None, None, None, None

    for symbol in symbols_to_process:
        sym_file = symbol.replace('/', '_')
        print(f"\n>>> Processing {symbol} <<<")
        
        full_feat = engineer_features_for_symbol(sym_file)
        
        n = len(full_feat)
        te = int(n * config.TRAIN_RATIO)
        ve = te + int(n * config.VAL_RATIO)
        
        train_feat = full_feat.iloc[:te]
        val_feat   = full_feat.iloc[te:ve]
        test_feat  = full_feat.iloc[ve:]

        feature_cols = [c for c in full_feat.columns if c not in ("Close", "ATR")]

        train_s, val_s, test_s = scale_splits(
            train_feat, val_feat, test_feat, feature_cols,
            force_refit=config.FORCE_RETRAIN
        )

        train_s.to_csv(os.path.join(config.DATA_DIR, f"{sym_file}_train_features.csv"))
        val_s.to_csv(os.path.join(config.DATA_DIR, f"{sym_file}_val_features.csv"))
        test_s.to_csv(os.path.join(config.DATA_DIR, f"{sym_file}_test_features.csv"))
        
        # If this is the primary symbol, save to the standard paths as well
        if symbol == config.SYMBOL:
            train_s.to_csv(config.TRAIN_FEAT_PATH)
            val_s.to_csv(config.VAL_FEAT_PATH)
            test_s.to_csv(config.TEST_FEAT_PATH)
            primary_train_s, primary_val_s, primary_test_s, primary_cols = train_s, val_s, test_s, feature_cols

    print("\n  PHASE 1 FEATURE ENGINEERING COMPLETE OK")
    return primary_train_s, primary_val_s, primary_test_s, primary_cols

if __name__ == "__main__":
    run_feature_engineering()
