"""
phase1_data_extraction.py — Multi-Timeframe CCXT Data Downloader
=================================================================
Phase 1, Step 1 of the pipeline.

What this script does
---------------------
1.  Downloads 5 years of BTC/USDT OHLCV from Binance for three timeframes:
    1h, 4h, and 1d — using the ccxt library.
2.  Merges the 4h and 1d data onto the 1h timeline with STRICT lookahead-bias
    prevention:
        • Higher-timeframe candles are shifted by 1 period *before* ffill.
        • e.g., the 1h bar at 14:00 can only "see" the daily bar that
          CLOSED at the start of that day (00:00), not the in-progress one.
3.  Saves the merged DataFrame to CSV so Phase 2 can load it without
    re-downloading.
4.  Performs a chronological train / validation / test split:
        Train : first 4 years
        Val   : next 6 months
        Test  : final 6 months

CRITICAL ANTI-LOOKAHEAD RULE
-----------------------------
When we merge, we always do:
    higher_tf_df = higher_tf_df.shift(1)   # shift by 1 candle of that TF
    then reindex to the 1h timeline and forward-fill (ffill)

This guarantees that at time t the agent only sees data from *completed*
higher-timeframe candles, never from the candle that is still forming.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime, timezone

# Import our central configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ──────────────────────────────────────────────────────────────────────────────
# 1. CCXT DOWNLOADER
# ──────────────────────────────────────────────────────────────────────────────

def download_ohlcv(symbol: str, timeframe: str, since_ms: int,
                   exchange_id: str = "binance") -> pd.DataFrame:
    """
    Downloads ALL OHLCV candles for a symbol/timeframe pair from `since_ms`
    until the present, batching requests to stay within exchange limits.

    Parameters
    ----------
    symbol      : Trading pair, e.g. 'BTC/USDT'
    timeframe   : Candle size string, e.g. '1h', '4h', '1d'
    since_ms    : Start timestamp in milliseconds (UTC)
    exchange_id : ccxt exchange id (default: 'binance')

    Returns
    -------
    pd.DataFrame  with columns [Open, High, Low, Close, Volume] indexed by UTC
    datetime.
    """
    # Initialise the exchange with rate-limit enforcement
    exchange = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,   # ccxt automatically sleeps between calls
    })

    print(f"  [CCXT] Downloading {symbol} @ {timeframe} from "
          f"{datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).date()} …")

    all_ohlcv = []
    limit     = 1000    # Most exchanges return at most 1 000 candles per call

    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe,
                                         since=since_ms, limit=limit)
        except ccxt.NetworkError as e:
            print(f"  [CCXT] Network error: {e}. Retrying in 10 s …")
            time.sleep(10)
            continue
        except ccxt.ExchangeError as e:
            print(f"  [CCXT] Exchange error: {e}. Aborting.")
            break

        if not batch:
            break                   # No more data available

        all_ohlcv.extend(batch)

        last_ts  = batch[-1][0]
        last_dt  = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"    … fetched up to {last_dt.strftime('%Y-%m-%d %H:%M')} UTC "
              f"({len(all_ohlcv):,} candles so far)", end="\r")

        if len(batch) < limit:
            break                   # We have reached the current time

        since_ms = last_ts + 1     # Next batch starts 1 ms after the last candle

    print()  # newline after the \r progress updates

    if not all_ohlcv:
        raise RuntimeError(f"No data downloaded for {symbol} @ {timeframe}!")

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low",
                                           "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df.drop(columns=["Timestamp"], inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="first")]   # remove any duplicate rows
    df.sort_index(inplace=True)

    print(f"  [CCXT] Done — {len(df):,} candles for {symbol} @ {timeframe}.")
    return df


def fetch_all_timeframes(symbol: str, force_redownload: bool = False) -> dict:
    """
    Downloads (or loads from cache) OHLCV data for all configured timeframes for a given symbol.

    Returns
    -------
    dict  {timeframe_str: pd.DataFrame}
    """
    # Calculate start timestamp: 5 years back from now
    since_ms = int((time.time() - config.LOOKBACK_DAYS * 24 * 3600) * 1000)
    dfs      = {}
    
    symbol_file = symbol.replace("/", "_")

    for tf in config.TIMEFRAMES:
        save_path = os.path.join(config.DATA_DIR, f"{symbol_file}_{tf}_raw.csv")

        # ── Use cached file if it exists and we're not forcing a re-download ──
        if os.path.exists(save_path) and not force_redownload:
            print(f"  [Cache] Loading {tf} data from {save_path}")
            df = pd.read_csv(save_path, index_col="Date", parse_dates=True)
            # Ensure timezone-aware index
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            dfs[tf] = df
        else:
            # ── Download from Binance ──────────────────────────────────────
            df = download_ohlcv(symbol, tf, since_ms)
            df.to_csv(save_path)
            print(f"  [Save] Saved to {save_path}")
            dfs[tf] = df

    return dfs


# ──────────────────────────────────────────────────────────────────────────────
# 2. MULTI-TIMEFRAME ALIGNMENT (LOOKAHEAD-BIAS FREE)
# ──────────────────────────────────────────────────────────────────────────────

def merge_timeframes(dfs: dict) -> pd.DataFrame:
    """
    Merges 4h and 1d DataFrames onto the 1h timeline with strict lookahead
    prevention.

    CRITICAL METHODOLOGY
    --------------------
    For the 4h timeframe:
        - A 4h candle that *closes* at 16:00 is only visible at 20:00 (the
          *next* 4h candle's open).
        - We shift the 4h DataFrame by 1 row *before* reindexing so that
          at 16:00 the agent still sees the candle that closed at 12:00.

    For the 1d timeframe:
        - Similarly, shift by 1 day before forward-filling.

    After shifting we reindex to the 1h index and forward-fill (ffill) so
    every 1h bar carries the last *closed* higher-tf candle values.

    Parameters
    ----------
    dfs : dict  {tf: pd.DataFrame}

    Returns
    -------
    pd.DataFrame  merged on the 1h base timeline, columns prefixed by tf
    """
    base_df = dfs[config.BASE_TF].copy()

    # Rename base columns to have the '1h_' prefix for clarity
    base_df.columns = [f"1h_{c}" for c in base_df.columns]

    merged = base_df.copy()

    for tf in config.TIMEFRAMES:
        if tf == config.BASE_TF:
            continue    # Already have the base

        htf_df = dfs[tf].copy()

        # ── ANTI-LOOKAHEAD: shift by 1 candle of the higher timeframe ──
        # shift(1) pushes the values down by one row, so the value at time t
        # reflects the candle that CLOSED at t-1 (the previous completed bar).
        htf_df = htf_df.shift(1)

        # Prefix columns
        htf_df.columns = [f"{tf}_{c}" for c in htf_df.columns]

        # Reindex to the 1h timeline and forward-fill
        # forward-fill ensures every 1h bar carries the last seen value from
        # the higher timeframe (the last *completed* candle).
        htf_df = htf_df.reindex(merged.index, method="ffill")

        # Merge by joining on the shared index
        merged = merged.join(htf_df, how="left")

    # Drop the initial rows where the forward-fill has NaNs (at the very start
    # before even one 4h / 1d candle has closed)
    merged.dropna(inplace=True)
    print(f"  [Merge] Merged DataFrame shape: {merged.shape}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 3. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame) -> tuple:
    """
    Splits a time-series DataFrame into train, validation, and test sets
    using the ratios defined in config.py.  Data is NEVER shuffled.

    Returns
    -------
    (df_train, df_val, df_test)
    """
    n         = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end   = train_end + int(n * config.VAL_RATIO)

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    print(f"\n  [Split] Chronological split (NO shuffle):")
    print(f"    Train : {len(df_train):>7,} rows  "
          f"({df_train.index[0].date()} → {df_train.index[-1].date()})")
    print(f"    Val   : {len(df_val):>7,} rows  "
          f"({df_val.index[0].date()}   → {df_val.index[-1].date()})")
    print(f"    Test  : {len(df_test):>7,} rows  "
          f"({df_test.index[0].date()}   → {df_test.index[-1].date()})")

    return df_train, df_val, df_test


# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_extraction(force_redownload: bool = False) -> dict:
    """
    Full Phase 1 data extraction pipeline. Processes multiple coins if MULTI_COIN_MODE is true.

    Returns
    -------
    dict: {symbol: (df_train, df_val, df_test)}
    """
    print("\n" + "=" * 70)
    print("  PHASE 1 — DATA EXTRACTION & MULTI-TIMEFRAME MERGING")
    print("=" * 70)

    symbols_to_process = config.MULTI_COINS if config.MULTI_COIN_MODE else [config.SYMBOL]
    all_splits = {}

    for symbol in symbols_to_process:
        print(f"\n>>> Processing {symbol} <<<")
        
        # ── Step 1: Download / load all timeframes ──
        print("\n[Step 1] Fetching OHLCV data …")
        dfs = fetch_all_timeframes(symbol, force_redownload=force_redownload)

        for tf, df in dfs.items():
            print(f"    {tf}: {len(df):,} candles  "
                  f"({df.index[0].date()} → {df.index[-1].date()})")

        # ── Step 2: Merge with lookahead-bias prevention ──
        print("\n[Step 2] Merging timeframes (anti-lookahead shift + ffill) …")
        merged = merge_timeframes(dfs)

        # Save merged dataset
        symbol_file = symbol.replace("/", "_")
        merged_path = os.path.join(config.DATA_DIR, f"{symbol_file}_merged_all_tfs.csv")
        merged.to_csv(merged_path)
        print(f"  [Save] Merged dataset saved to {merged_path}")

        # ── Step 3: Chronological split ──
        print("\n[Step 3] Splitting into train / val / test …")
        df_train, df_val, df_test = chronological_split(merged)
        
        all_splits[symbol] = (df_train, df_val, df_test)

    print("\n  PHASE 1 COMPLETE ✓")
    return all_splits


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cached CSVs exist")
    args = parser.parse_args()

    all_splits = run_extraction(force_redownload=args.force)
    first_sym = list(all_splits.keys())[0]
    train, val, test = all_splits[first_sym]
    print(f"\nTrain columns ({len(train.columns)}) for {first_sym}: {list(train.columns[:5])} …")
    print("Phase 1 standalone test PASSED ✓")
