import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# ==============================================================================
# DATA DOWNLOADER FOR PAIRS TRADING (EQUITIES)
# Downloads historical OHLCV data for stock pairs using yfinance.
# ==============================================================================

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def fetch_ohlcv(symbol: str, years: int = 10) -> pd.DataFrame:
    """Downloads historical OHLCV data from Yahoo Finance for a single asset."""
    print(f"\n{'-'*60}")
    print(f"  Fetching {config.TIMEFRAME} OHLCV data for {symbol} ({years} years)")
    print(f"{'-'*60}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        # yfinance handles the API limits automatically
        df = yf.download(symbol, start=start_date, end=end_date, interval=config.TIMEFRAME, progress=False)
        
        if df.empty:
            print(f"  FAILED: No OHLCV data fetched for {symbol}")
            return pd.DataFrame()
            
        # Clean up column names which sometimes have multi-index in yf
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index.name = 'Date'
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Equities don't have funding rates, but we add a 0 column so downstream processing works
        df['Funding_Rate'] = 0.0
        
        print(f"  SUCCESS: {len(df):,} rows fetched for {symbol}")
        return df
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

def download_and_align_pair(force_download=False):
    """
    Downloads OHLCV for both stock assets, aligns them by timestamp,
    and saves individual + merged CSVs.
    """
    if not force_download and os.path.exists(config.ASSET_A_PATH) and os.path.exists(config.ASSET_B_PATH) and os.path.exists(config.MERGED_PATH):
        print("=" * 70)
        print("  PAIRS TRADING DATA DOWNLOAD")
        print("=" * 70)
        print(f"\n  Loading cached data from:")
        print(f"  {config.MERGED_PATH}")
        
        df_a = pd.read_csv(config.ASSET_A_PATH, index_col='Date', parse_dates=True)
        df_b = pd.read_csv(config.ASSET_B_PATH, index_col='Date', parse_dates=True)
        df_merged = pd.read_csv(config.MERGED_PATH, index_col='Date', parse_dates=True)
        return df_a, df_b, df_merged

    print("=" * 70)
    print("  PAIRS TRADING DATA DOWNLOAD")
    print(f"  Asset A: {config.ASSET_A}  |  Asset B: {config.ASSET_B}")
    print(f"  Timeframe: {config.TIMEFRAME}  |  Period: {config.DATA_YEARS} years")
    print("=" * 70)

    df_a = fetch_ohlcv(config.ASSET_A, config.DATA_YEARS)
    df_b = fetch_ohlcv(config.ASSET_B, config.DATA_YEARS)

    if df_a.empty or df_b.empty:
        raise RuntimeError("Failed to fetch OHLCV data. Check network or valid Yahoo Finance tickers.")

    # ── Align timestamps (inner join) ─────────────────────────────────────
    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    print(f"\n  Aligned timestamps: {len(common_idx):,} common trading days")

    # ── Build merged DataFrame ────────────────────────────────────────────
    df_merged = pd.DataFrame(index=common_idx)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Funding_Rate']:
        df_merged[f'{col}_A'] = df_a[col].values
        df_merged[f'{col}_B'] = df_b[col].values

    # ── Save to CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.ASSET_A_PATH), exist_ok=True)
    df_a.to_csv(config.ASSET_A_PATH)
    df_b.to_csv(config.ASSET_B_PATH)
    df_merged.to_csv(config.MERGED_PATH)

    print(f"\n  Saved: {config.ASSET_A_PATH}")
    print(f"  Saved: {config.ASSET_B_PATH}")
    print(f"  Saved: {config.MERGED_PATH}")
    print(f"\nSUCCESS: Data download complete!")

    return df_a, df_b, df_merged


if __name__ == '__main__':
    df_a, df_b, df_merged = download_and_align_pair(force_download=True)
    print(f"\n{'─'*60}")
    print(f"  Summary")
    print(f"{'─'*60}")
    print(f"  {config.ASSET_A_LABEL}: {len(df_a):>8,} rows | "
          f"{df_a.index[0].date()} → {df_a.index[-1].date()}")
    print(f"  {config.ASSET_B_LABEL}: {len(df_b):>8,} rows | "
          f"{df_b.index[0].date()} → {df_b.index[-1].date()}")
    print(f"  Merged:  {len(df_merged):>8,} aligned days")
