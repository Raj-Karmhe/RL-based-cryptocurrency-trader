import ccxt
import pandas as pd
import time
import os
from datetime import datetime

# ==============================================================================
# DATA DOWNLOADER FOR PAIRS TRADING (ENHANCED)
# Downloads historical OHLCV data AND funding rates for BOTH assets.
# Data is fetched from Binance, aligned by timestamp, and saved to CSV.
# ==============================================================================

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def fetch_ohlcv(symbol: str, timeframe: str = '1h', years: int = 5) -> pd.DataFrame:
    """Downloads historical OHLCV data from Binance for a single asset."""
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int((time.time() - years * 365.25 * 24 * 3600) * 1000)
    limit = 1000

    print(f"\n{'-'*60}")
    print(f"  Fetching {timeframe} OHLCV data for {symbol}")
    print(f"  Starting from: {datetime.fromtimestamp(since_ms / 1000)}")
    print(f"{'-'*60}")

    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            last_ts = ohlcv[-1][0]
            since_ms = last_ts + 1

            if len(all_ohlcv) % 5000 < limit:
                print(f"  Fetched {len(all_ohlcv):>8,} candles | "
                      f"Up to: {datetime.fromtimestamp(last_ts / 1000)}")

            if len(ohlcv) < limit:
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(5)
            continue

    if not all_ohlcv:
        print(f"  FAILED: No OHLCV data fetched for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)

    print(f"  SUCCESS: {len(df):,} candles fetched for {symbol}")
    return df


def fetch_funding_rates(symbol: str, years: int = 5) -> pd.DataFrame:
    """Downloads historical funding rates for a perpetual futures symbol."""
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int((time.time() - years * 365.25 * 24 * 3600) * 1000)
    limit = 1000
    
    # Map 'ETH/USDT' to 'ETH/USDT:USDT' for Binance perpetuals
    perp_symbol = symbol if ':' in symbol else f"{symbol}:USDT"

    print(f"\n{'-'*60}")
    print(f"  Fetching funding rates for {perp_symbol}")
    print(f"  Starting from: {datetime.fromtimestamp(since_ms / 1000)}")
    print(f"{'-'*60}")

    all_rates = []
    while True:
        try:
            rates = exchange.fetch_funding_rate_history(perp_symbol, since=since_ms, limit=limit)
            if not rates:
                break
            
            all_rates.extend(rates)
            last_ts = rates[-1]['timestamp']
            since_ms = last_ts + 1
            
            if len(all_rates) % 1000 < limit:
                print(f"  Fetched {len(all_rates):>8,} funding rates | "
                      f"Up to: {datetime.fromtimestamp(last_ts / 1000)}")
                
            if len(rates) < limit:
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(5)
            continue
            
    if not all_rates:
        print(f"  FAILED: No funding rates fetched for {perp_symbol}")
        # Return an empty df with Funding_Rate column so joins don't fail
        df = pd.DataFrame(columns=['Date', 'Funding_Rate'])
        df.set_index('Date', inplace=True)
        return df

    df = pd.DataFrame(all_rates)
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['fundingRate']]
    df.rename(columns={'fundingRate': 'Funding_Rate'}, inplace=True)
    # Ensure values are float
    df['Funding_Rate'] = df['Funding_Rate'].astype(float)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)

    print(f"  SUCCESS: {len(df):,} funding rates fetched for {perp_symbol}")
    return df


def download_and_align_pair(force_download=False):
    """
    Downloads OHLCV & funding rates for both assets, aligns them by timestamp,
    forward-fills funding rates, and saves individual + merged CSVs.
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

    # Note: For development speed, we skip caching if force_download is true,
    # but the system should normally cache data. Since we just added funding,
    # we force a fresh download this one time by ignoring the cache if missing columns.

    df_a = fetch_ohlcv(config.ASSET_A, config.TIMEFRAME, config.DATA_YEARS)
    df_b = fetch_ohlcv(config.ASSET_B, config.TIMEFRAME, config.DATA_YEARS)
    
    funding_a = fetch_funding_rates(config.ASSET_A, config.DATA_YEARS)
    funding_b = fetch_funding_rates(config.ASSET_B, config.DATA_YEARS)

    if df_a.empty or df_b.empty:
        raise RuntimeError("Failed to fetch OHLCV data. Check network/API.")

    # Join funding rates (which are 8-hourly) into the hourly OHLCV data
    df_a = df_a.join(funding_a, how='left')
    df_a['Funding_Rate'] = df_a['Funding_Rate'].ffill().fillna(0.0)
    
    df_b = df_b.join(funding_b, how='left')
    df_b['Funding_Rate'] = df_b['Funding_Rate'].ffill().fillna(0.0)

    # ── Align timestamps (inner join) ─────────────────────────────────────
    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    print(f"\n  Aligned timestamps: {len(common_idx):,} common candles")

    # ── Build merged DataFrame ────────────────────────────────────────────
    df_merged = pd.DataFrame(index=common_idx)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Funding_Rate']:
        df_merged[f'{col}_A'] = df_a[col].values
        df_merged[f'{col}_B'] = df_b[col].values

    # ── Save to CSV ───────────────────────────────────────────────────────
    df_a.to_csv(config.ASSET_A_PATH)
    df_b.to_csv(config.ASSET_B_PATH)
    df_merged.to_csv(config.MERGED_PATH)

    print(f"\n  Saved: {config.ASSET_A_PATH}")
    print(f"  Saved: {config.ASSET_B_PATH}")
    print(f"  Saved: {config.MERGED_PATH}")
    print(f"\nSUCCESS: Data download complete!")

    return df_a, df_b, df_merged


if __name__ == '__main__':
    df_a, df_b, df_merged = download_and_align_pair()
    print(f"\n{'─'*60}")
    print(f"  Summary")
    print(f"{'─'*60}")
    print(f"  {config.ASSET_A_LABEL}: {len(df_a):>8,} candles | "
          f"{df_a.index[0].date()} → {df_a.index[-1].date()}")
    print(f"  {config.ASSET_B_LABEL}: {len(df_b):>8,} candles | "
          f"{df_b.index[0].date()} → {df_b.index[-1].date()}")
    print(f"  Merged:  {len(df_merged):>8,} aligned candles")
