"""
phase1_data_extraction.py - Multi-Timeframe Data Downloader & Merger

This script downloads historical OHLCV (Open, High, Low, Close, Volume) data
from Binance for multiple timeframes, aligns them to a base timeframe without lookahead bias,
and splits the data chronologically into train, validation, and test sets.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone

# Ensure workspace directory is in the path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def download_historical_ohlcv(symbol, timeframe, start_ms):
    """
    Downloads historical OHLCV data using ccxt from Binance.
    Fetches data in batches until the current time is reached.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True
    })
    
    print(f"Downloading {symbol} OHLCV for timeframe '{timeframe}'...")
    
    ohlcv_data = []
    current_since = start_ms
    limit = 1000
    
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not batch:
                break
                
            ohlcv_data.extend(batch)
            last_timestamp = batch[-1][0]
            
            # Print progress
            last_date = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            print(f"  Downloaded up to {last_date} ({len(ohlcv_data)} candles)", end='\r')
            
            if len(batch) < limit:
                break
                
            current_since = last_timestamp + 1
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
            
        except Exception as e:
            print(f"\nError encountered during fetch: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            
    print()  # Newline after progress print
    
    if not ohlcv_data:
        raise ValueError(f"No data returned for {symbol} on timeframe {timeframe}")
        
    df = pd.DataFrame(ohlcv_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    df = df.astype(float)
    
    # Remove any duplicates and sort chronologically
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    return df

def align_and_merge(timeframe_dfs):
    """
    Merges higher timeframe data (4h, 1d) onto the base 1h timeframe.
    To prevent lookahead bias, higher timeframes are shifted by 1 candle BEFORE
    reindexing and forward-filling.
    """
    base_tf = config.BASE_TF
    base_df = timeframe_dfs[base_tf].copy()
    
    # Prefix columns for base timeframe
    base_df.columns = [f"{base_tf}_{col}" for col in base_df.columns]
    merged_df = base_df
    
    for tf, df in timeframe_dfs.items():
        if tf == base_tf:
            continue
            
        # Shift HTF dataframe by 1 period to prevent lookahead bias
        shifted_df = df.shift(1)
        
        # Prefix columns
        shifted_df.columns = [f"{tf}_{col}" for col in shifted_df.columns]
        
        # Reindex to base timeframe timeline and forward-fill values
        aligned_df = shifted_df.reindex(merged_df.index, method='ffill')
        
        # Join dataframes
        merged_df = merged_df.join(aligned_df, how='left')
        
    # Drop rows that have NaN values (mostly warm-up period at start of the series)
    merged_df.dropna(inplace=True)
    print(f"Timeframes merged. Combined shape: {merged_df.shape}")
    return merged_df

def split_data(df):
    """
    Splits the dataframe chronologically into Train (4 years), Val (6 months), and Test (6 months)
    based on exact dates, relative to the end of the dataset.
    """
    max_date = df.index.max()
    test_start = max_date - pd.Timedelta(days=180)
    val_start = test_start - pd.Timedelta(days=180)
    train_start = val_start - pd.Timedelta(days=4*365)
    
    train_df = df[(df.index >= train_start) & (df.index < val_start)].copy()
    val_df = df[(df.index >= val_start) & (df.index < test_start)].copy()
    test_df = df[df.index >= test_start].copy()
    
    print("\nData Split Summary:")
    for label, split_df in [("Train", train_df), ("Val  ", val_df), ("Test ", test_df)]:
        if len(split_df) > 0:
            print(f"  {label} : {len(split_df):,} rows ({split_df.index[0].date()} to {split_df.index[-1].date()})")
        else:
            print(f"  {label} : 0 rows (EMPTY — insufficient data for this split)")
    
    return train_df, val_df, test_df

def run_extraction_pipeline():
    """
    Main pipeline to orchestrate downloading, merging, and splitting.
    """
    print("=" * 60)
    print("Starting Data Extraction and Preparation Pipeline")
    print("=" * 60)
    
    symbols = config.MULTI_COINS if config.MULTI_COIN_MODE else [config.SYMBOL]
    start_time_sec = time.time() - (config.LOOKBACK_DAYS * 24 * 60 * 60)
    start_ms = int(start_time_sec * 1000)
    
    results = {}
    
    for symbol in symbols:
        print(f"\nProcessing Symbol: {symbol}")
        symbol_file = symbol.replace("/", "_")
        
        timeframe_dfs = {}
        for tf in config.TIMEFRAMES:
            raw_file_name = f"{symbol_file}_{tf}_raw.csv"
            raw_file_path = os.path.join(config.DATA_DIR, raw_file_name)
            
            # Download and save raw data
            df = download_historical_ohlcv(symbol, tf, start_ms)
            df.to_csv(raw_file_path)
            print(f"  Saved raw data to {raw_file_path}")
            timeframe_dfs[tf] = df
            
        # Merge timeframes with lookahead bias checks
        merged_df = align_and_merge(timeframe_dfs)
        merged_path = os.path.join(config.DATA_DIR, f"{symbol_file}_merged_all_tfs.csv")
        merged_df.to_csv(merged_path)
        print(f"  Saved merged data to {merged_path}")
        
        # Split data chronologically
        train_df, val_df, test_df = split_data(merged_df)
        
        # Save splits
        train_df.to_csv(os.path.join(config.DATA_DIR, f"{symbol_file}_train_raw.csv"))
        val_df.to_csv(os.path.join(config.DATA_DIR, f"{symbol_file}_val_raw.csv"))
        test_df.to_csv(os.path.join(config.DATA_DIR, f"{symbol_file}_test_raw.csv"))
        
        results[symbol] = (train_df, val_df, test_df)
        
    print("\nData extraction and alignment completed successfully.")
    return results

if __name__ == "__main__":
    run_extraction_pipeline()
