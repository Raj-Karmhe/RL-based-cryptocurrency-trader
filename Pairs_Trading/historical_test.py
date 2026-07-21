import os
import sys
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import run_agent, compute_metrics, compute_spread_baseline
from stable_baselines3 import PPO

# 1. DOWNLOAD DATA
def fetch_historical_ohlcv(symbol: str, start_ms: int, end_ms: int, timeframe: str = '1h') -> pd.DataFrame:
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = start_ms
    limit = 1000
    all_ohlcv = []
    
    print(f"Fetching {symbol} OHLCV from {datetime.fromtimestamp(start_ms/1000)} to {datetime.fromtimestamp(end_ms/1000)}")
    while since_ms < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            if not ohlcv:
                break
            # filter out candles beyond end_ms
            ohlcv = [x for x in ohlcv if x[0] <= end_ms]
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            since_ms = ohlcv[-1][0] + 1
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue
            
    if not all_ohlcv:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    return df

def fetch_historical_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = start_ms
    limit = 1000
    perp_symbol = symbol if ':' in symbol else f"{symbol}:USDT"
    
    all_rates = []
    print(f"Fetching {perp_symbol} funding from {datetime.fromtimestamp(start_ms/1000)} to {datetime.fromtimestamp(end_ms/1000)}")
    while since_ms < end_ms:
        try:
            rates = exchange.fetch_funding_rate_history(perp_symbol, since=since_ms, limit=limit)
            if not rates:
                break
            rates = [r for r in rates if r['timestamp'] <= end_ms]
            if not rates:
                break
            all_rates.extend(rates)
            since_ms = rates[-1]['timestamp'] + 1
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue
            
    if not all_rates:
        df = pd.DataFrame(columns=['Date', 'Funding_Rate'])
        df.set_index('Date', inplace=True)
        return df
        
    df = pd.DataFrame(all_rates)
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['fundingRate']]
    df.rename(columns={'fundingRate': 'Funding_Rate'}, inplace=True)
    df['Funding_Rate'] = df['Funding_Rate'].astype(float)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    return df

def process_historical_data(df_merged):
    import data_processing
    
    close_a = df_merged['Close_A']
    close_b = df_merged['Close_B']
    log_a = np.log(close_a)
    log_b = np.log(close_b)
    
    hedge_ratio, intercepts = data_processing.compute_rolling_hedge_ratio(log_a, log_b)
    spread = data_processing.compute_spread(log_a, log_b, hedge_ratio, intercepts)
    
    feat_a = data_processing.add_asset_features(df_merged, '_A')
    feat_b = data_processing.add_asset_features(df_merged, '_B')
    feat_spread = data_processing.add_spread_features(spread, hedge_ratio)
    feat_cross = data_processing.add_cross_features(close_a, close_b, df_merged['Volume_A'], df_merged['Volume_B'])
    feat_tf = data_processing.add_multi_tf_features(df_merged)
    turbulence = data_processing.compute_turbulence(close_a, close_b)
    
    df_full = pd.concat([df_merged, feat_a, feat_b, feat_spread, feat_cross, feat_tf], axis=1)
    df_full['Spread'] = spread
    df_full['Hedge_Ratio'] = hedge_ratio
    df_full['Turbulence'] = turbulence
    
    df_full = df_full.dropna(subset=config.FEATURE_COLUMNS)
    df = df_full.copy()
    
    do_not_scale = [
        'Spread_ZScore', 'Spread_ZScore_Velocity', 'Spread_ZScore_Accel',
        'Spread_ZScore_4h', 'Spread_ZScore_1d', 'Price_Ratio_ZScore',
        'Spread_CDF_KDE', 'Cointegration_P_Value_4h', 'Cointegration_P_Value_1d',
        'Spread_BB_Position', 'BB_Position_A', 'BB_Position_B', 'Half_Life', 'Volume_Corr'
    ]
    columns_to_scale = [col for col in config.FEATURE_COLUMNS if col not in do_not_scale]
    
    df_scaled = df.copy()
    for col in columns_to_scale:
        if 'RSI' in col:
            df_scaled[col] = df_scaled[col] / 100.0
        elif col in df_scaled.columns:
            col_mean = df_scaled[col].rolling(window=720, min_periods=1).mean()
            col_std = df_scaled[col].rolling(window=720, min_periods=1).std()
            df_scaled[col] = (df_scaled[col] - col_mean) / (col_std + 1e-8)
            df_scaled[col] = df_scaled[col].fillna(0)
            
    return df, df_scaled

def main():
    # Target period: July 2019 to July 2021 (7 years ago to 5 years ago from 2026)
    # But just to be precise to the previous logs: 2021-07-21 14:00 is end.
    end_ms = int(datetime(2021, 7, 21, 14, 0).timestamp() * 1000)
    start_ms = int(datetime(2019, 7, 21, 14, 0).timestamp() * 1000)
    
    print(f"Targeting data from 2019-07-21 to 2021-07-21")
    
    df_a = fetch_historical_ohlcv(config.ASSET_A, start_ms, end_ms)
    df_b = fetch_historical_ohlcv(config.ASSET_B, start_ms, end_ms)
    
    fund_a = fetch_historical_funding(config.ASSET_A, start_ms, end_ms)
    fund_b = fetch_historical_funding(config.ASSET_B, start_ms, end_ms)
    
    if df_a.empty or df_b.empty:
        print("Not enough data fetched.")
        return
        
    df_a = df_a.join(fund_a, how='left')
    df_a['Funding_Rate'] = df_a['Funding_Rate'].ffill().fillna(0.0)
    
    df_b = df_b.join(fund_b, how='left')
    df_b['Funding_Rate'] = df_b['Funding_Rate'].ffill().fillna(0.0)
    
    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]
    
    df_merged = pd.DataFrame(index=common_idx)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Funding_Rate']:
        df_merged[f'{col}_A'] = df_a[col].values
        df_merged[f'{col}_B'] = df_b[col].values
        
    print(f"Merged aligned candles: {len(df_merged)}")
    
    df_raw, df_scaled = process_historical_data(df_merged)
    print(f"Processed features: {len(df_raw)} candles remaining")
    
    # Split into 4 parts (test2, test3, test4, test5)
    n_splits = 4
    split_size = len(df_raw) // n_splits
    
    print("\nLoading model...")
    model = PPO.load(config.MODEL_PATH)
    
    results = {}
    
    for i in range(n_splits):
        test_name = f"test{i+2}"
        start_idx = i * split_size
        end_idx = (i+1) * split_size if i < n_splits - 1 else len(df_raw)
        
        part_raw = df_raw.iloc[start_idx:end_idx].copy()
        part_scaled = df_scaled.iloc[start_idx:end_idx].copy()
        
        print(f"\n--- Running {test_name} ({part_raw.index[0]} to {part_raw.index[-1]}) ---")
        
        res = run_agent(model, part_scaled, part_raw, config.FEATURE_COLUMNS)
        tm = res['metrics']
        
        baseline_pv = compute_spread_baseline(part_raw)
        baseline_metrics = compute_metrics(baseline_pv, config.TIMEFRAME)
        
        print(f"  Agent Return: {tm['total_return']*100:.2f}% | Max DD: {tm['max_drawdown']*100:.2f}%")
        print(f"  Base  Return: {baseline_metrics['total_return']*100:.2f}% | Max DD: {baseline_metrics['max_drawdown']*100:.2f}%")
        print(f"  Trades: {res['total_trades']}")
        
        # Save metrics
        metrics_path = os.path.join(config.RESULTS_DIR, f'{test_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(tm, f, indent=2)
            
        results[test_name] = tm
        
    print("\nCompleted all historical tests!")

if __name__ == '__main__':
    main()
