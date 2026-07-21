import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import itertools

# Top 5 cryptocurrencies by market cap (excluding stablecoins)
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
TIMEFRAME = '1d'
DAYS = 730  # 2 years of daily data

def fetch_daily_data(exchange, symbol, days):
    """Fetch daily closing prices for a given symbol."""
    print(f"Fetching data for {symbol}...")
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
            time.sleep(0.5) # rate limit
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
            
    if not all_ohlcv:
        print(f"Warning: No data fetched for {symbol}.")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    return df['close'].rename(symbol.split('/')[0])

def main():
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Fetch data for all symbols
    series_list = []
    for symbol in SYMBOLS:
        series = fetch_daily_data(exchange, symbol, DAYS)
        if series is not None:
            series_list.append(series)
            
    if not series_list:
        print("Error: No data could be fetched for any symbols.")
        return
    # Merge into a single DataFrame
    df_merged = pd.concat(series_list, axis=1, join='inner').dropna()
    print(f"\nSuccessfully downloaded {len(df_merged)} days of aligned data.\n")
    
    # 1. Calculate Correlation Matrix
    print("="*60)
    print("  PEARSON CORRELATION MATRIX (CLOSING PRICES)")
    print("="*60)
    corr_matrix = df_merged.corr()
    print(corr_matrix.round(4))
    print("\n* Note: Correlation measures how prices move together linearly (-1 to 1).")
    print("* High correlation (> 0.8) means they tend to go up and down together.\n")
    
    # 2. Calculate Pairwise Cointegration
    print("="*60)
    print("  COINTEGRATION P-VALUES (STATIONARY SPREAD TEST)")
    print("="*60)
    
    assets = df_merged.columns
    coint_results = []
    
    for pair in itertools.combinations(assets, 2):
        asset1, asset2 = pair
        # Test cointegration in both directions (dependent variable matters)
        score1, pval1, _ = coint(df_merged[asset1], df_merged[asset2])
        score2, pval2, _ = coint(df_merged[asset2], df_merged[asset1])
        
        # Take the best (lowest) p-value
        best_pval = min(pval1, pval2)
        is_coint = "YES" if best_pval < 0.05 else "NO"
        
        coint_results.append({
            'Pair': f"{asset1} / {asset2}",
            'p-value': best_pval,
            'Cointegrated (<0.05)': is_coint
        })
        
    coint_df = pd.DataFrame(coint_results).sort_values(by='p-value')
    # Format p-value for readability
    coint_df['p-value'] = coint_df['p-value'].apply(lambda x: f"{x:.4e}")
    print(coint_df.to_string(index=False))
    
    print("\n* Note: Cointegration tests if the spread between the two assets is mean-reverting.")
    print("* p-value < 0.05 means they are statistically cointegrated (good for pairs trading).")
    print("* Even if highly correlated, assets might NOT be cointegrated if their spread drifts.")

if __name__ == "__main__":
    main()
