import pandas as pd
import numpy as np

merged_file = r"c:\IITISOC\clstm_rl_pipeline\data\merged_all_tfs.csv"
try:
    df = pd.read_csv(merged_file)
    
    if "1h_High" in df.columns:
        high = df["1h_High"]
        low = df["1h_Low"]
        close = df["1h_Close"]
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # 14-period SMA of TR
        atr = tr.rolling(window=14).mean()
        
        atr_pct = (atr / close) * 100
        
        print("--- RAW 1H ATR PERCENTAGE STATS ---")
        print(f"Mean:   {atr_pct.mean():.4f}%")
        print(f"Median: {atr_pct.median():.4f}%")
        print(f"StdDev: {atr_pct.std():.4f}%")
        print(f"Max:    {atr_pct.max():.4f}%")
        print(f"Min:    {atr_pct.min():.4f}%")
        
except Exception as e:
    print(f"Error: {e}")
