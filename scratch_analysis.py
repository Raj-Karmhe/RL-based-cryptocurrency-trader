import pandas as pd
import numpy as np

trades_file = r"c:\IITISOC\clstm_rl_pipeline\results\test_trades.csv"
try:
    df = pd.read_csv(trades_file, parse_dates=["date"])
    
    # Analyze Turbulence
    turb_exits = df[df["exit_reason"] == "turbulence"].copy()
    turb_exits.sort_values("date", inplace=True)
    
    print("--- TURBULENCE ANALYSIS ---")
    print(f"Total Turbulence Exits: {len(turb_exits)}")
    if len(turb_exits) > 1:
        time_diffs = turb_exits["date"].diff().dropna()
        hours_diff = time_diffs.dt.total_seconds() / 3600
        print(f"Average gap between turbulence exits: {hours_diff.mean():.2f} hours")
        print(f"Median gap: {hours_diff.median():.2f} hours")
        print(f"Min gap: {hours_diff.min():.2f} hours")
        print(f"Max gap: {hours_diff.max():.2f} hours")
    else:
        print("Not enough turbulence exits to compute gaps.")
        
except Exception as e:
    print(f"Error reading trades: {e}")

# Estimate Slippage Proxy from ATR in test_features.csv
feat_file = r"c:\IITISOC\clstm_rl_pipeline\data\test_features.csv"
try:
    feat = pd.read_csv(feat_file)
    # If we assume slippage is roughly proportional to the 1h high-low range (spread proxy)
    # We can calculate the high-low percentage
    if "1h_High" in feat.columns and "1h_Low" in feat.columns and "1h_Close" in feat.columns:
        hl_pct = (feat["1h_High"] - feat["1h_Low"]) / feat["1h_Close"]
        # Typical slippage on a liquid exchange for a medium order might be 5% of the 1h High-Low spread
        # This is just a proxy since we don't have order book data
        proxy_slippage = hl_pct * 0.05 
        
        print("\n--- SLIPPAGE PROXY ANALYSIS (Based on 5% of 1h High-Low Range) ---")
        print(f"Average Slippage Proxy: {proxy_slippage.mean()*100:.4f}%")
        print(f"Median Slippage Proxy: {proxy_slippage.median()*100:.4f}%")
        print(f"Std Dev: {proxy_slippage.std()*100:.4f}%")
        print(f"Max Slippage Proxy: {proxy_slippage.max()*100:.4f}%")
        print(f"Min Slippage Proxy: {proxy_slippage.min()*100:.4f}%")
    else:
        print("\nCould not find 1h_High/Low columns in features.")
except Exception as e:
    print(f"Error reading features: {e}")
