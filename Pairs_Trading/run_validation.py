import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_processing import process_and_split_data
from utils import run_agent, compute_metrics, compute_spread_baseline
from stable_baselines3 import PPO

def main():
    print("Loading model...")
    model = PPO.load(config.MODEL_PATH)

    print("Fetching and processing data (this will use cache or fresh)...")
    _, val_scaled, test_scaled, _, val_raw, test_raw = process_and_split_data()

    print("\n--- Running Validation Set ---")
    val_res = run_agent(model, val_scaled, val_raw, config.FEATURE_COLUMNS)
    val_tm = val_res['metrics']
    val_baseline_pv = compute_spread_baseline(val_raw)
    val_baseline_metrics = compute_metrics(val_baseline_pv, config.TIMEFRAME)
    
    print(f"  Agent Return: {val_tm['total_return']*100:.2f}% | Max DD: {val_tm['max_drawdown']*100:.2f}%")
    print(f"  Base  Return: {val_baseline_metrics['total_return']*100:.2f}% | Max DD: {val_baseline_metrics['max_drawdown']*100:.2f}%")
    print(f"  Trades: {val_res['total_trades']}")

    metrics_path = os.path.join(config.RESULTS_DIR, 'val_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(val_tm, f, indent=2)

    print("\n--- Running Standard Test Set (Test 1) ---")
    test_res = run_agent(model, test_scaled, test_raw, config.FEATURE_COLUMNS)
    test_tm = test_res['metrics']
    test_baseline_pv = compute_spread_baseline(test_raw)
    test_baseline_metrics = compute_metrics(test_baseline_pv, config.TIMEFRAME)
    
    print(f"  Agent Return: {test_tm['total_return']*100:.2f}% | Max DD: {test_tm['max_drawdown']*100:.2f}%")
    print(f"  Base  Return: {test_baseline_metrics['total_return']*100:.2f}% | Max DD: {test_baseline_metrics['max_drawdown']*100:.2f}%")
    print(f"  Trades: {test_res['total_trades']}")
    
    metrics_path = os.path.join(config.RESULTS_DIR, 'test1_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_tm, f, indent=2)

if __name__ == '__main__':
    main()
