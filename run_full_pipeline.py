"""
run_full_pipeline.py — Single Entry Point for the Complete CLSTM-RL Pipeline
=============================================================================
Run this script to execute ALL four phases sequentially:

    Phase 1a: Multi-timeframe CCXT data download (1h, 4h, 1d)
    Phase 1b: 25-50 indicator feature engineering + stationarity enforcement
    Phase 2 : 3-stage feature selection (MI → Spearman → SHAP)
    Phase 3 : CLSTM-PPO agent training + validation
    Phase 4 : Out-of-sample test + interactive Plotly visualization

Usage
-----
    python run_full_pipeline.py              # full run (uses cache if available)
    python run_full_pipeline.py --redownload # force fresh data download
    python run_full_pipeline.py --skip-train # skip training, load saved model
    python run_full_pipeline.py --phase 2   # start from a specific phase

Output files (all in clstm_rl_pipeline/)
-----------------------------------------
    data/
        BTC_USDT_1h_raw.csv            Raw OHLCV for each timeframe
        BTC_USDT_4h_raw.csv
        BTC_USDT_1d_raw.csv
        merged_all_tfs.csv             Merged multi-timeframe dataset
        train_features.csv             Engineered + scaled features (train)
        val_features.csv               Engineered + scaled features (val)
        test_features.csv              Engineered + scaled features (test)
        golden_features.json           Final selected feature names
        feature_scaler.pkl             Fitted StandardScaler

    models/
        clstm_ppo_btcusdt.zip          Trained PPO model

    results/
        mi_scores.png                  Stage 1 MI bar chart
        correlation_heatmap.png        Stage 2 Spearman heatmap
        shap_beeswarm.png              Stage 3 SHAP beeswarm
        shap_bar.png                   Stage 3 SHAP bar chart
        training_reward_curve.png      Training loss curve
        interactive_backtest.html      🌐 Interactive Plotly dashboard
        test_metrics.json              Final performance metrics
        test_trades.csv                Annotated trade log
        test_summary_chart.png         Static fallback chart
"""

import argparse
import os
import sys
import io
import time

# Force UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CLSTM-RL PPO Cryptocurrency Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--redownload", action="store_true",
        help="Force re-download of raw data from Binance (ignore cached CSVs)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip PPO training and load the saved model instead"
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3, 4],
        help="Start from a specific phase (1=data, 2=selection, 3=train, 4=test)"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    t_start  = time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   CLSTM-RL PPO Cryptocurrency Trading Agent — Full Pipeline     ║")
    print("║   BTC/USDT | Binance | 5 Years | 1h/4h/1d Multi-Timeframe      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1a — DATA EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────
    if args.phase <= 1:
        from phase1_data_extraction import run_extraction
        all_splits = run_extraction(
            force_redownload=args.redownload
        )
    else:
        # Load already-merged data
        import pandas as pd
        print("[Skip] Phase 1a — Loading merged data from disk …")
        merged = pd.read_csv(config.MERGED_DATA_PATH, index_col=0, parse_dates=True)
        n      = len(merged)
        te     = int(n * config.TRAIN_RATIO)
        ve     = te + int(n * config.VAL_RATIO)
        df_train_raw = merged.iloc[:te]
        df_val_raw   = merged.iloc[te:ve]
        df_test_raw  = merged.iloc[ve:]

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1b — FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────────
    if args.phase <= 1:
        from phase1_feature_engineering import run_feature_engineering
        train_feat, val_feat, test_feat, feature_cols = run_feature_engineering()
    else:
        import pandas as pd
        print("[Skip] Phase 1b — Loading feature CSVs from disk …")
        train_feat   = pd.read_csv(config.TRAIN_FEAT_PATH, index_col=0, parse_dates=True)
        val_feat     = pd.read_csv(config.VAL_FEAT_PATH,   index_col=0, parse_dates=True)
        test_feat    = pd.read_csv(config.TEST_FEAT_PATH,  index_col=0, parse_dates=True)
        feature_cols = [c for c in train_feat.columns if c not in ("Close", "ATR")]

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — FEATURE SELECTION
    # ──────────────────────────────────────────────────────────────────────────
    if args.phase <= 2:
        from phase2_feature_selection import run_feature_selection
        golden_features = run_feature_selection(
            train_feat, val_feat, test_feat, feature_cols
        )
    else:
        import json
        print("[Skip] Phase 2 — Loading golden features from disk …")
        with open(config.GOLDEN_FEATURES_PATH) as fp:
            golden_features = json.load(fp)
        print(f"  Golden features: {golden_features}")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3 — TRAINING
    # ──────────────────────────────────────────────────────────────────────────
    if args.skip_train:
        config.FORCE_RETRAIN = False
        print("\n[Skip] Phase 3 — FORCE_RETRAIN=False, will load saved model …")

    if args.phase <= 3:
        from phase3_train import run_training
        model = run_training()
    else:
        from stable_baselines3 import PPO
        print("[Skip] Phase 3 — Loading model from disk …")
        model = PPO.load(config.MODEL_PATH)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4 — TESTING & VISUALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    if args.phase <= 4:
        from phase4_test_and_visualize import run_test_and_visualize
        run_test_and_visualize()

    # ──────────────────────────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   ✅  FULL PIPELINE COMPLETE                                     ║")
    print(f"║   Total runtime: {minutes}m {seconds}s{' '*(47 - len(str(minutes)) - len(str(seconds)))}║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("📁 Key output files:")
    print(f"   🌐 Interactive chart : {os.path.join(config.RESULTS_DIR, 'interactive_backtest.html')}")
    print(f"   📊 Metrics JSON      : {os.path.join(config.RESULTS_DIR, 'test_metrics.json')}")
    print(f"   📋 Trade log CSV     : {os.path.join(config.RESULTS_DIR, 'test_trades.csv')}")
    print(f"   🤖 Trained model     : {config.MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
