"""
phase2_feature_selection.py - Three-Stage Systematic Feature Selection

This script loads the scaled features, prepares a supervised proxy target (forward log-returns),
and runs three stages of feature selection:
1. Mutual Information (MI) filter to drop non-informative features.
2. Spearman Correlation filter to remove redundant/collinear features.
3. LightGBM + SHAP explainer to rank and select the top 4 to 8 golden features.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    import shap
except ImportError as e:
    raise ImportError(f"Missing required feature selection library: {e}. Install with: pip install lightgbm shap")

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def construct_target(close_series: pd.Series, k_horizon: int) -> pd.Series:
    """
    Computes the k-period forward log-return as a proxy target for supervised selection.
    y_t = log(Close_{t+k} / Close_t)
    """
    result = np.log(close_series.shift(-k_horizon) / (close_series + 1e-8))
    # Bug #3 fix: clip infinities and NaNs that can arise from edge cases
    return result.replace([np.inf, -np.inf], np.nan)

def run_stage1_mi(X: pd.DataFrame, y: pd.Series, drop_ratio: float) -> tuple:
    """
    Stage 1: Mutual Information regression filter.
    Calculates mutual information between features and target, and drops the bottom drop_ratio.
    """
    print("\n[Feature Selection - Stage 1] Running Mutual Information Regression...")
    mi_vals = mutual_info_regression(X, y, random_state=config.SEED)
    mi_series = pd.Series(mi_vals, index=X.columns).sort_values(ascending=False)
    
    print("  Top 15 features by MI score:")
    for feat, score in mi_series.head(15).items():
        print(f"    {feat:<45}: {score:.6f}")
        
    num_to_drop = int(len(mi_series) * drop_ratio)
    # Keep top (total - num_to_drop) features by rank to avoid ties at boundary
    num_to_keep = len(mi_series) - num_to_drop
    surviving_features = mi_series.head(num_to_keep).index.tolist()
    
    print(f"  Stage 1 complete: Dropping bottom {drop_ratio*100:.0f}% ({num_to_drop} features).")
    print(f"  Surviving features: {len(surviving_features)}")
    
    return surviving_features, mi_series

def run_stage2_correlation(X: pd.DataFrame, mi_series: pd.Series, correlation_threshold: float) -> list:
    """
    Stage 2: Spearman Rank Correlation filter.
    Identifies collinear feature pairs. Between two highly correlated features, the one with
    the lower Mutual Information score is dropped.
    """
    print(f"\n[Feature Selection - Stage 2] Pruning collinear features (threshold = {correlation_threshold})...")
    features_list = X.columns.tolist()
    
    if len(features_list) <= 1:
        return features_list
        
    # Bug #14 fix: spearmanr returns a scalar when only 2 features are passed;
    # handle this edge case to prevent pd.DataFrame construction failure
    if len(features_list) == 2:
        corr_val, _ = spearmanr(X[features_list[0]], X[features_list[1]])
        corr_matrix = np.array([[1.0, corr_val], [corr_val, 1.0]])
    else:
        corr_matrix, _ = spearmanr(X[features_list])
    corr_df = pd.DataFrame(corr_matrix, index=features_list, columns=features_list)
    
    features_to_drop = set()
    
    for idx_i, feat_i in enumerate(features_list):
        if feat_i in features_to_drop:
            continue
        for idx_j in range(idx_i + 1, len(features_list)):
            feat_j = features_list[idx_j]
            if feat_j in features_to_drop:
                continue
                
            abs_corr = abs(corr_df.loc[feat_i, feat_j])
            if abs_corr > correlation_threshold:
                # Drop the feature with the lower MI score
                mi_i = mi_series.get(feat_i, 0.0)
                mi_j = mi_series.get(feat_j, 0.0)
                
                discard_feat = feat_j if mi_i >= mi_j else feat_i
                features_to_drop.add(discard_feat)
                print(f"  Pruning [{discard_feat}] (correlation = {abs_corr:.4f} with [{feat_i if discard_feat == feat_j else feat_j}])")
                
    surviving_features = [f for f in features_list if f not in features_to_drop]
    print(f"  Stage 2 complete: Dropped {len(features_to_drop)} collinear features.")
    print(f"  Surviving features: {len(surviving_features)}")
    
    return surviving_features

def run_stage3_shap(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, features: list) -> list:
    """
    Stage 3: LightGBM Regressor + SHAP value importance selection.
    Trains a LightGBM model, computes SHAP values on out-of-sample validation data, and ranks features.
    NOTE: Uses VALIDATION set (not test) to prevent data leakage.
    """
    print("\n[Feature Selection - Stage 3] Training LightGBM proxy model and running SHAP analysis...")
    
    X_tr = X_train[features]
    X_vl = X_val[features]
    
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=config.SEED,
        verbose=-1,
        n_jobs=-1
    )
    
    # Train proxy model
    model.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    print("  Model training completed. Calculating SHAP values on validation set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_vl)
    
    # Calculate global importance using mean absolute SHAP values
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=features
    ).sort_values(ascending=False)
    
    print("\n  Global Feature Importance (SHAP):")
    for feat, score in mean_abs_shap.items():
        print(f"    {feat:<45}: {score:.6f}")
        
    # Ensure a balanced feature representation (timing vs macro regime filters):
    # Select the top 4 hourly (1h_) features and the top 4 daily/4-hourly (1d_ or 4h_) features by SHAP importance.
    hourly_pool = [f for f in features if f.startswith("1h_")]
    macro_pool = [f for f in features if f.startswith("1d_") or f.startswith("4h_")]
    
    # Bug #8 fix: Use config.N_GOLDEN_FEATURES tuple instead of hardcoded counts
    n_hourly, n_macro = config.N_GOLDEN_FEATURES
    
    top_hourly = mean_abs_shap[mean_abs_shap.index.isin(hourly_pool)].head(n_hourly).index.tolist()
    top_macro = mean_abs_shap[mean_abs_shap.index.isin(macro_pool)].head(n_macro).index.tolist()
    
    golden_features = top_hourly + top_macro
    print(f"\n  Selected {len(golden_features)} Balanced Golden Features:")
    print("  Hourly Features (Timing):")
    for rank, feat in enumerate(top_hourly, 1):
        print(f"    {rank}. {feat:<40} (SHAP = {mean_abs_shap[feat]:.6f})")
    print("  Macro Features (Regime):")
    for rank, feat in enumerate(top_macro, 1):
        print(f"    {rank}. {feat:<40} (SHAP = {mean_abs_shap[feat]:.6f})")
        
    return golden_features

def execute_feature_selection(symbol: str):
    """
    Executes the entire feature selection workflow for a single coin.
    """
    print("=" * 60)
    print(f"Running Feature Selection for {symbol}")
    print("=" * 60)
    
    symbol_file = symbol.replace("/", "_")
    
    # Load scaled feature files
    train_path = os.path.join(config.DATA_DIR, f"{symbol_file}_train_features.csv")
    val_path = os.path.join(config.DATA_DIR, f"{symbol_file}_val_features.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Feature CSV files not found. Run phase1_feature_engineering.py first.")
        
    train_df = pd.read_csv(train_path, index_col="Date", parse_dates=True)
    val_df = pd.read_csv(val_path, index_col="Date", parse_dates=True)
    
    # Exclude non-feature columns
    feature_cols = [col for col in train_df.columns if col not in ("Close", "ATR")]
    
    # 0. Build Target (forward log-returns as proxy for supervised selection)
    y_train_raw = construct_target(train_df["Close"], config.FORWARD_K)
    y_val_raw = construct_target(val_df["Close"], config.FORWARD_K)
    
    # Align features and target (remove NaN rows at end)
    train_aligned = train_df[feature_cols].copy()
    train_aligned["__target"] = y_train_raw
    train_aligned.dropna(inplace=True)
    X_train = train_aligned[feature_cols]
    y_train = train_aligned["__target"]
    
    val_aligned = val_df[feature_cols].copy()
    val_aligned["__target"] = y_val_raw
    val_aligned.dropna(inplace=True)
    X_val = val_aligned[feature_cols]
    y_val = val_aligned["__target"]
    
    # Run Stage 1 (Mutual Information)
    surv_s1, mi_scores = run_stage1_mi(X_train, y_train, config.MI_DROP_BOTTOM_PCT)
    
    # Run Stage 2 (Correlation pruning)
    surv_s2 = run_stage2_correlation(X_train[surv_s1], mi_scores, config.CORR_THRESHOLD)
    
    # Run Stage 3 (SHAP value ranking) — uses VALIDATION set, NOT test
    golden_feats = run_stage3_shap(X_train, y_train, X_val, y_val, surv_s2)
    
    return golden_feats

def run_feature_selection_pipeline():
    """
    Main orchestration function for feature selection.
    """
    symbols = config.MULTI_COINS if config.MULTI_COIN_MODE else [config.SYMBOL]
    
    all_golden_features = []
    
    for symbol in symbols:
        golden = execute_feature_selection(symbol)
        all_golden_features.extend(f for f in golden if f not in all_golden_features)
    
    print("\n" + "=" * 60)
    print("Selected Golden Features (State Space)")
    print("=" * 60)
    for feat in all_golden_features:
         print(f"  - {feat}")
         
    # Save the selected list to json
    with open(config.GOLDEN_FEATURES_PATH, "w") as f:
        json.dump(all_golden_features, f, indent=4)
        
    print(f"\nSaved golden features list to {config.GOLDEN_FEATURES_PATH}")
    return all_golden_features

if __name__ == "__main__":
    run_feature_selection_pipeline()
