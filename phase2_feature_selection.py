"""
phase2_feature_selection.py — 3-Stage Systematic Feature Selection Pipeline
============================================================================
Phase 2 of the pipeline — produces the "Golden State Space".

WHY WE DO THIS
--------------
The RL agent is sample-inefficient and extremely sensitive to noisy,
redundant, or non-informative features.  Having 80+ features (25-50
indicators × 3 timeframes) would cause:
    • Curse of dimensionality → poor generalisation
    • Correlated features → the LSTM wastes capacity on redundant information
    • Noisy features → the agent learns spurious correlations

We use a 3-stage supervised proxy approach to identify the 4–8 most
informative and orthogonal features WITHOUT ever touching the RL agent
(which would be far too slow to use as a feature selector).

THE 3 STAGES
------------
Stage 1 — Mutual Information Filter
    Measures non-linear statistical dependency between each feature and the
    forward log return (our proxy target).  Features with zero or near-zero
    MI carry no predictive signal and are dropped (bottom 30%).

Stage 2 — Spearman Correlation Filter
    Among the MI survivors, we remove redundant features that are near-
    perfectly correlated with each other (|ρ| > 0.80).  When two features are
    highly correlated, one is redundant.  We keep the one with the higher MI
    score.

Stage 3 — SHAP Proxy Model
    We train a LightGBM regressor on the remaining orthogonal features to
    predict the forward return.  SHAP (SHapley Additive exPlanations) gives us
    the global feature importances, revealing which features actually drive the
    model's predictions.  The top 4–8 SHAP-ranked features become our
    "Golden State Space".

OUTPUT
------
• Console printout of every surviving feature at each stage
• Plots saved to results/:
    - mi_scores.png         (Stage 1 bar chart)
    - correlation_heatmap.png (Stage 2 heatmap)
    - shap_beeswarm.png     (Stage 3 beeswarm)
    - shap_bar.png          (Stage 3 bar)
• golden_features.json      (list of final selected feature names)
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — saves plots to files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")   # suppress LightGBM / SHAP verbosity

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm not found. Install with: pip install lightgbm")

try:
    import shap
except ImportError:
    raise ImportError("shap not found. Install with: pip install shap")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — build supervised proxy target
# ──────────────────────────────────────────────────────────────────────────────

def build_target(close: pd.Series, k: int = config.FORWARD_K) -> pd.Series:
    """
    Creates the k-period forward log return:
        y_t = log(Close_{t+k} / Close_t)

    This is our supervised proxy target — it represents the raw reward signal
    that the RL agent is ultimately trying to maximise.

    We shift by -k so that y_t is aligned WITH the features at time t
    (i.e., at time t, we already know what y_t will be for the feature
    selection purposes — we do NOT shift the features; only the target).
    """
    log_ret = np.log(close.shift(-k) / (close + 1e-8))
    return log_ret   # last k rows will be NaN → dropped later


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — MUTUAL INFORMATION FILTER
# ──────────────────────────────────────────────────────────────────────────────

def stage1_mi_filter(X_train: pd.DataFrame, y_train: pd.Series,
                     drop_bottom_pct: float = config.MI_DROP_BOTTOM_PCT
                     ) -> tuple:
    """
    Calculates Mutual Information between each feature and the target.
    Drops the bottom `drop_bottom_pct` fraction of features.

    Parameters
    ----------
    X_train         : Feature matrix (training split)
    y_train         : Forward return target (training split)
    drop_bottom_pct : Fraction to drop (default 0.30 → drop bottom 30%)

    Returns
    -------
    (survivors: list[str], mi_series: pd.Series)
    """
    print("\n[Stage 1] Computing Mutual Information scores …")
    print(f"  Input features: {X_train.shape[1]}")

    # mutual_info_regression handles non-linear dependencies
    mi_scores = mutual_info_regression(X_train, y_train, random_state=config.SEED)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

    # Print top 20
    print("\n  Top 20 features by MI score:")
    for feat, score in mi_series.head(20).items():
        print(f"    {feat:<45} {score:.6f}")

    # ── Drop bottom `drop_bottom_pct` ──────────────────────────────────────
    n_drop     = int(len(mi_series) * drop_bottom_pct)
    threshold  = mi_series.iloc[-(n_drop + 1)]   # MI value at the cutoff
    survivors  = mi_series[mi_series > threshold].index.tolist()

    print(f"\n  Dropping bottom {drop_bottom_pct*100:.0f}% = {n_drop} features "
          f"(MI <= {threshold:.6f}).")
    print(f"  Survivors after Stage 1: {len(survivors)} features")

    # ── Plot horizontal bar chart ──────────────────────────────────────────
    _plot_mi(mi_series, n_drop, os.path.join(config.RESULTS_DIR, "mi_scores.png"))

    return survivors, mi_series


def _plot_mi(mi_series: pd.Series, n_drop: int, save_path: str):
    top_n  = min(40, len(mi_series))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors  = ["#e74c3c" if i >= (len(mi_series) - n_drop) else "#2ecc71"
               for i in range(len(mi_series))][:top_n][::-1]
    mi_plot = mi_series.head(top_n).iloc[::-1]
    ax.barh(mi_plot.index, mi_plot.values, color=colors[::-1])
    ax.set_title("Stage 1 — Mutual Information Scores (green = kept, red = dropped)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("MI Score")
    ax.axvline(mi_series.iloc[-(n_drop + 1)], color="red", linestyle="--",
               label=f"Drop threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] MI bar chart saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — SPEARMAN CORRELATION FILTER
# ──────────────────────────────────────────────────────────────────────────────

def stage2_correlation_filter(X_train: pd.DataFrame, survivors: list,
                               mi_series: pd.Series,
                               threshold: float = config.CORR_THRESHOLD) -> list:
    """
    Removes redundant features using Spearman rank correlation.

    Algorithm
    ---------
    1. Build the Spearman correlation matrix for all survivor features.
    2. Iterate over all pairs.  If |corr(i, j)| > threshold:
        - Drop the feature with the LOWER MI score (keep the more informative one).
    3. Repeat until no pair exceeds the threshold.

    Parameters
    ----------
    X_train   : Feature matrix
    survivors : Feature names that passed Stage 1
    mi_series : MI scores from Stage 1 (used to break ties)
    threshold : |correlation| above which one feature is dropped

    Returns
    -------
    list of surviving feature names
    """
    print(f"\n[Stage 2] Spearman correlation filter (threshold = {threshold}) …")
    print(f"  Input features: {len(survivors)}")

    X_surv = X_train[survivors]

    # Compute the Spearman correlation matrix
    corr_matrix, _ = spearmanr(X_surv)
    if len(survivors) == 1:
        corr_matrix = np.array([[1.0]])
    corr_df = pd.DataFrame(corr_matrix, index=survivors, columns=survivors)

    # ── Plot heatmap ──────────────────────────────────────────────────────
    _plot_corr_heatmap(corr_df, os.path.join(config.RESULTS_DIR, "correlation_heatmap.png"))

    # ── Greedy correlated feature removal ────────────────────────────────
    to_drop     = set()
    feat_list   = list(survivors)

    for i in range(len(feat_list)):
        if feat_list[i] in to_drop:
            continue
        for j in range(i + 1, len(feat_list)):
            if feat_list[j] in to_drop:
                continue
            corr_val = abs(corr_df.loc[feat_list[i], feat_list[j]])
            if corr_val > threshold:
                # Drop the one with the lower MI score
                mi_i = mi_series.get(feat_list[i], 0)
                mi_j = mi_series.get(feat_list[j], 0)
                loser = feat_list[j] if mi_i >= mi_j else feat_list[i]
                to_drop.add(loser)
                print(f"    Drop [{loser}] (corr={corr_val:.3f} with "
                      f"{'['+ feat_list[i] +']' if loser==feat_list[j] else '['+ feat_list[j] +']'})")

    final = [f for f in survivors if f not in to_drop]
    print(f"\n  Dropped {len(to_drop)} correlated features.")
    print(f"  Survivors after Stage 2: {len(final)} features")
    print(f"  Features: {final}")
    return final


def _plot_corr_heatmap(corr_df: pd.DataFrame, save_path: str):
    size   = max(8, len(corr_df) * 0.4)
    fig, ax = plt.subplots(figsize=(size, size))
    mask   = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=len(corr_df) <= 20, fmt=".2f", linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Stage 2 — Spearman Rank Correlation Matrix",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Correlation heatmap saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — LIGHTGBM PROXY MODEL + SHAP
# ──────────────────────────────────────────────────────────────────────────────

def stage3_shap_selection(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test:  pd.DataFrame, y_test:  pd.Series,
                           ortho_features: list,
                           n_top: tuple = config.N_GOLDEN_FEATURES) -> list:
    """
    Trains a LightGBM regressor on the orthogonal features and uses SHAP to
    rank them.  Selects the top n features as the Golden State Space.

    Parameters
    ----------
    X_train, y_train : Training split features and target
    X_test,  y_test  : Test split features and target (never trained on)
    ortho_features   : Features that survived Stages 1 & 2
    n_top            : (min, max) number of features to select

    Returns
    -------
    list of selected feature names (the "Golden State Space")
    """
    print(f"\n[Stage 3] Training LightGBM proxy model …")
    print(f"  Input features: {len(ortho_features)}")

    X_tr  = X_train[ortho_features]
    X_te  = X_test[ortho_features]

    # ── Train a fast gradient boosting regressor ──────────────────────────
    model = lgb.LGBMRegressor(
        n_estimators    = 500,
        learning_rate   = 0.05,
        max_depth       = 6,
        num_leaves      = 31,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        random_state    = config.SEED,
        n_jobs          = -1,
        verbose         = -1,
    )
    model.fit(
        X_tr, y_train,
        eval_set         = [(X_te, y_test)],
        callbacks        = [lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(period=-1)],
    )

    # ── Compute SHAP values on the TEST set (never used in training) ──────
    print("  Computing SHAP values on test set …")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_te)

    # Mean absolute SHAP = global feature importance
    mean_abs_shap = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=ortho_features
    ).sort_values(ascending=False)

    print("\n  SHAP Global Feature Importance:")
    for feat, shap_imp in mean_abs_shap.items():
        print(f"    {feat:<45} {shap_imp:.6f}")

    # ── SHAP Plots ─────────────────────────────────────────────────────────
    _plot_shap_beeswarm(explainer, X_te,
                        os.path.join(config.RESULTS_DIR, "shap_beeswarm.png"))
    _plot_shap_bar(mean_abs_shap,
                   os.path.join(config.RESULTS_DIR, "shap_bar.png"))

    # ── Select top features ────────────────────────────────────────────────
    n_select    = min(n_top[1], max(n_top[0], len(ortho_features)))
    golden_feats = mean_abs_shap.head(n_select).index.tolist()

    print(f"\n  Selected top {n_select} features (Golden State Space):")
    for i, f in enumerate(golden_feats, 1):
        print(f"    {i}. {f}  (SHAP={mean_abs_shap[f]:.6f})")

    return golden_feats


def _plot_shap_beeswarm(explainer, X_te: pd.DataFrame, save_path: str):
    shap_vals = explainer.shap_values(X_te.sample(min(500, len(X_te)),
                                                   random_state=42))
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_te.sample(min(500, len(X_te)),
                                              random_state=42),
                      show=False, max_display=20)
    plt.title("Stage 3 — SHAP Beeswarm (Feature Impact Direction & Magnitude)",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] SHAP beeswarm saved to {save_path}")


def _plot_shap_bar(mean_abs_shap: pd.Series, save_path: str):
    top20 = mean_abs_shap.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, len(top20) * 0.35)))
    ax.barh(top20.index, top20.values, color="#3498db")
    ax.set_title("Stage 3 — Mean |SHAP| Feature Importance",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("|SHAP| value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] SHAP bar chart saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────


def run_feature_selection(train_feat: pd.DataFrame,
                           val_feat:   pd.DataFrame,
                           test_feat:  pd.DataFrame,
                           feature_cols: list,
                           coin_name: str = "") -> list:
    """
    Executes all three stages of the feature selection pipeline.

    Parameters
    ----------
    train_feat, val_feat, test_feat : Scaled feature DataFrames from Phase 1
    feature_cols                    : List of feature column names (excl. Close/ATR)
    coin_name                       : Name of the coin being processed (for logging)

    Returns
    -------
    golden_features : list[str] — the final selected feature names
    """
    print("\n" + "=" * 70)
    print(f"  PHASE 2 — FEATURE SELECTION for {coin_name}")
    print("=" * 70)

    # ── Build the supervised proxy target ─────────────────────────────────
    print("\n[Step 0] Building forward log-return target …")
    # Use un-scaled Close price for accurate log-return calculation
    train_close = train_feat["Close"]
    test_close  = test_feat["Close"]

    y_train_raw = build_target(train_close, k=config.FORWARD_K)
    y_test_raw  = build_target(test_close,  k=config.FORWARD_K)

    # Align features and target (drop NaN rows from the forward shift)
    train_aligned = train_feat[feature_cols].copy()
    train_aligned["__y"] = y_train_raw.values
    train_aligned.dropna(inplace=True)
    X_train = train_aligned.drop(columns=["__y"])
    y_train = train_aligned["__y"]

    test_aligned = test_feat[feature_cols].copy()
    test_aligned["__y"] = y_test_raw.values
    test_aligned.dropna(inplace=True)
    X_test  = test_aligned.drop(columns=["__y"])
    y_test  = test_aligned["__y"]

    print(f"  Train target: {len(y_train):,} rows, "
          f"mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    # ── Stage 1: MI Filter ────────────────────────────────────────────────
    survivors_s1, mi_series = stage1_mi_filter(X_train, y_train)

    # ── Stage 2: Correlation Filter ───────────────────────────────────────
    survivors_s2 = stage2_correlation_filter(X_train, survivors_s1, mi_series)

    # ── Stage 3: SHAP Proxy Model ─────────────────────────────────────────
    golden_features = stage3_shap_selection(
        X_train, y_train,
        X_test,  y_test,
        ortho_features = survivors_s2,
    )

    print("\n" + "=" * 70)
    print(f"  [GOLDEN] FEATURES FOR {coin_name}")
    print("=" * 70)
    for i, f in enumerate(golden_features, 1):
        print(f"  {i:2d}. {f}")
    print("=" * 70)

    return golden_features


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    symbols_to_process = config.MULTI_COINS if config.MULTI_COIN_MODE else [config.SYMBOL]
    
    all_selected_features = set()
    coin_features_map = {}

    for symbol in symbols_to_process:
        symbol_file = symbol.replace("/", "_")
        t_path = os.path.join(config.DATA_DIR, f"{symbol_file}_train_features.csv")
        v_path = os.path.join(config.DATA_DIR, f"{symbol_file}_val_features.csv")
        te_path = os.path.join(config.DATA_DIR, f"{symbol_file}_test_features.csv")
        
        # Fallback to single coin mode paths if explicit coin paths don't exist
        if not os.path.exists(t_path) and symbol == config.SYMBOL:
            t_path = config.TRAIN_FEAT_PATH
            v_path = config.VAL_FEAT_PATH
            te_path = config.TEST_FEAT_PATH
            
        if not os.path.exists(t_path):
            print(f"⚠️ Warning: Could not find feature files for {symbol} at {t_path}. Skipping.")
            continue

        print(f"\nLoading feature CSVs for {symbol} …")
        train_feat = pd.read_csv(t_path, index_col=0, parse_dates=True)
        val_feat   = pd.read_csv(v_path,   index_col=0, parse_dates=True)
        test_feat  = pd.read_csv(te_path,  index_col=0, parse_dates=True)

        feature_cols = [c for c in train_feat.columns if c not in ("Close", "ATR", "Turbulence")]

        golden = run_feature_selection(train_feat, val_feat, test_feat, feature_cols, coin_name=symbol)
        coin_features_map[symbol] = golden
        
        for f in golden:
            all_selected_features.add(f)

    # ── Compute Union ──────────────────────────────────────────────────────
    union_features = list(all_selected_features)
    
    print("\n" + "=" * 70)
    print("  [UNION] MULTI-COIN FEATURE UNION (COMBINED STATE SPACE)")
    print("=" * 70)
    for sym, feats in coin_features_map.items():
        print(f"  {sym:10s} : {len(feats)} features")
        
    print(f"\nFinal Unified Golden State Space ({len(union_features)} features):")
    for f in union_features:
        print(f"  • {f}")

    # ── Save Union to JSON ─────────────────────────────────────────────────
    with open(config.GOLDEN_FEATURES_PATH, "w") as fp:
        json.dump(union_features, fp, indent=2)
    print(f"\n  [Save] Unified Golden features saved to {config.GOLDEN_FEATURES_PATH}")
    print("\nPhase 2 standalone test PASSED")
