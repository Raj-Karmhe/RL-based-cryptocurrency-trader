import numpy as np
import pandas as pd
import os
import sys
from scipy import stats

# ==============================================================================
# PAIR SELECTION & COINTEGRATION VALIDATION
# Statistical tests to confirm the chosen pair is actually mean-reverting
# and suitable for pairs trading. If tests fail, the strategy has no edge.
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def engle_granger_cointegration(price_a: pd.Series, price_b: pd.Series):
    """
    Engle-Granger two-step cointegration test.

    Step 1: Run OLS regression  log(A) = α + β * log(B) + ε
    Step 2: Test residuals (spread) for stationarity using ADF

    Returns:
        dict with keys: hedge_ratio, intercept, adf_stat, pvalue, critical_values, is_cointegrated
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    log_a = np.log(price_a).values
    log_b = np.log(price_b).values

    # Step 1: OLS regression to find hedge ratio
    X = add_constant(log_b)
    model = OLS(log_a, X).fit()
    intercept = model.params[0]
    hedge_ratio = model.params[1]

    # Step 2: Test residuals for stationarity
    residuals = log_a - (intercept + hedge_ratio * log_b)
    adf_result = adfuller(residuals, maxlag=20, regression='c', autolag='AIC')

    adf_stat = adf_result[0]
    pvalue = adf_result[1]
    critical_values = adf_result[4]

    return {
        'hedge_ratio': hedge_ratio,
        'intercept': intercept,
        'adf_stat': adf_stat,
        'pvalue': pvalue,
        'critical_values': critical_values,
        'is_cointegrated': pvalue < config.COINT_PVALUE_THRESHOLD,
        'residuals': residuals,
    }


def johansen_cointegration(price_a: pd.Series, price_b: pd.Series, det_order: int = 0, k_ar_diff: int = 1):
    """
    Johansen cointegration test — multivariate test for cointegrating vectors.
    More robust than Engle-Granger for pairs with unclear causal direction.

    Returns:
        dict with trace_stat, critical_values_95, n_cointegrating_vectors
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    log_prices = np.column_stack([np.log(price_a.values), np.log(price_b.values)])
    result = coint_johansen(log_prices, det_order=det_order, k_ar_diff=k_ar_diff)

    # Trace statistic test at 95% confidence
    trace_stats = result.lr1                    # Trace statistics
    crit_95 = result.cvt[:, 1]                  # 95% critical values

    n_coint = int(np.sum(trace_stats > crit_95))

    return {
        'trace_stats': trace_stats,
        'critical_values_95': crit_95,
        'n_cointegrating_vectors': n_coint,
        'is_cointegrated': n_coint >= 1,
        'eigenvectors': result.evec,
    }


def adf_test_spread(spread: np.ndarray):
    """
    Augmented Dickey-Fuller test directly on the spread to confirm stationarity.

    Returns:
        dict with adf_stat, pvalue, is_stationary
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(spread, maxlag=20, regression='c', autolag='AIC')

    return {
        'adf_stat': result[0],
        'pvalue': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < config.ADF_PVALUE_THRESHOLD,
    }


def compute_half_life(spread: np.ndarray) -> float:
    """
    Estimates the half-life of mean reversion using an Ornstein-Uhlenbeck model.

    Model: Δspread_t = θ * (spread_{t-1} - μ) + ε
    Half-life = -ln(2) / θ

    A shorter half-life means faster mean reversion = better for trading.

    Returns:
        float: half-life in periods (hours). Returns np.inf if not mean-reverting.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    # OLS: Δspread = α + θ * spread_lag
    X = add_constant(spread_lag)
    model = OLS(spread_diff, X).fit()

    theta = model.params[1]

    if theta >= 0:
        # Not mean-reverting (spread is diverging or random walk)
        return np.inf

    half_life = -np.log(2) / theta
    return float(half_life)


def compute_hurst_exponent(spread: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimates the Hurst exponent using the rescaled range (R/S) method.

    H < 0.5: Mean-reverting (good for pairs trading)
    H = 0.5: Random walk
    H > 0.5: Trending

    Returns:
        float: Hurst exponent
    """
    lags = range(2, min(max_lag, len(spread) // 4))
    tau = []
    rs_values = []

    for lag in lags:
        # Compute the variance of lagged differences
        diffs = spread[lag:] - spread[:-lag]
        std_diff = np.std(diffs)
        if std_diff > 0:
            tau.append(lag)
            rs_values.append(std_diff)

    if len(tau) < 10:
        return 0.5  # Not enough data, assume random walk

    # Log-log regression: log(R/S) = H * log(tau) + c
    log_tau = np.log(tau)
    log_rs = np.log(rs_values)

    slope, _, _, _, _ = stats.linregress(log_tau, log_rs)

    return float(slope)


def rolling_cointegration_check(price_a: pd.Series, price_b: pd.Series,
                                 window: int = None):
    """
    Tests whether the cointegration relationship is stable over time
    by running the Engle-Granger test on rolling windows.

    Returns:
        dict with pvalues (series), pass_rate, is_stable
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    if window is None:
        window = config.ROLLING_COINT_WINDOW

    log_a = np.log(price_a.values)
    log_b = np.log(price_b.values)
    n = len(log_a)

    pvalues = []
    timestamps = []

    for i in range(0, n - window + 1, window // 4):  # Start from 0, step by 1/4 window
        end = i + window
        if end > n:
            break

        la = log_a[i:end]
        lb = log_b[i:end]

        X = add_constant(lb)
        model = OLS(la, X).fit()
        residuals = la - model.predict(X)

        try:
            adf_result = adfuller(residuals, maxlag=10, regression='c', autolag='AIC')
            pvalues.append(adf_result[1])
        except Exception:
            pvalues.append(1.0)

        if i < len(price_a.index):
            timestamps.append(price_a.index[i])

    pvalues = np.array(pvalues)
    pass_rate = float(np.mean(pvalues < config.COINT_PVALUE_THRESHOLD))

    return {
        'pvalues': pvalues,
        'pass_rate': pass_rate,
        'is_stable': pass_rate >= config.ROLLING_COINT_MIN_PASS,
        'n_windows': len(pvalues),
    }


def run_full_validation(price_a: pd.Series, price_b: pd.Series) -> dict:
    """
    Runs the complete battery of cointegration and mean-reversion tests.

    Returns:
        dict with all test results and an overall pass/fail verdict
    """
    results = {}
    all_pass = True

    # ── Test 1: Engle-Granger Cointegration ───────────────────────────────
    print("\n  Test 1: Engle-Granger Cointegration")
    eg = engle_granger_cointegration(price_a, price_b)
    results['engle_granger'] = eg
    status = "[PASS]" if eg['is_cointegrated'] else "[FAIL]"
    all_pass = all_pass and eg['is_cointegrated']
    print(f"    ADF Stat:    {eg['adf_stat']:.4f}")
    print(f"    P-Value:     {eg['pvalue']:.6f}")
    print(f"    Hedge Ratio: {eg['hedge_ratio']:.4f}")
    print(f"    Result:      {status}")

    # ── Test 2: Johansen Cointegration ────────────────────────────────────
    print("\n  Test 2: Johansen Cointegration")
    joh = johansen_cointegration(price_a, price_b)
    results['johansen'] = joh
    status = "[PASS]" if joh['is_cointegrated'] else "[FAIL]"
    all_pass = all_pass and joh['is_cointegrated']
    print(f"    Trace Stats:     {joh['trace_stats']}")
    print(f"    Critical (95%):  {joh['critical_values_95']}")
    print(f"    Coint Vectors:   {joh['n_cointegrating_vectors']}")
    print(f"    Result:          {status}")

    # ── Test 3: ADF on Spread ─────────────────────────────────────────────
    print("\n  Test 3: ADF Test on Spread")
    spread = eg['residuals']
    adf = adf_test_spread(spread)
    results['adf_spread'] = adf
    status = "[PASS]" if adf['is_stationary'] else "[FAIL]"
    all_pass = all_pass and adf['is_stationary']
    print(f"    ADF Stat:    {adf['adf_stat']:.4f}")
    print(f"    P-Value:     {adf['pvalue']:.6f}")
    print(f"    Result:      {status}")

    # ── Test 4: Half-Life of Mean Reversion ───────────────────────────────
    print("\n  Test 4: Half-Life of Mean Reversion")
    hl = compute_half_life(spread)
    results['half_life'] = hl
    is_good_hl = 1 < hl < 500  # Too fast or too slow = bad
    status = "[PASS]" if is_good_hl else "[FAIL]"
    all_pass = all_pass and is_good_hl
    print(f"    Half-Life:   {hl:.1f} hours ({hl/24:.1f} days)")
    print(f"    Result:      {status}")

    # ── Test 5: Hurst Exponent ────────────────────────────────────────────
    print("\n  Test 5: Hurst Exponent")
    hurst = compute_hurst_exponent(spread)
    results['hurst'] = hurst
    is_mean_reverting = hurst < config.HURST_THRESHOLD
    status = "[PASS]" if is_mean_reverting else "[FAIL]"
    all_pass = all_pass and is_mean_reverting
    print(f"    Hurst:       {hurst:.4f}")
    print(f"    Interpret:   {'Mean-Reverting' if hurst < 0.5 else 'Trending' if hurst > 0.5 else 'Random Walk'}")
    print(f"    Result:      {status}")

    # ── Test 6: Rolling Cointegration Stability ───────────────────────────
    print("\n  Test 6: Rolling Cointegration Stability")
    rolling = rolling_cointegration_check(price_a, price_b)
    results['rolling_coint'] = rolling
    status = "[PASS]" if rolling['is_stable'] else "[FAIL]"
    all_pass = all_pass and rolling['is_stable']
    print(f"    Windows:     {rolling['n_windows']}")
    print(f"    Pass Rate:   {rolling['pass_rate']*100:.1f}%")
    print(f"    Result:      {status}")

    # ── Overall Verdict ───────────────────────────────────────────────────
    results['overall_pass'] = all_pass
    print(f"\n  {'='*50}")
    if all_pass:
        print(f"  OVERALL VERDICT: [PASS] ALL TESTS PASSED")
        print(f"  The pair {config.ASSET_A_LABEL}/{config.ASSET_B_LABEL} is suitable for pairs trading.")
    else:
        print(f"  OVERALL VERDICT: [FAIL] SOME TESTS FAILED")
        print(f"  WARNING: Proceed with caution — the pair may not be reliably mean-reverting.")
    print(f"  {'='*50}")

    return results


if __name__ == '__main__':
    from download_data import download_and_align_pair

    print("=" * 70)
    print("  PAIR VALIDATION: COINTEGRATION & MEAN-REVERSION TESTS")
    print(f"  Pair: {config.ASSET_A_LABEL} / {config.ASSET_B_LABEL}")
    print("=" * 70)

    df_a, df_b, df_merged = download_and_align_pair()

    results = run_full_validation(df_a['Close'], df_b['Close'])
