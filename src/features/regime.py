"""
Group 5: Regime Identification & Transition Features
Computes adaptive volatility regime labels + features that detect regime changes.
"""
import numpy as np
import pandas as pd


def hurst_exponent(series, max_lag=20):
    """
    Estimate Hurst exponent using rescaled range (R/S) method.
    H < 0.5: mean-reverting regime
    H = 0.5: random walk
    H > 0.5: trending regime
    """
    series = series.dropna().values
    if len(series) < max_lag * 2:
        return 0.5

    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, len(series) - lag, lag)]
        rs_values = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_values.append(R / S)
        if rs_values:
            tau.append(np.mean(rs_values))
        else:
            tau.append(1)

    if len(tau) < 2:
        return 0.5

    lags_log = np.log(list(lags)[:len(tau)])
    tau_log = np.log(np.array(tau) + 1e-10)

    try:
        poly = np.polyfit(lags_log, tau_log, 1)
        return max(0, min(1, poly[0]))
    except:
        return 0.5


def compute_regime_features(df):
    """
    Add regime identification and transition features.
    df must have: close, high, low, open, atr_14, log_return, hl_range, vix (if available)
    """
    df = df.copy()
    c, h, l = df['close'], df['high'], df['low']

    # --- Adaptive ATR percentiles ---
    # Short-term: percentile vs last 60 bars (local context)
    df['atr_pct_short'] = df['atr_14'].rolling(60, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    # Long-term: percentile vs last 252 bars (historical context)
    df['atr_pct_long'] = df['atr_14'].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # --- Realized vol ratio (regime change detector) ---
    df['vol_ratio_5_60'] = df['log_return'].rolling(5).std() / df['log_return'].rolling(60).std().replace(0, np.nan)
    df['vol_ratio_5_20'] = df['log_return'].rolling(5).std() / df['log_return'].rolling(20).std().replace(0, np.nan)

    # --- Vol compression (squeeze detection) ---
    atr_avg = df['atr_14'].rolling(20).mean()
    df['vol_compressed'] = (df['atr_14'] < atr_avg * 0.7).astype(int)
    # Duration of compression
    compressed = df['vol_compressed'].values
    duration = np.zeros(len(compressed))
    for i in range(1, len(compressed)):
        if compressed[i] == 1:
            duration[i] = duration[i-1] + 1
    df['compression_duration'] = duration

    # --- Range contraction ratio ---
    avg_range = df['hl_range'].rolling(20).mean()
    df['range_contraction'] = df['hl_range'] / avg_range.replace(0, np.nan)

    # --- Hurst exponent (rolling, trending vs mean-reverting) ---
    hurst_values = []
    returns = df['log_return'].fillna(0).values
    for i in range(len(returns)):
        if i < 60:
            hurst_values.append(0.5)
        else:
            h = hurst_exponent(pd.Series(returns[i-60:i]), max_lag=15)
            hurst_values.append(h)
    df['hurst_exponent'] = hurst_values

    # --- Consecutive directional bars ---
    direction = np.sign(df['log_return'].fillna(0)).values
    consec = np.zeros(len(direction))
    for i in range(1, len(direction)):
        if direction[i] == direction[i-1] and direction[i] != 0:
            consec[i] = consec[i-1] + 1
    df['consecutive_dir_bars'] = consec

    # --- Cross-asset vol divergence (if VIX available) ---
    if 'vix' in df.columns:
        vix_change_norm = df['vix'].pct_change().rolling(10).mean()
        pair_vol_change = df['atr_14'].pct_change().rolling(10).mean()
        df['vol_divergence'] = vix_change_norm - pair_vol_change

    # --- Event proximity × volatility interaction ---
    if 'days_to_next_event' in df.columns:
        df['event_vol_interaction'] = df['days_to_next_event'].fillna(30) * df['atr_pct_short'].fillna(50) / 100

    # --- Bar range expansion (volume proxy) ---
    df['range_expansion'] = df['hl_range'] / df['hl_range'].rolling(20).mean().replace(0, np.nan)

    # --- Gap size as flow indicator ---
    df['abs_gap'] = np.abs(df['open'] - c.shift(1)) / c.shift(1)
    df['gap_expansion'] = df['abs_gap'] / df['abs_gap'].rolling(20).mean().replace(0, np.nan)

    return df


def compute_adaptive_regime_labels(df, short_window=60):
    """
    Create adaptive volatility regime labels using short-window percentiles.
    Labels: 0=LOW, 1=MEDIUM, 2=HIGH
    """
    df = df.copy()

    # Use short-term ATR percentile for adaptive labeling
    df['regime_label'] = pd.cut(
        df['atr_pct_short'],
        bins=[-1, 33, 66, 101],
        labels=[0, 1, 2]
    ).astype(float)

    # Forward-fill any NaN from warmup period
    df['regime_label'] = df['regime_label'].ffill().fillna(1).astype(int)

    # Also create the NEXT bar's regime as prediction target
    df['target_regime'] = df['regime_label'].shift(-1)
    df['target_regime'] = df['target_regime'].ffill().astype(int)

    # Regime change flag (useful feature + ground truth for detection validation)
    df['regime_changed'] = (df['regime_label'] != df['regime_label'].shift(1)).astype(int)

    # TARGET: Will regime change in the next N bars? (transition prediction)
    for horizon in [6, 12, 18]:  # 6=1day, 12=2days, 18=3days
        future_labels = df['regime_label'].shift(-horizon)
        df[f'target_transition_{horizon}'] = (df['regime_label'] != future_labels).astype(float)
        df[f'target_transition_{horizon}'] = df[f'target_transition_{horizon}'].fillna(0).astype(int)

    # Bars since last regime change
    changed = df['regime_changed'].values
    bars_since = np.zeros(len(changed))
    for i in range(1, len(changed)):
        if changed[i] == 1:
            bars_since[i] = 0
        else:
            bars_since[i] = bars_since[i-1] + 1
    df['bars_since_regime_change'] = bars_since

    return df
