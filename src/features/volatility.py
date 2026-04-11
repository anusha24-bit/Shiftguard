"""
Group 2: Volatility Features
Advanced regime-level volatility estimators from 4H OHLCV.
"""
import numpy as np


def compute_volatility_features(df):
    """
    Input: df with columns [open, high, low, close] + log_return (from Group 1)
    Output: df with volatility feature columns appended
    """
    h, l, c, o = df['high'], df['low'], df['close'], df['open']

    # Garman-Klass volatility (uses OHLC, more efficient than close-to-close)
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    df['gk_vol'] = np.sqrt(
        (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(20).mean()
    )

    # Parkinson volatility (High-Low based)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(h / l)**2).rolling(20).mean()
    )

    # Rolling standard deviation of returns (multiple windows)
    df['rolling_std_5'] = df['log_return'].rolling(5).std()
    df['rolling_std_20'] = df['log_return'].rolling(20).std()
    df['rolling_std_60'] = df['log_return'].rolling(60).std()

    # Volatility of volatility (second-order)
    df['vol_of_vol'] = df['atr_14'].rolling(20).std()

    # Volatility ratio (short-term / long-term) — spike = regime change
    df['vol_ratio'] = df['rolling_std_5'] / df['rolling_std_60'].replace(0, np.nan)

    # Max drawdown (rolling 20-bar)
    rolling_max = c.rolling(20).max()
    df['drawdown_20'] = (c - rolling_max) / rolling_max

    return df
