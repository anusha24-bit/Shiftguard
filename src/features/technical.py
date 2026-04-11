"""
Group 1: Technical Indicators
Computed from 4H OHLCV bars using the `ta` library.
"""
import numpy as np
import ta


def compute_technical_features(df):
    """
    Input: df with columns [open, high, low, close] (4H bars)
    Output: df with technical indicator columns appended
    """
    c, h, l, o = df['close'], df['high'], df['low'], df['open']

    # --- Trend ---
    df['sma_20'] = ta.trend.sma_indicator(c, window=20)
    df['sma_50'] = ta.trend.sma_indicator(c, window=50)
    df['ema_12'] = ta.trend.ema_indicator(c, window=12)
    df['ema_26'] = ta.trend.ema_indicator(c, window=26)

    macd = ta.trend.MACD(c)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    df['adx'] = ta.trend.adx(h, l, c, window=14)
    df['plus_di'] = ta.trend.adx_pos(h, l, c, window=14)
    df['minus_di'] = ta.trend.adx_neg(h, l, c, window=14)

    ichimoku = ta.trend.IchimokuIndicator(h, l)
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()

    # --- Momentum ---
    df['rsi_14'] = ta.momentum.rsi(c, window=14)

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['williams_r'] = ta.momentum.williams_r(h, l, c)
    df['cci_20'] = ta.trend.cci(h, l, c, window=20)
    df['roc_10'] = ta.momentum.roc(c, window=10)

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(c)
    df['bb_width'] = bb.bollinger_wband()
    df['bb_pctb'] = bb.bollinger_pband()

    # --- ATR ---
    df['atr_14'] = ta.volatility.average_true_range(h, l, c, window=14)

    # --- Keltner ---
    kc = ta.volatility.KeltnerChannel(h, l, c)
    df['kc_upper'] = kc.keltner_channel_hband()
    df['kc_lower'] = kc.keltner_channel_lband()

    # --- Derived price features ---
    df['bar_return'] = c.pct_change()
    df['log_return'] = np.log(c / c.shift(1))
    df['hl_range'] = (h - l) / c
    df['co_range'] = (c - o) / o
    df['gap'] = (o - c.shift(1)) / c.shift(1)

    # --- Session as categorical (one-hot encode) ---
    if 'session' in df.columns:
        session_dummies = df['session'].str.get_dummies()
        session_dummies.columns = ['sess_' + col for col in session_dummies.columns]
        df = df.join(session_dummies)

    return df
