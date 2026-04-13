"""
Strong technical baseline for ShiftGuard.
"""
import numpy as np
import pandas as pd


def generate_technical_signals(df):
    signals = []

    for _, row in df.iterrows():
        score = 0

        ema_12 = row.get('ema_12')
        ema_26 = row.get('ema_26')
        sma_20 = row.get('sma_20')
        sma_50 = row.get('sma_50')
        rsi = row.get('rsi_14')
        macd = row.get('macd_hist')
        bb_pctb = row.get('bb_pctb')
        atr_pct = row.get('atr_pct_short')
        bar_return = row.get('bar_return')

        if any(pd.isna(v) for v in [ema_12, ema_26, sma_20, sma_50, rsi, macd]):
            signals.append(0)
            continue

        if ema_12 > ema_26:
            score += 1
        elif ema_12 < ema_26:
            score -= 1

        if sma_20 > sma_50:
            score += 1
        elif sma_20 < sma_50:
            score -= 1

        if rsi < 35:
            score += 1
        elif rsi > 65:
            score -= 1

        if macd > 0:
            score += 1
        elif macd < 0:
            score -= 1

        if pd.notna(bb_pctb):
            if bb_pctb < 0.2:
                score += 1
            elif bb_pctb > 0.8:
                score -= 1

        if pd.notna(bar_return):
            if bar_return > 0:
                score += 0.5
            elif bar_return < 0:
                score -= 0.5

        if pd.notna(atr_pct) and atr_pct > 95:
            signals.append(0)
            continue

        if score >= 2:
            signals.append(1)
        elif score <= -2:
            signals.append(-1)
        else:
            signals.append(0)

    return np.array(signals, dtype=float)
