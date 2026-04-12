"""
Trading Analytics: enriches detected shifts with actionable metrics.
Called from the dashboard to compute metrics on-the-fly from price + shift data.
"""
import pandas as pd
import numpy as np


def enrich_shift(shift_row, price_df, window_bars=30):
    """
    Given a single shift row and the full price DataFrame,
    compute trading-relevant metrics around the shift.

    Returns a dict with all computed fields.
    """
    dt = pd.to_datetime(shift_row.get('datetime_utc', shift_row.name))
    price_df = price_df.copy()
    price_df['datetime_utc'] = pd.to_datetime(price_df['datetime_utc'])

    # Find the shift index in price data
    idx = price_df['datetime_utc'].searchsorted(dt)
    if idx >= len(price_df):
        idx = len(price_df) - 1

    # Define windows
    pre_start = max(0, idx - window_bars)
    post_end = min(len(price_df), idx + window_bars)

    pre = price_df.iloc[pre_start:idx]
    post = price_df.iloc[idx:post_end]
    shift_zone = price_df.iloc[max(0, idx - 3):min(len(price_df), idx + 3)]

    result = {'shift_datetime': str(dt)}

    # ── 1. Shift Start/End Timestamps ────────────────────────────────────
    if not pre.empty and not post.empty:
        result['shift_start'] = str(pre['datetime_utc'].iloc[-min(3, len(pre))])
        result['shift_end'] = str(post['datetime_utc'].iloc[min(2, len(post) - 1)])
        hours = (pd.to_datetime(result['shift_end']) - pd.to_datetime(result['shift_start'])).total_seconds() / 3600
        result['shift_duration_hours'] = round(hours, 1)
    else:
        result['shift_start'] = str(dt)
        result['shift_end'] = str(dt)
        result['shift_duration_hours'] = 0

    # ── 2. Price Change During Shift ─────────────────────────────────────
    if not pre.empty and not post.empty:
        price_before = pre['close'].iloc[-1]
        price_after = post['close'].iloc[min(5, len(post) - 1)]
        result['price_before'] = round(float(price_before), 5)
        result['price_after'] = round(float(price_after), 5)
        result['price_change_abs'] = round(float(price_after - price_before), 5)
        result['price_change_pct'] = round(float((price_after - price_before) / price_before * 100), 3)
    else:
        result['price_before'] = 0
        result['price_after'] = 0
        result['price_change_abs'] = 0
        result['price_change_pct'] = 0

    # ── 3. Volatility Measure ────────────────────────────────────────────
    if not pre.empty and not post.empty and len(pre) > 5 and len(post) > 5:
        pre_returns = pre['close'].pct_change().dropna()
        post_returns = post['close'].pct_change().dropna()

        pre_vol = float(pre_returns.std()) if len(pre_returns) > 1 else 0
        post_vol = float(post_returns.std()) if len(post_returns) > 1 else 0

        # ATR-style: average of high-low range
        if 'high' in pre.columns and 'low' in pre.columns:
            pre_atr = float((pre['high'] - pre['low']).mean())
            post_atr = float((post['high'] - post['low']).mean())
        else:
            pre_atr = pre_vol
            post_atr = post_vol

        result['pre_shift_volatility'] = round(pre_vol * 100, 4)  # as %
        result['post_shift_volatility'] = round(post_vol * 100, 4)
        result['volatility_change_pct'] = round((post_vol / pre_vol - 1) * 100, 1) if pre_vol > 0 else 0
        result['pre_shift_atr'] = round(pre_atr, 5)
        result['post_shift_atr'] = round(post_atr, 5)
    else:
        result['pre_shift_volatility'] = 0
        result['post_shift_volatility'] = 0
        result['volatility_change_pct'] = 0
        result['pre_shift_atr'] = 0
        result['post_shift_atr'] = 0

    # ── 4. Support/Resistance Levels ─────────────────────────────────────
    if not shift_zone.empty and 'high' in shift_zone.columns and 'low' in shift_zone.columns:
        result['resistance'] = round(float(shift_zone['high'].max()), 5)
        result['support'] = round(float(shift_zone['low'].min()), 5)
        result['range'] = round(result['resistance'] - result['support'], 5)
    elif not shift_zone.empty:
        result['resistance'] = round(float(shift_zone['close'].max()), 5)
        result['support'] = round(float(shift_zone['close'].min()), 5)
        result['range'] = round(result['resistance'] - result['support'], 5)
    else:
        result['resistance'] = 0
        result['support'] = 0
        result['range'] = 0

    # ── 5. Alert Threshold ───────────────────────────────────────────────
    severity = int(shift_row.get('severity', 1))
    ks_stat = float(shift_row.get('mean_ks_stat', 0) or 0)
    mmd = float(shift_row.get('mmd_score', 0) or 0)

    # Composite alert score: 0-100
    sev_score = severity * 20  # 0-100
    ks_score = min(ks_stat * 100, 100)  # 0-100
    mmd_score = min(mmd * 500, 100)  # 0-100
    alert_score = round((sev_score * 0.4 + ks_score * 0.3 + mmd_score * 0.3), 1)

    result['alert_score'] = alert_score
    if alert_score >= 80:
        result['alert_level'] = 'CRITICAL'
    elif alert_score >= 60:
        result['alert_level'] = 'HIGH'
    elif alert_score >= 40:
        result['alert_level'] = 'MODERATE'
    else:
        result['alert_level'] = 'LOW'

    # Threshold for similar future detection
    result['ks_threshold'] = round(ks_stat, 4)
    result['mmd_threshold'] = round(mmd, 6)

    return result
