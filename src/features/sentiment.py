"""
Group 4: Sentiment & Cross-Asset Features
VIX, DXY, S&P 500, oil, cross-asset correlations, news volume, gold-specific factors.
Daily data forward-filled onto 4H bars by date.
"""
import pandas as pd
import numpy as np


def compute_sentiment_features(df, pair_name, data_dir):
    """
    Merge sentiment/cross-asset features onto 4H bars by date.
    df must have columns: date, log_return
    """
    df = df.copy()

    # --- Load VIX/DXY/S&P/Oil ---
    sent = pd.read_csv(f'{data_dir}/sentiment/vix_dxy_daily.csv', parse_dates=['date'])
    sent['date'] = sent['date'].dt.date.astype(str)
    sent = sent.sort_values('date')

    # Rename to match expected columns
    col_map = {}
    for col in sent.columns:
        cl = col.lower()
        if 'vix' in cl:
            col_map[col] = 'vix'
        elif 'dxy' in cl:
            col_map[col] = 'dxy'
        elif 'sp500' in cl or 'sp_500' in cl:
            col_map[col] = 'sp500'
        elif 'oil' in cl:
            col_map[col] = 'oil'
    sent = sent.rename(columns=col_map)

    # Compute daily changes before merging
    sent['vix_change'] = sent['vix'].pct_change(fill_method=None)
    sent['dxy_change'] = sent['dxy'].pct_change(fill_method=None)
    sent['sp500_return'] = sent['sp500'].pct_change(fill_method=None)
    sent['oil_return'] = sent['oil'].pct_change(fill_method=None)
    sent['vix_sma_10'] = sent['vix'].rolling(10).mean()
    sent['vix_above_avg'] = (sent['vix'] > sent['vix_sma_10']).astype(int)

    # Select columns to merge
    sent_cols = ['date', 'vix', 'vix_change', 'vix_above_avg', 'dxy', 'dxy_change',
                 'sp500_return', 'oil_return']
    sent_merge = sent[[c for c in sent_cols if c in sent.columns]].copy()

    df = df.merge(sent_merge, on='date', how='left')

    # Forward-fill sentiment values (weekends)
    for col in sent_merge.columns:
        if col != 'date':
            df[col] = df[col].ffill()

    # --- Cross-asset correlations (rolling 20 bars ≈ ~3 days) ---
    if 'sp500_return' in df.columns and 'log_return' in df.columns:
        df['corr_with_sp500'] = df['log_return'].rolling(120).corr(df['sp500_return'])  # 120 bars ≈ 20 days
    if 'dxy_change' in df.columns and 'log_return' in df.columns:
        df['corr_with_dxy'] = df['log_return'].rolling(120).corr(df['dxy_change'])

    # --- News volume ---
    try:
        news = pd.read_csv(f'{data_dir}/sentiment/news_volume_daily.csv', parse_dates=['date'])
        news['date'] = news['date'].dt.date.astype(str)

        # Pick the relevant currency news volume
        news_col_map = {
            'EURUSD': ['usd_news_volume', 'eur_news_volume'],
            'GBPJPY': ['gbp_news_volume', 'jpy_news_volume'],
            'XAUUSD': ['usd_news_volume'],
        }
        cols = news_col_map.get(pair_name, ['usd_news_volume'])
        available_cols = [c for c in cols if c in news.columns]

        if available_cols:
            news['news_volume'] = news[available_cols].sum(axis=1)
            news_merge = news[['date', 'news_volume']].copy()
            df = df.merge(news_merge, on='date', how='left')
            df['news_volume'] = df['news_volume'].ffill().fillna(0)
            df['news_spike'] = (
                df['news_volume'] > df['news_volume'].rolling(180).mean() +
                2 * df['news_volume'].rolling(180).std()
            ).astype(int)
    except FileNotFoundError:
        pass

    # --- Gold-specific factors (XAU/USD only) ---
    if pair_name == 'XAUUSD':
        try:
            gold = pd.read_csv(f'{data_dir}/sentiment/gold_specific_factors.csv', parse_dates=['date'])
            gold['date'] = gold['date'].dt.date.astype(str)
            gold = gold.sort_values('date')

            # Forward-fill weekly/monthly data
            for col in gold.columns:
                if col != 'date':
                    gold[col] = gold[col].ffill()

            gold_cols = [c for c in ['date', 'gld_holdings', 'cot_net_long', 'us_real_yield', 'm2_money_supply']
                         if c in gold.columns]
            gold_merge = gold[gold_cols].copy()

            df = df.merge(gold_merge, on='date', how='left')

            # Forward-fill and compute changes
            for col in gold_cols:
                if col != 'date':
                    df[col] = df[col].ffill()

            if 'gld_holdings' in df.columns:
                df['gld_holdings_change'] = df['gld_holdings'].pct_change(6, fill_method=None)  # day-over-day (6 bars/day)
            if 'm2_money_supply' in df.columns:
                df['m2_yoy_change'] = df['m2_money_supply'].pct_change(6 * 252, fill_method=None)  # ~1 year
        except FileNotFoundError:
            pass

    return df
