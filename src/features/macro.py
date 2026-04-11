"""
Group 3: Macro Features
Merges interest rates, calendar event surprises, and event flags onto 4H bars.
Macro data is daily/monthly — forward-filled so each 4H bar in a day gets the same value.
"""
import pandas as pd
import numpy as np


# Which calendar to use for each pair
PAIR_CURRENCIES = {
    'EURUSD': ['USD', 'EUR'],
    'GBPJPY': ['GBP', 'JPY'],
    'XAUUSD': ['USD'],
}

# Which rate columns to use for each pair
PAIR_RATES = {
    'EURUSD': {
        'rate_diff': ('fed_funds_rate', 'ecb_rate'),
        'yield_spread': ('us_10y_yield', 'german_10y_bund'),
    },
    'GBPJPY': {
        'rate_diff': ('boe_rate', 'boj_rate'),
        'yield_spread': ('uk_10y_gilt', 'japan_10y_jgb'),
    },
    'XAUUSD': {
        'rate_diff': ('fed_funds_rate', 'ecb_rate'),
        'yield_spread': ('us_10y_yield', 'german_10y_bund'),
    },
}


def load_rates(data_dir):
    """Load and forward-fill interest rates."""
    rates = pd.read_csv(f'{data_dir}/macro/interest_rates_daily.csv', parse_dates=['date'])
    rates = rates.sort_values('date').set_index('date')
    rates = rates.ffill()
    return rates


def load_calendar(data_dir, currencies):
    """Load and combine economic calendars for the relevant currencies."""
    dfs = []
    for curr in currencies:
        path = f'{data_dir}/calendar/{curr}_economic_calendar.csv'
        cal = pd.read_csv(path)
        cal['date'] = pd.to_datetime(cal['date'])
        dfs.append(cal)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def compute_event_surprises(calendar_df, dates):
    """
    Compute event surprise features per date.
    surprise = actual - forecast (where both are numeric).
    Returns a DataFrame indexed by date with surprise columns.
    """
    cal = calendar_df.copy()

    # Extract numeric values from actual/forecast (they may have %, K, M suffixes)
    for col in ['actual_value', 'forecast_value']:
        cal[col] = (
            cal[col].astype(str)
            .str.replace('%', '', regex=False)
            .str.replace('K', '', regex=False)
            .str.replace('M', '', regex=False)
            .str.replace('B', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        cal[col] = pd.to_numeric(cal[col], errors='coerce')

    cal['surprise'] = cal['actual_value'] - cal['forecast_value']

    # Aggregate: mean surprise per date (multiple events may occur on same day)
    high_impact = cal[cal['impact_level'] == 'High']
    daily_surprise = high_impact.groupby('date')['surprise'].mean().rename('event_surprise')
    event_count = high_impact.groupby('date').size().rename('event_count')

    result = pd.DataFrame(index=dates)
    result = result.join(daily_surprise)
    result = result.join(event_count)
    result['event_surprise'] = result['event_surprise'].fillna(0)
    result['event_count'] = result['event_count'].fillna(0).astype(int)

    return result


def compute_event_proximity(dates, calendar_df):
    """Days until next high-impact event for each date."""
    high_dates = sorted(
        calendar_df[calendar_df['impact_level'] == 'High']['date'].unique()
    )
    high_dates = pd.to_datetime(high_dates)

    proximity = []
    for d in dates:
        future = high_dates[high_dates > d]
        if len(future) > 0:
            proximity.append((future[0] - d).days)
        else:
            proximity.append(np.nan)

    return pd.Series(proximity, index=dates, name='days_to_next_event')


def compute_macro_features(df, pair_name, data_dir):
    """
    Merge macro features onto 4H bars by date.
    df must have a 'date' column (string YYYY-MM-DD).
    """
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])

    # --- Interest rates ---
    rates = load_rates(data_dir)
    pair_cfg = PAIR_RATES.get(pair_name, PAIR_RATES['EURUSD'])

    # Merge rates by date
    rate_cols_needed = set()
    for col_pair in pair_cfg.values():
        rate_cols_needed.update(col_pair)
    rate_cols_needed = [c for c in rate_cols_needed if c in rates.columns]

    rates_subset = rates[rate_cols_needed].reset_index()
    rates_subset['date'] = rates_subset['date'].dt.date.astype(str)
    df = df.merge(rates_subset, on='date', how='left')

    # Forward-fill rate values (weekends/holidays)
    for col in rate_cols_needed:
        df[col] = df[col].ffill()

    # Compute differentials
    r1, r2 = pair_cfg['rate_diff']
    if r1 in df.columns and r2 in df.columns:
        df['rate_diff'] = df[r1] - df[r2]
        df['rate_diff_delta'] = df['rate_diff'] - df['rate_diff'].shift(30 * 6)  # 30 days × 6 bars/day

    y1, y2 = pair_cfg['yield_spread']
    if y1 in df.columns and y2 in df.columns:
        df['yield_spread'] = df[y1] - df[y2]

    # Yield curve (US only, useful for all pairs)
    if 'us_10y_yield' in df.columns and 'us_2y_yield' in df.columns:
        df['yield_curve'] = df['us_10y_yield'] - df['us_2y_yield']

    # --- Calendar events ---
    currencies = PAIR_CURRENCIES.get(pair_name, ['USD'])
    calendar = load_calendar(data_dir, currencies)
    unique_dates = df['date_dt'].unique()

    surprises = compute_event_surprises(calendar, unique_dates)
    surprises.index = surprises.index.date.astype('str') if hasattr(surprises.index, 'date') else surprises.index
    # Handle index type
    surprise_df = surprises.reset_index()
    surprise_df.columns = ['date', 'event_surprise', 'event_count']
    surprise_df['date'] = surprise_df['date'].astype(str)
    df = df.merge(surprise_df, on='date', how='left')
    df['event_surprise'] = df['event_surprise'].fillna(0)
    df['event_count'] = df['event_count'].fillna(0)

    # Event proximity
    proximity = compute_event_proximity(unique_dates, calendar)
    prox_df = pd.DataFrame({'date': proximity.index.astype(str), 'days_to_next_event': proximity.values})
    df = df.merge(prox_df, on='date', how='left')

    # Binary event flags
    rate_events = calendar[calendar['event_name'].str.contains('Interest Rate|Rate Decision', case=False, na=False)]
    df['is_rate_decision_day'] = df['date'].isin(rate_events['date'].dt.date.astype(str).values).astype(int)

    nfp_events = calendar[calendar['event_name'].str.contains('Nonfarm|Non-Farm|NFP', case=False, na=False)]
    df['is_nfp_day'] = df['date'].isin(nfp_events['date'].dt.date.astype(str).values).astype(int)

    cpi_events = calendar[calendar['event_name'].str.contains('CPI|Consumer Price', case=False, na=False)]
    df['is_cpi_day'] = df['date'].isin(cpi_events['date'].dt.date.astype(str).values).astype(int)

    # Drop helper columns
    df = df.drop(columns=['date_dt'], errors='ignore')

    return df
