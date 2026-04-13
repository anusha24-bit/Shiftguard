import pandas as pd
import numpy as np

PAIRS = ['EURUSD', 'GBPJPY', 'XAUUSD']
CAPITAL = 10000
SL = {'EURUSD': 0.005, 'GBPJPY': 0.01, 'XAUUSD': 0.01}
TRADE_COST = {'EURUSD': 0.00015, 'GBPJPY': 0.00025, 'XAUUSD': 0.00035}
LEVERAGE = 20
TAX_RATE = 0.30
BASE_DIR = 'C:/Users/Sohan M/Desktop/Shiftguard/results/retraining'

HORIZONS = {
    '1 Month':  ('2026-02-01', '2026-03-18'),
    '3 Months': ('2025-12-18', '2026-03-18'),
    '6 Months': ('2025-09-18', '2026-03-18'),
    '1 Year':   ('2025-03-18', '2026-03-18'),
    '2 Years':  ('2024-03-18', '2026-03-18'),
    '5 Years':  ('2021-03-18', '2026-03-18'),
}

STRATEGIES = {
    'No Retrain': 'pred_none',
    'Blind Monthly': 'pred_blind',
    'ShiftGuard': 'pred_shap',
    'Oracle': 'pred_oracle',
}

def compute(signals, returns, sl, cost, capital=10000):
    tm = signals != 0
    nt = tm.sum()
    if nt == 0:
        return {'trades': 0, 'win_rate': 0, 'profit': 0, 'balance': capital, 'pf': 0}
    pnl = []
    for s, r in zip(signals[tm], returns[tm]):
        p = s * r
        if p < -sl:
            p = -sl
        p = (p - cost) * LEVERAGE
        pnl.append(p)
    pnl = np.array(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / nt * 100
    total_profit = capital * pnl.sum()
    total_profit = max(total_profit, -capital)
    tax = total_profit * TAX_RATE if total_profit > 0 else 0
    after_tax = total_profit - tax
    balance = capital + after_tax
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
    acc = (pnl > 0).sum() / nt * 100
    return {
        'trades': int(nt), 'win_rate': round(wr, 1), 'profit': round(after_tax, 0),
        'balance': round(balance, 0), 'pf': round(min(pf, 99), 2), 'acc': round(acc, 1),
    }

for pair in PAIRS:
    df = pd.read_csv(f'{BASE_DIR}/{pair}_walkforward_bars.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    print(f"\n{'='*110}")
    print(f"  {pair} — PIPELINE RESULTS — $10K, 1:20 leverage, costs, 30% tax, SL={'0.5%' if pair=='EURUSD' else '1%'}")
    print(f"{'='*110}")
    header = f"{'Horizon':<10} | {'Strategy':<16} | {'WinR':>5} | {'Trades':>6} | {'PF':>5} | {'Profit':>12} | {'Balance':>10}"
    print(header)
    print("-" * len(header))

    for hname, (start, end) in HORIZONS.items():
        mask = (df['datetime_utc'] >= start) & (df['datetime_utc'] <= end)
        chunk = df[mask]
        if len(chunk) < 5:
            continue

        first = True
        for sname, col in STRATEGIES.items():
            preds = chunk[col].values
            actuals = chunk['actual'].values
            signals = np.where(preds > 0, 1, -1)
            sl = SL[pair]
            r = compute(signals, actuals, sl, TRADE_COST[pair], CAPITAL)
            h_label = hname if first else ""
            first = False
            profit_str = f"${r['profit']:>+11,.0f}"
            balance_str = f"${r['balance']:>9,.0f}"
            print(f"{h_label:<10} | {sname:<16} | {r['win_rate']:>5.1f} | {r['trades']:>6} | {r['pf']:>5.2f} | {profit_str} | {balance_str}")
        print("-" * len(header))
