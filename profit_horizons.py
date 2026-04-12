import pandas as pd, numpy as np

capital = 10000
pairs = ['EURUSD', 'GBPJPY', 'XAUUSD']

# Time horizons (end date = latest available, ~April 2026)
horizons = {
    '1 Day':    ('2026-04-09', '2026-04-10'),
    '1 Week':   ('2026-04-03', '2026-04-10'),
    '1 Month':  ('2026-03-10', '2026-04-10'),
    '6 Months': ('2025-10-10', '2026-04-10'),
    '1 Year':   ('2025-04-10', '2026-04-10'),
}

# Load all data
data = {}
for pair in pairs:
    df = pd.read_csv(f'C:/Users/Sohan M/Desktop/Shiftguard/results/winrate/{pair}_winrate_trades.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    data[pair] = df

for hname, (start, end) in horizons.items():
    print(f"\n{'='*70}")
    print(f"  {hname}  ({start} to {end})")
    print(f"{'='*70}")

    total_tech_profit = 0
    total_sg_profit = 0

    for pair in pairs:
        d = data[pair]
        mask = (d['datetime_utc'] >= start) & (d['datetime_utc'] < end)
        chunk = d[mask]

        if len(chunk) == 0:
            print(f"  {pair}: No data in this period")
            continue

        results = {}
        for sname, col in [('Tech', 'tech_signal'), ('SG', 'sg_signal')]:
            s = chunk[col].values
            r = chunk['actual_return'].values
            tm = s != 0
            nt = tm.sum()

            if nt == 0:
                results[sname] = {'wr': 0, 'nt': 0, 'profit': 0, 'bal': capital}
                continue

            pnl = s[tm] * r[tm]
            w = pnl[pnl > 0]
            l = pnl[pnl < 0]
            wr = len(w) / nt * 100
            pr = capital * pnl.sum()
            bal = capital + pr

            results[sname] = {'wr': wr, 'nt': nt, 'profit': pr, 'bal': bal}

        t = results['Tech']
        sg = results['SG']
        total_tech_profit += t['profit']
        total_sg_profit += sg['profit']

        print(f"  {pair}:  Tech: {t['wr']:5.1f}% win, {t['nt']:4d} trades, ${t['profit']:>8,.0f} → ${t['bal']:>8,.0f}  |  "
              f"SG: {sg['wr']:5.1f}% win, {sg['nt']:4d} trades, ${sg['profit']:>8,.0f} → ${sg['bal']:>8,.0f}")

    print(f"\n  TOTAL ($30K):  Tech profit: ${total_tech_profit:>8,.0f}  |  SG profit: ${total_sg_profit:>8,.0f}  |  SG edge: ${total_sg_profit - total_tech_profit:>+8,.0f}")
