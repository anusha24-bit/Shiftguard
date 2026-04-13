"""P&L with 0.5% stop loss from existing trade CSVs. No model runs."""
import pandas as pd, numpy as np

pairs = ['EURUSD', 'GBPJPY', 'XAUUSD']
base = r'C:\Users\Sohan M\Desktop\Shiftguard\results\winrate'
capital = 10000
sl = 0.005  # 0.5% stop loss
spread = {'EURUSD': 0.00010, 'GBPJPY': 0.00020, 'XAUUSD': 0.00030}
commission = 0.00003
slippage = 0.00002
swap = 0.000005
tax_rate = 0.30

horizons = {
    '1 Month':  ('2026-03-01', '2026-04-01'),
    '3 Months': ('2026-01-01', '2026-04-01'),
    '6 Months': ('2025-10-01', '2026-04-01'),
    '1 Year':   ('2025-04-01', '2026-04-01'),
    '2 Years':  ('2024-04-01', '2026-04-01'),
    '5 Years':  ('2021-04-01', '2026-04-01'),
}

data = {}
for p in pairs:
    df = pd.read_csv(f'{base}/{p}_winrate_trades.csv', parse_dates=['datetime_utc'])
    data[p] = df

strats = [('Technical', 'tech_signal'), ('ML Direction', 'ml_signal'), ('ShiftGuard', 'sg_signal')]

for lev_label, lev_val in [('NO LEVERAGE (1:1)', 1), ('WITH LEVERAGE (1:20)', 20)]:
    print()
    print('=' * 90)
    print('  ' + lev_label + ' | $10,000/pair | 0.5% SL | costs + 30% tax')
    print('=' * 90)
    for hname, (start, end) in horizons.items():
        print()
        print('  ' + hname + ' (' + start + ' to ' + end + ')')
        hdr = '  {:<16}{:>8}{:>8}{:>14}{:>14}{:>10}'.format(
            'Strategy', 'Trades', 'Win%', 'After Tax', 'Balance', 'Result')
        print(hdr)
        print('  ' + '-' * 68)
        for sname, col in strats:
            tot_trades = 0
            tot_wins = 0
            tot_at = 0.0
            tot_stopped = 0
            for p in pairs:
                lev = lev_val
                d = data[p]
                mask = (d['datetime_utc'] >= start) & (d['datetime_utc'] < end)
                chunk = d[mask]
                if len(chunk) == 0:
                    continue
                s = chunk[col].values
                r = chunk['actual_return'].values
                tm = s != 0
                nt = int(tm.sum())
                if nt == 0:
                    continue
                raw = s[tm] * r[tm]
                lp = raw * lev
                slv = sl * lev
                capped = np.where(lp < -slv, -slv, lp)
                cpt = (spread[p] + commission + slippage + swap) * lev
                net = capped - cpt
                gross = capital * capped.sum()
                costs = capital * cpt * nt
                pt = gross - costs
                tx = max(0, pt * tax_rate)
                at = pt - tx
                tot_trades += nt
                tot_wins += int((net > 0).sum())
                tot_at += at
                tot_stopped += int((lp < -slv).sum())

            wr = tot_wins / tot_trades * 100 if tot_trades > 0 else 0
            bal = 30000 + tot_at
            if bal <= 0:
                status = 'BLOWN'
            elif tot_at > 0:
                status = 'PROFIT'
            else:
                status = 'LOSS'
            row = '  {:<16}{:>8}{:>7.1f}%  ${:>+12,.0f}  ${:>12,.0f}  {:>8}'.format(
                sname, tot_trades, wr, tot_at, bal, status)
            print(row)
