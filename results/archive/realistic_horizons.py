import pandas as pd, numpy as np

capital = 10000
sl = 0.01
leverage = {'EURUSD': 20, 'GBPJPY': 10, 'XAUUSD': 10}
spread_pips = {'EURUSD': 0.00010, 'GBPJPY': 0.00020, 'XAUUSD': 0.00030}
commission = 0.00003
slippage = 0.00002
swap = 0.000005
tax_rate = 0.30
pairs = ['EURUSD', 'GBPJPY', 'XAUUSD']

horizons = {
    '1 Month':  ('2026-03-01', '2026-04-01'),
    '3 Months': ('2026-01-01', '2026-04-01'),
    '6 Months': ('2025-10-01', '2026-04-01'),
    '1 Year':   ('2025-04-01', '2026-04-01'),
}

data = {}
for pair in pairs:
    df = pd.read_csv('C:/Users/Sohan M/Desktop/Shiftguard/results/winrate/' + pair + '_winrate_trades.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    data[pair] = df

for hname, (start, end) in horizons.items():
    print('=' * 75)
    print('  ' + hname + ' (' + start + ' to ' + end + ') | $30K portfolio')
    print('=' * 75)

    for sname, col in [('Technical', 'tech_signal'), ('ML Direction', 'ml_signal'), ('ShiftGuard', 'sg_signal')]:
        total_gross = 0
        total_costs = 0
        total_trades = 0
        total_wins = 0
        total_stopped = 0

        for pair in pairs:
            lev = leverage[pair]
            sp = spread_pips[pair]
            d = data[pair]
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
            lev_pnl = raw * lev
            sl_lev = sl * lev
            capped = np.where(lev_pnl < -sl_lev, -sl_lev, lev_pnl)

            cost_per = (sp + commission + slippage + swap) * lev
            net = capped - cost_per

            total_gross += capital * capped.sum()
            total_costs += capital * cost_per * nt
            total_trades += nt
            total_wins += int((net > 0).sum())
            total_stopped += int((lev_pnl < -sl_lev).sum())

        pre_tax = total_gross - total_costs
        tax = max(0, pre_tax * tax_rate)
        after_tax = pre_tax - tax
        balance = 30000 + after_tax
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        status = 'PROFIT' if after_tax > 0 else 'LOSS'
        print('  ' + sname + ':')
        print('    Trades: ' + str(total_trades) + ' | Win: ' + str(round(wr,1)) + '% | Stopped: ' + str(total_stopped))
        print('    Gross: $' + str(int(total_gross)) + ' | Costs: $' + str(int(total_costs)) + ' | Tax: $' + str(int(tax)))
        print('    AFTER TAX: $' + str(int(after_tax)) + ' | Balance: $' + str(int(balance)) + ' [' + status + ']')
        print()
