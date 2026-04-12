"""
Final Results Table
All models × All time horizons × $10K investment
"""
import pandas as pd
import numpy as np
import json

WINRATE_DIR = 'C:/Users/Sohan M/Desktop/Shiftguard/results/winrate'
pairs = ['EURUSD', 'GBPJPY', 'XAUUSD']
capital = 10000

horizons = {
    '1 Month':  ('2026-03-01', '2026-04-01'),
    '3 Months': ('2026-01-01', '2026-04-01'),
    '6 Months': ('2025-10-01', '2026-04-01'),
    '1 Year':   ('2025-04-01', '2026-04-01'),
    '2 Years':  ('2024-01-01', '2026-04-01'),
    '5 Years':  ('2021-01-01', '2026-04-01'),
}

# Load all trade data
data = {}
for pair in pairs:
    df = pd.read_csv(f'{WINRATE_DIR}/{pair}_winrate_trades.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    data[pair] = df


def compute_stats(signals, returns, capital=10000):
    tm = signals != 0
    nt = tm.sum()
    if nt == 0:
        return {'trades': 0, 'win_rate': 0, 'profit': 0, 'balance': capital, 'pf': 0, 'return_pct': 0}
    pnl = signals[tm] * returns[tm]
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]
    wr = len(w) / nt * 100
    total_ret = pnl.sum()
    profit = capital * total_ret
    balance = capital + profit
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else 99
    return {
        'trades': int(nt),
        'win_rate': round(wr, 1),
        'profit': round(profit, 0),
        'balance': round(balance, 0),
        'pf': round(min(pf, 99), 2),
        'return_pct': round(total_ret * 100, 1),
    }


# ============================================================
# PER-PAIR TABLES
# ============================================================
for pair in pairs:
    print(f"\n{'='*90}")
    print(f"  {pair} — $10,000 Investment Results")
    print(f"{'='*90}")
    
    header = f"{'Horizon':<12} | {'Strategy':<28} | {'Trades':>7} | {'Win%':>6} | {'Profit':>10} | {'Balance':>10} | {'Return':>8} | {'PF':>5}"
    print(header)
    print("-" * len(header))
    
    df = data[pair]
    
    for hname, (start, end) in horizons.items():
        mask = (df['datetime_utc'] >= start) & (df['datetime_utc'] < end)
        chunk = df[mask]
        if len(chunk) == 0:
            continue
        
        strategies = {
            'Technical (RSI/MACD)': chunk['tech_signal'].values,
            'ML Direction (XGBoost)': chunk['ml_signal'].values,
            'ShiftGuard (Regime)': chunk['sg_signal'].values,
        }
        
        first = True
        for sname, signals in strategies.items():
            returns = chunk['actual_return'].values
            r = compute_stats(signals, returns, capital)
            
            h_label = hname if first else ""
            first = False
            
            profit_str = f"${r['profit']:>+9,.0f}" if r['profit'] != 0 else f"{'$0':>10}"
            balance_str = f"${r['balance']:>9,.0f}"
            
            print(f"{h_label:<12} | {sname:<28} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | {profit_str} | {balance_str} | {r['return_pct']:>+7.1f}% | {r['pf']:>5.2f}")
        
        print("-" * len(header))


# ============================================================
# COMBINED PORTFOLIO ($10K per pair = $30K total)
# ============================================================
print(f"\n\n{'='*90}")
print(f"  COMBINED PORTFOLIO — $10K per pair = $30K Total")
print(f"{'='*90}")

header = f"{'Horizon':<12} | {'Strategy':<28} | {'Trades':>7} | {'Win%':>6} | {'Profit':>10} | {'Balance':>10} | {'Return':>8}"
print(header)
print("-" * len(header))

for hname, (start, end) in horizons.items():
    strategy_totals = {}
    
    for sname_key, col in [('Technical (RSI/MACD)', 'tech_signal'),
                            ('ML Direction (XGBoost)', 'ml_signal'),
                            ('ShiftGuard (Regime)', 'sg_signal')]:
        total_profit = 0
        total_trades = 0
        total_wins = 0
        total_traded = 0
        
        for pair in pairs:
            df = data[pair]
            mask = (df['datetime_utc'] >= start) & (df['datetime_utc'] < end)
            chunk = df[mask]
            if len(chunk) == 0:
                continue
            
            signals = chunk[col].values
            returns = chunk['actual_return'].values
            tm = signals != 0
            nt = tm.sum()
            
            if nt > 0:
                pnl = signals[tm] * returns[tm]
                total_profit += capital * pnl.sum()
                total_trades += nt
                total_wins += (pnl > 0).sum()
                total_traded += nt
        
        wr = total_wins / total_traded * 100 if total_traded > 0 else 0
        balance = 30000 + total_profit
        ret_pct = total_profit / 30000 * 100
        
        strategy_totals[sname_key] = {
            'trades': total_trades,
            'win_rate': round(wr, 1),
            'profit': round(total_profit, 0),
            'balance': round(balance, 0),
            'return_pct': round(ret_pct, 1),
        }
    
    first = True
    for sname, r in strategy_totals.items():
        h_label = hname if first else ""
        first = False
        
        profit_str = f"${r['profit']:>+9,.0f}"
        balance_str = f"${r['balance']:>9,.0f}"
        
        print(f"{h_label:<12} | {sname:<28} | {r['trades']:>7,} | {r['win_rate']:>5.1f}% | {profit_str} | {balance_str} | {r['return_pct']:>+7.1f}%")
    
    print("-" * len(header))


# ============================================================
# SUMMARY: ShiftGuard vs Technical edge
# ============================================================
print(f"\n\n{'='*90}")
print(f"  SHIFTGUARD EDGE OVER TECHNICAL ($30K Portfolio)")
print(f"{'='*90}")
print(f"{'Horizon':<12} | {'Tech Profit':>12} | {'SG Profit':>12} | {'SG Extra':>12} | {'SG Fewer Trades':>16}")
print("-" * 70)

for hname, (start, end) in horizons.items():
    tech_p = 0
    sg_p = 0
    tech_t = 0
    sg_t = 0
    
    for pair in pairs:
        df = data[pair]
        mask = (df['datetime_utc'] >= start) & (df['datetime_utc'] < end)
        chunk = df[mask]
        if len(chunk) == 0:
            continue
        
        for col, accum_p, accum_t in [('tech_signal', 'tech', None), ('sg_signal', 'sg', None)]:
            signals = chunk[col].values
            returns = chunk['actual_return'].values
            tm = signals != 0
            pnl = signals[tm] * returns[tm]
            if col == 'tech_signal':
                tech_p += capital * pnl.sum()
                tech_t += tm.sum()
            else:
                sg_p += capital * pnl.sum()
                sg_t += tm.sum()
    
    saved = tech_t - sg_t
    extra = sg_p - tech_p
    print(f"{hname:<12} | ${tech_p:>+11,.0f} | ${sg_p:>+11,.0f} | ${extra:>+11,.0f} | {saved:>15,} fewer")
