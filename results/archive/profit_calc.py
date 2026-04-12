import json

capital = 10000
pairs = ['EURUSD', 'GBPJPY', 'XAUUSD']
total_tech = 0
total_sg = 0

for pair in pairs:
    with open(f'C:/Users/Sohan M/Desktop/Shiftguard/results/winrate/{pair}_winrate_summary.json') as f:
        r = json.load(f)

    tech = r['Technical (RSI/MACD)']
    sg = r['ShiftGuard (Regime-Filtered)']

    tech_profit = capital * tech['total_return']
    sg_profit = capital * sg['total_return']

    total_tech += tech_profit
    total_sg += sg_profit

    print(f"{pair}:")
    print(f"  Technical:  ${tech_profit:>12,.2f}  ({tech['total_return']*100:+.1f}%)  {tech['n_trades']} trades")
    print(f"  ShiftGuard: ${sg_profit:>12,.2f}  ({sg['total_return']*100:+.1f}%)  {sg['n_trades']} trades")
    print(f"  SG made ${sg_profit - tech_profit:,.2f} more")
    print()

print("=" * 50)
print(f"TOTAL ($10K per pair, $30K deployed):")
print(f"  Technical:  ${total_tech:>12,.2f}")
print(f"  ShiftGuard: ${total_sg:>12,.2f}")
print(f"  ShiftGuard made ${total_sg - total_tech:,.2f} MORE")
print(f"  ShiftGuard return: {total_sg/30000*100:.1f}%")
print(f"  Technical return:  {total_tech/30000*100:.1f}%")
