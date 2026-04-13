"""
All Models Comparison: B1, B2, B3, Technical, ML, ShiftGuard
With stop losses and time horizons
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PAIRS = ['EURUSD', 'GBPJPY', 'XAUUSD']
CAPITAL = 10000
SL = {'EURUSD': 0.005, 'GBPJPY': 0.01, 'XAUUSD': 0.01}  # 0.5% and 1%

# Realistic trading costs per trade (as fraction of price)
# Spread + commission + slippage combined
TRADE_COST = {
    'EURUSD': 0.00015,  # ~1.5 pips (spread 1 pip + commission 0.3 pip + slippage 0.2 pip)
    'GBPJPY': 0.00025,  # ~2.5 pips (wider spread for cross pair)
    'XAUUSD': 0.00035,  # ~35 cents on gold (wider spread + higher slippage)
}

LEVERAGE = 20  # 1:20 leverage — profits and losses multiplied by 20
TAX_RATE = 0.30  # 30% tax on net profit (California short-term capital gains)

HORIZONS = {
    '1 Month':  ('2026-03-01', '2026-04-01'),
    '3 Months': ('2026-01-01', '2026-04-01'),
    '6 Months': ('2025-10-01', '2026-04-01'),
    '1 Year':   ('2025-04-01', '2026-04-01'),
    '2 Years':  ('2024-04-01', '2026-04-01'),
    '5 Years':  ('2021-04-01', '2026-04-01'),
}

BASE_DIR = 'C:/Users/Sohan M/Desktop/Shiftguard/results'


def apply_sl(signal, actual_return, sl):
    """Apply stop loss: cap loss at -sl"""
    pnl = signal * actual_return
    if pnl < -sl:
        return -sl
    return pnl


def compute_trading_metrics(signals, returns, sl, cost, capital=10000):
    """Compute trading metrics with stop loss + trading costs"""
    tm = signals != 0
    nt = tm.sum()
    if nt == 0:
        return {'trades': 0, 'win_rate': 0, 'profit': 0, 'balance': capital, 'pf': 0}

    # Apply SL then subtract cost per trade, then apply leverage
    pnl = np.array([(apply_sl(s, r, sl) - cost) * LEVERAGE for s, r in zip(signals[tm], returns[tm])])
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / nt * 100
    total_profit = capital * pnl.sum()
    # Can't lose more than capital (margin call)
    total_profit = max(total_profit, -capital)
    # Apply tax on net profit (only if profitable)
    tax = total_profit * TAX_RATE if total_profit > 0 else 0
    after_tax_profit = total_profit - tax
    balance = capital + after_tax_profit
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99

    return {
        'trades': int(nt),
        'win_rate': round(wr, 1),
        'profit': round(after_tax_profit, 0),
        'balance': round(balance, 0),
        'tax': round(tax, 0),
        'pf': round(min(pf, 99), 2),
    }


def compute_ml_metrics(actual_dir, pred_dir):
    """Compute classification metrics"""
    acc = accuracy_score(actual_dir, pred_dir)
    f1 = f1_score(actual_dir, pred_dir, zero_division=0)
    prec = precision_score(actual_dir, pred_dir, zero_division=0)
    rec = recall_score(actual_dir, pred_dir, zero_division=0)
    return {
        'accuracy': round(acc * 100, 1),
        'f1': round(f1, 3),
        'precision': round(prec, 3),
        'recall': round(rec, 3),
    }


# ============================================================
# LOAD DATA
# ============================================================
winrate_data = {}
for pair in PAIRS:
    df = pd.read_csv(f'{BASE_DIR}/winrate/{pair}_winrate_trades.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    winrate_data[pair] = df

baseline_data = {}
for bname, bdir in [('B1 LSTM', 'baseline1/baseline1_lstm'),
                     ('B2 BiLSTM', 'baseline2/baseline2_bilstm'),
                     ('B3 Stacked', 'baseline3/baseline3_stacked')]:
    baseline_data[bname] = {}
    for pair in PAIRS:
        try:
            df = pd.read_csv(f'{BASE_DIR}/{bdir}_{pair}.csv')
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            df['actual_dir'] = (df['actual'] > 0).astype(int)
            df['pred_dir'] = (df['predicted'] > 0).astype(int)
            # Create trading signal from prediction
            df['signal'] = np.where(df['predicted'] > 0, 1, -1)
            baseline_data[bname][pair] = df
        except Exception as e:
            baseline_data[bname][pair] = None


# ============================================================
# COMPUTE AND PRINT
# ============================================================
# Per-pair tables: Baselines vs ML Direction vs ShiftGuard
for pair in PAIRS:
    print(f"\n{'='*120}")
    print(f"  {pair} — $10K, 1:20 leverage, costs, 30% tax, SL={'0.5%' if pair=='EURUSD' else '1%'}")
    print(f"{'='*120}")
    header = f"{'Horizon':<10} | {'Model':<14} | {'Acc':>5} | {'F1':>5} | {'WinR':>5} | {'Trades':>6} | {'PF':>5} | {'Profit':>12} | {'Balance':>10}"
    print(header)
    print("─" * len(header))

    sl = SL[pair]

    for hname, (start, end) in HORIZONS.items():
        models = []

        # Baselines
        for bname in ['B1 LSTM', 'B2 BiLSTM', 'B3 Stacked']:
            bdf = baseline_data[bname][pair]
            if bdf is None:
                models.append((bname, None, None))
                continue
            mask = (bdf['datetime_utc'] >= start) & (bdf['datetime_utc'] < end)
            chunk = bdf[mask]
            if len(chunk) < 5:
                models.append((bname, None, None))
                continue
            ml = compute_ml_metrics(chunk['actual_dir'].values, chunk['pred_dir'].values)
            tr = compute_trading_metrics(chunk['signal'].values, chunk['actual'].values, sl, TRADE_COST[pair], CAPITAL)
            models.append((bname, ml, tr))

        # ML Direction + ShiftGuard
        wdf = winrate_data[pair]
        mask = (wdf['datetime_utc'] >= start) & (wdf['datetime_utc'] < end)
        chunk = wdf[mask]
        if len(chunk) >= 5:
            for sname, col in [('ML Direction', 'ml_signal'), ('ShiftGuard', 'sg_signal')]:
                signals = chunk[col].values
                returns = chunk['actual_return'].values
                tr = compute_trading_metrics(signals, returns, sl, TRADE_COST[pair], CAPITAL)
                traded = signals != 0
                if traded.sum() > 0:
                    actual_dir = (returns[traded] > 0).astype(int)
                    pred_dir = (signals[traded] > 0).astype(int)
                    ml = compute_ml_metrics(actual_dir, pred_dir)
                else:
                    ml = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
                models.append((sname, ml, tr))

        first = True
        for mname, ml, tr in models:
            h_label = hname if first else ""
            first = False
            if ml is None or tr is None:
                print(f"{h_label:<10} | {mname:<14} | {'—':>5} | {'—':>5} | {'—':>5} | {'—':>6} | {'—':>5} | {'—':>12} | {'—':>10}")
                continue
            profit_str = f"${tr['profit']:>+11,.0f}" if tr['profit'] != 0 else f"{'$0':>12}"
            balance_str = f"${tr['balance']:>9,.0f}"
            print(f"{h_label:<10} | {mname:<14} | {ml['accuracy']:>5.1f} | {ml['f1']:>5.3f} | {tr['win_rate']:>5.1f} | {tr['trades']:>6} | {tr['pf']:>5.2f} | {profit_str} | {balance_str}")
        print("─" * len(header))
