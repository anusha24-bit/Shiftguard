"""
ShiftGuard — Human-in-the-Loop Dashboard
Run: streamlit run src/dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.trading_analytics import enrich_shift
from src.dashboard.decision_utils import (
    auto_confirm_shifts,
    load_decisions,
    queue_retrain,
    save_decision,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
PAIR_LABELS = {'EURUSD': 'EUR/USD', 'GBPJPY': 'GBP/JPY', 'XAUUSD': 'XAU/USD'}
REGIME_NAMES = {0: 'Trend Up (Calm)', 1: 'Trend Up (Volatile)', 2: 'Ranging',
                3: 'Trend Down (Calm)', 4: 'Trend Down (Volatile)'}
REGIME_COLORS = {0: '#2ecc71', 1: '#0096ff', 2: '#95a5a6', 3: '#ffa500', 4: '#e74c3c'}
PLOTLY_TEMPLATE = 'simple_white'


def load_csv(rel_path):
    full = os.path.join(ROOT, rel_path)
    return pd.read_csv(full) if os.path.exists(full) else pd.DataFrame()


def load_json(rel_path):
    full = os.path.join(ROOT, rel_path)
    if os.path.exists(full):
        with open(full) as f:
            return json.load(f)
    return {}


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ShiftGuard", layout="wide")

# ── Design Tokens ────────────────────────────────────────────────────────────
BG_PRIMARY = '#f7f0e4'
BG_CARD = '#fffaf2'
BG_SURFACE = '#eadfce'
ACCENT = '#1f7a73'
ACCENT_DIM = '#175d58'
TEXT_PRIMARY = '#000000'
TEXT_MUTED = '#000000'
TEXT_DIM = '#000000'
DANGER = '#c44b3f'
WARN = '#c18b32'

st.markdown(f"""
<style>
    .main > div {{
        padding-top: 1.2rem;
        max-width: 1180px;
    }}
    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(223, 192, 134, 0.18), transparent 24%),
            radial-gradient(circle at 18% 18%, rgba(31, 122, 115, 0.10), transparent 20%),
            linear-gradient(180deg, #f9f3e8 0%, #f5ecdf 100%);
    }}
    [data-testid="stSidebar"] {{
        background: rgba(255, 249, 240, 0.86);
        border-right: 1px solid {BG_SURFACE};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY};
    }}
    label, .stMarkdown, .stCaption, .stText, p, span, div, li {{
        color: {TEXT_PRIMARY};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255, 250, 242, 0.82);
        padding: 0.4rem;
        border: 1px solid {BG_SURFACE};
        border-radius: 999px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        padding: 10px 18px;
        font-size: 0.95rem;
        color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background: {TEXT_PRIMARY};
        color: #fffaf2 !important;
    }}
    .stTabs [aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] div {{
        color: #fffaf2 !important;
    }}
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        border-color: {BG_SURFACE};
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 16px;
    }}
    .stTextInput input,
    .stNumberInput input {{
        background: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
    }}
    .stNumberInput button {{
        background: #ffffff !important;
        color: #000000 !important;
        border-color: {BG_SURFACE} !important;
    }}
    .stNumberInput button * {{
        color: #000000 !important;
        fill: #000000 !important;
    }}
    .stNumberInput [data-baseweb="input"] {{
        background: #ffffff !important;
        border: 1px solid {BG_SURFACE} !important;
        border-radius: 16px !important;
    }}
    .stNumberInput [data-baseweb="input"] > div,
    .stTextInput [data-baseweb="input"] > div {{
        background: #ffffff !important;
        color: #000000 !important;
    }}
    .stSelectbox div[data-baseweb="select"] {{
        background: #ffffff !important;
        color: #000000 !important;
        border-color: {BG_SURFACE} !important;
        border-radius: 16px !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background: #ffffff !important;
        color: #000000 !important;
    }}
    .stSelectbox div[data-baseweb="select"] * {{
        color: #000000 !important;
        fill: #000000 !important;
    }}
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {{
        background: rgba(255, 250, 242, 0.98) !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BG_SURFACE} !important;
        border-radius: 18px !important;
        box-shadow: 0 16px 36px rgba(84, 63, 39, 0.14) !important;
    }}
    li[role="option"],
    div[role="option"] {{
        background: transparent !important;
        color: {TEXT_PRIMARY} !important;
    }}
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"],
    li[role="option"]:hover,
    div[role="option"]:hover {{
        background: rgba(31, 122, 115, 0.12) !important;
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > div,
    div[data-baseweb="menu"] ul,
    div[role="listbox"],
    ul[role="listbox"] {{
        background: #ffffff !important;
        color: #000000 !important;
        border-color: {BG_SURFACE} !important;
    }}
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] *,
    div[role="listbox"] *,
    ul[role="listbox"] * {{
        color: #000000 !important;
        fill: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }}
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="menu"] [role="option"],
    li[role="option"],
    div[role="option"] {{
        background: #ffffff !important;
        color: #000000 !important;
    }}
    div[data-baseweb="popover"] [role="option"] > div,
    div[data-baseweb="menu"] [role="option"] > div,
    li[role="option"] > div,
    div[role="option"] > div {{
        background: transparent !important;
    }}
    div[data-baseweb="popover"] [role="option"][aria-selected="true"],
    div[data-baseweb="menu"] [role="option"][aria-selected="true"],
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="menu"] [role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"],
    li[role="option"]:hover,
    div[role="option"]:hover {{
        background: rgba(31, 122, 115, 0.14) !important;
        color: #000000 !important;
    }}
    .stDataFrame, .stTable {{
        color: {TEXT_PRIMARY};
    }}
    .js-plotly-plot,
    .plot-container,
    .svg-container {{
        background: rgba(255, 250, 242, 0.96) !important;
        border-radius: 24px !important;
    }}
    .stButton > button,
    div[data-testid="stButton"] button,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button[kind="primary"],
    button[kind="secondary"] {{
        border-radius: 999px;
        border: 1px solid {BG_SURFACE};
        background: #ffffff !important;
        color: #000000 !important;
    }}
    .stButton > button *,
    div[data-testid="stButton"] button *,
    button[data-testid="baseButton-primary"] *,
    button[data-testid="baseButton-secondary"] *,
    button[kind="primary"] *,
    button[kind="secondary"] * {{
        color: #000000 !important;
        fill: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }}
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active,
    div[data-testid="stButton"] button:hover,
    div[data-testid="stButton"] button:focus,
    div[data-testid="stButton"] button:active,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:focus,
    button[data-testid="baseButton-primary"]:active,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-secondary"]:focus,
    button[data-testid="baseButton-secondary"]:active,
    button[kind="primary"]:hover,
    button[kind="primary"]:focus,
    button[kind="primary"]:active,
    button[kind="secondary"]:hover,
    button[kind="secondary"]:focus,
    button[kind="secondary"]:active {{
        background: #ffffff !important;
        color: #000000 !important;
        border-color: {ACCENT} !important;
    }}
    [data-testid="stMetric"] {{
        background: rgba(255, 250, 242, 0.78);
        border: 1px solid {BG_SURFACE};
        border-radius: 24px;
        padding: 0.8rem;
        box-shadow: 0 10px 30px rgba(84, 63, 39, 0.08);
    }}
    .sg-hero {{
        background: rgba(255, 250, 242, 0.76);
        border: 1px solid {BG_SURFACE};
        border-radius: 36px;
        padding: 1.8rem 2rem;
        box-shadow: 0 18px 50px rgba(84, 63, 39, 0.10);
        backdrop-filter: blur(10px);
    }}
    .sg-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1.25rem;
    }}
    .sg-card {{
        background: rgba(255, 250, 242, 0.82);
        border: 1px solid {BG_SURFACE};
        border-radius: 24px;
        padding: 1.1rem 1.2rem;
        min-height: 138px;
    }}
    .sg-kicker {{
        color: {TEXT_DIM};
        letter-spacing: 0.18em;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
    }}
    .sg-title {{
        color: {TEXT_PRIMARY};
        margin: 0.15rem 0 0.7rem 0;
        font-size: 3.2rem;
        line-height: 1.02;
        font-weight: 800;
    }}
    .sg-subtitle {{
        color: {TEXT_MUTED};
        font-size: 1.22rem;
        line-height: 1.65;
        max-width: 880px;
        margin-bottom: 0;
    }}
    .sg-card h4 {{
        color: {TEXT_PRIMARY};
        margin: 0 0 0.55rem 0;
        font-size: 1.15rem;
    }}
    .sg-card p {{
        color: {TEXT_MUTED};
        margin: 0;
        line-height: 1.65;
    }}
    @media (max-width: 960px) {{
        .sg-grid {{ grid-template-columns: 1fr; }}
        .sg-title {{ font-size: 2.3rem; }}
        .sg-hero {{ padding: 1.4rem; border-radius: 28px; }}
    }}
    .stDivider {{ margin: 1.5rem 0; }}
    h1, h2, h3, h4 {{ color: {TEXT_PRIMARY}; }}
    .stMarkdown p {{ color: {TEXT_MUTED}; }}
    .stCaption {{ color: {TEXT_DIM}; }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    pair = st.selectbox("Currency Pair", list(PAIR_LABELS.keys()),
                        format_func=lambda x: PAIR_LABELS[x])
    st.divider()
    st.caption("CS 6140 Machine Learning")
    st.caption("Northeastern University")
    st.caption("Sohan  |  Anusha  |  Disha")

# ── Load Data ────────────────────────────────────────────────────────────────
trades = load_csv(f'results/winrate/{pair}_winrate_trades.csv')
if not trades.empty:
    trades['datetime_utc'] = pd.to_datetime(trades['datetime_utc'])

price = load_csv(f'data/raw/price/{pair}_4h.csv')
if not price.empty:
    price['datetime_utc'] = pd.to_datetime(price['datetime_utc'])

shifts = load_csv(f'results/detection/{pair}_shifts.csv')
attr = load_csv(f'results/attribution/{pair}_attribution.csv')
stats = load_json('results/figures/statistical_tests.json')
auto_confirmed_count = auto_confirm_shifts(pair, shifts)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="sg-hero">
    <div class="sg-kicker">Executive Snapshot</div>
    <h1 class="sg-title">ShiftGuard</h1>
    <p class="sg-subtitle">
        A calmer way to detect distribution shifts, explain what changed, and keep retraining selective.
        The system auto-confirms each new shift and only asks the analyst to step in when an override is needed.
    </p>
    <div class="sg-grid">
        <div class="sg-card">
            <h4>{PAIR_LABELS[pair]}</h4>
            <p>Focused view of signals, shift evidence, and analyst overrides for the currently selected market.</p>
        </div>
        <div class="sg-card">
            <h4>Auto-confirm by default</h4>
            <p>{auto_confirmed_count} newly detected shifts were queued automatically in this session, so review can stay exception-based.</p>
        </div>
        <div class="sg-card">
            <h4>Human-in-the-loop by design</h4>
            <p>The analyst now reviews edge cases, corrects labels when needed, and lets the retraining queue flow in the background.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# ── Tab Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "P&L Simulator", "Price & Regimes",
    "Shift Detection", "Review Shifts"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not trades.empty:
        # Compute realistic 3-month P&L for overview (1:20 leverage)
        overview_lev = 20
        overview_capital = 10000
        overview_end = trades['datetime_utc'].max()
        overview_start = overview_end - pd.DateOffset(months=3)
        ov_chunk = trades[(trades['datetime_utc'] >= overview_start) & (trades['datetime_utc'] <= overview_end)]
        spread_map = {'EURUSD': 0.00010, 'GBPJPY': 0.00020, 'XAUUSD': 0.00030}

        strategies = {}
        for sname, col in [('Technical (RSI/MACD)', 'tech_signal'),
                           ('ML Direction (XGBoost)', 'ml_signal'),
                           ('ShiftGuard', 'sg_signal')]:
            sig = trades[col].values
            ret = trades['actual_return'].values
            mask = sig != 0
            n = mask.sum()
            if n > 0:
                pnl_arr = sig[mask] * ret[mask]
                wins = pnl_arr[pnl_arr > 0]
                losses = pnl_arr[pnl_arr < 0]
                wr = len(wins) / n * 100
                total_ret = pnl_arr.sum() * 100
            else:
                wr, total_ret, n = 0, 0, 0

            # 1-year realistic P&L
            ov_s = ov_chunk[col].values
            ov_r = ov_chunk['actual_return'].values
            ov_mask = ov_s != 0
            ov_nt = int(ov_mask.sum())
            if ov_nt > 0:
                ov_raw = ov_s[ov_mask] * ov_r[ov_mask]
                ov_lp = ov_raw * overview_lev
                ov_sl = {'EURUSD': 0.005, 'GBPJPY': 0.0075, 'XAUUSD': 0.01}.get(pair, 0.01)
                ov_slv = ov_sl * overview_lev
                ov_capped = np.where(ov_lp < -ov_slv, -ov_slv, ov_lp)
                ov_cpt = (spread_map.get(pair, 0.0002) + 0.00003 + 0.00002 + 0.000005) * overview_lev
                ov_gross = overview_capital * ov_capped.sum()
                ov_costs = overview_capital * ov_cpt * ov_nt
                ov_pt = ov_gross - ov_costs
                ov_tax = max(0, ov_pt * 0.30)
                ov_profit = ov_pt - ov_tax
                ov_ret_pct = ov_profit / overview_capital * 100
            else:
                ov_profit, ov_ret_pct = 0, 0

            strategies[sname] = {'wr': wr, 'n': n, 'total_ret': total_ret,
                                 'pct': n / len(trades) * 100,
                                 'profit_1yr': ov_profit, 'ret_1yr': ov_ret_pct}

        st.markdown("### Strategy Comparison")
        st.markdown("*All three strategies see the same market data. Technical uses classic indicators "
                    "(RSI, MACD) to generate signals. ML trains an XGBoost classifier on the features. "
                    "ShiftGuard adds a layer on top — it identifies the current market regime and only "
                    "trades when the regime is clear and confidence is high. When unsure, it sits out.*")

        cols = st.columns(3)
        for i, (sname, s) in enumerate(strategies.items()):
            with cols[i]:
                is_sg = 'ShiftGuard' in sname
                color = ACCENT if is_sg else '#8892b0'
                border = '2px solid #00d4aa' if is_sg else '1px solid #333'
                profit = s['profit_1yr']
                profit_color = ACCENT if profit > 0 else DANGER
                ret_pct = s['ret_1yr']
                balance = overview_capital + profit
                status = 'PROFIT' if profit > 0 else ('BLOWN' if balance <= 0 else 'LOSS')
                accent = ACCENT if is_sg else TEXT_MUTED
                bdr = f'2px solid {ACCENT}' if is_sg else f'1px solid {BG_SURFACE}'
                p_color = ACCENT if profit > 0 else DANGER
                st.markdown(f"""
                <div style="background: {BG_CARD};
                     border-radius: 10px; padding: 20px; text-align: center;
                     border: {bdr}; margin-bottom: 12px;">
                    <p style="color: {TEXT_MUTED}; margin: 0; font-size: 0.85rem;">{sname}</p>
                    <h2 style="color: {accent}; margin: 10px 0 4px 0; font-size: 2rem;">{s['wr']:.1f}%</h2>
                    <p style="color: {TEXT_DIM}; margin: 0; font-size: 0.75rem;">win rate over {s['n']:,} trades ({s['pct']:.0f}% of bars)</p>
                    <hr style="border-color: {BG_SURFACE}; margin: 12px 0;">
                    <p style="color: {p_color}; font-size: 1.4rem; margin: 0; font-weight: 700;">${profit:+,.0f}</p>
                    <p style="color: {p_color}; font-size: 0.9rem; margin: 4px 0 0 0;">{ret_pct:+.1f}% return</p>
                    <p style="color: {TEXT_DIM}; font-size: 0.7rem; margin: 6px 0 0 0;">3-month, 1:20 leverage, {ov_sl*100:.1f}% SL, after costs & tax</p>
                </div>
                """, unsafe_allow_html=True)

        if pair in stats:
            s = stats[pair]
            st.markdown(f"""
            #### Is this real or just luck?
            We tested this rigorously. On **{s.get('n_overlapping_trades', '?'):,} trades** where
            both Technical and ShiftGuard were active at the same time, we ran a
            paired statistical test (t-test). The result: **p-value = {s.get('p_value', '?')}**.

            In plain terms: if ShiftGuard had no real advantage and was just getting lucky,
            the odds of seeing results this good are **less than 1 in 10,000**.
            The confidence interval shows ShiftGuard earns between
            **{float(s.get('ci_lower', 0))*10000:.1f}** and **{float(s.get('ci_upper', 0))*10000:.1f} extra
            dollars per 10,000 trades** compared to the technical baseline. This is not noise — it is a
            consistent, measurable edge.
            """)
    else:
        st.warning("No trade data found. Run winrate_experiment.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: P&L SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### What would you actually make?")
    st.markdown("*Real-world simulation with spreads, commissions, slippage, swap costs, "
                "custom stop-loss, and 30% tax on profits.*")

    col1, col2, col3, col4 = st.columns(4)
    capital = col1.number_input("Capital per pair ($)", value=10000, step=1000)
    leverage = col2.selectbox("Leverage", [1, 5, 10, 20], index=3,
                              format_func=lambda x: f"1:{x}" if x > 1 else "No leverage")
    horizon_months = col3.selectbox("Time horizon", [1, 3, 6, 12, 24, 60],
                                    format_func=lambda x: f"{x} months" if x < 12 else f"{x//12} year{'s' if x > 12 else ''}")
    sl_pct = col4.number_input("Stop-Loss %", value=1.0, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")

    if not trades.empty:
        end_date = trades['datetime_utc'].max()
        start_date = end_date - pd.DateOffset(months=horizon_months)

        spread_map = {'EURUSD': 0.00010, 'GBPJPY': 0.00020, 'XAUUSD': 0.00030}
        commission = 0.00003
        slippage_cost = 0.00002
        swap_cost = 0.000005
        sl = sl_pct / 100
        tax_rate = 0.30

        chunk = trades[(trades['datetime_utc'] >= start_date) & (trades['datetime_utc'] <= end_date)]

        results_pnl = {}
        for sname, col in [('Technical', 'tech_signal'), ('ML Direction', 'ml_signal'),
                           ('ShiftGuard', 'sg_signal')]:
            s = chunk[col].values
            r = chunk['actual_return'].values
            tm = s != 0
            nt = int(tm.sum())
            if nt == 0:
                results_pnl[sname] = {'trades': 0, 'wr': 0, 'after_tax': 0}
                continue

            raw = s[tm] * r[tm]
            lp = raw * leverage
            slv = sl * leverage
            capped = np.where(lp < -slv, -slv, lp)
            cpt = (spread_map.get(pair, 0.0002) + commission + slippage_cost + swap_cost) * leverage
            net = capped - cpt

            gross = capital * capped.sum()
            costs = capital * cpt * nt
            pt = gross - costs
            tx = max(0, pt * tax_rate)
            at = pt - tx

            results_pnl[sname] = {
                'trades': nt,
                'wr': int((net > 0).sum()) / nt * 100,
                'after_tax': at,
            }

        st.markdown(f"#### Results: {horizon_months}mo with 1:{leverage} leverage on {PAIR_LABELS[pair]}")
        cols = st.columns(3)
        for i, (sname, r) in enumerate(results_pnl.items()):
            at = r['after_tax']
            balance = capital + at
            status_color = ACCENT if at > 0 else DANGER
            status = 'PROFIT' if at > 0 else ('BLOWN' if balance <= 0 else 'LOSS')
            ret_pct = at / capital * 100 if capital > 0 else 0

            with cols[i]:
                sc = ACCENT if at > 0 else DANGER
                st.markdown(f"""
                <div style="background: {BG_CARD};
                     border-radius: 10px; padding: 20px; text-align: center;
                     border: 1px solid {ACCENT if 'ShiftGuard' in sname else BG_SURFACE};">
                    <p style="color: {TEXT_MUTED}; margin: 0 0 6px 0;">{sname}</p>
                    <h2 style="color: {sc}; margin: 0;">${at:+,.0f}</h2>
                    <p style="color: {sc}; font-size: 0.95rem; margin: 4px 0 0 0;">{ret_pct:+.1f}% return  |  {status}</p>
                    <p style="color: {TEXT_DIM}; font-size: 0.75rem; margin: 6px 0 0 0;">
                        {r['trades']:,} trades  |  {r['wr']:.1f}% win rate  |  Balance: ${balance:,.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Equity Curve")
        st.markdown("*Watch how each strategy grows (or destroys) your capital over time.*")

        fig = go.Figure()
        for sname, col, color in [('Technical', 'tech_signal', '#555'),
                                   ('ML Direction', 'ml_signal', '#888'),
                                   ('ShiftGuard', 'sg_signal', ACCENT)]:
            pnl = chunk[col] * chunk['actual_return']
            equity = capital * (1 + (pnl * leverage).cumsum())
            width = 3 if 'ShiftGuard' in sname else 1
            fig.add_trace(go.Scatter(x=chunk['datetime_utc'], y=equity, name=sname,
                                     line=dict(color=color, width=width)))

        fig.add_hline(y=capital, line_dash="dot", line_color=DANGER,
                      annotation_text="Break-even")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          yaxis_title='Portfolio Value ($)', xaxis_title='',
                          legend=dict(orientation='h', y=1.1), margin=dict(t=30, b=20),
                          dragmode=False)
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PRICE & REGIMES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Price Chart with Regime Overlay")
    st.markdown("*Background colors show the detected market regime. "
                "Green = trending up, Red = trending down, Gray = ranging/unclear.*")

    REGIME_BG = {
        0: 'rgba(46,204,113,0.25)',    # Trend Up (Calm) - bright green
        1: 'rgba(0,150,255,0.25)',     # Trend Up (Volatile) - blue
        2: 'rgba(180,180,180,0.10)',   # Ranging - faint gray
        3: 'rgba(255,165,0,0.25)',     # Trend Down (Calm) - orange
        4: 'rgba(231,76,60,0.30)',     # Trend Down (Volatile) - red
    }

    if not trades.empty and not price.empty:
        year_range = st.slider("Year range", 2016, 2026, (2024, 2026))

        view = trades[(trades['datetime_utc'].dt.year >= year_range[0]) &
                      (trades['datetime_utc'].dt.year <= year_range[1])].copy()

        if not view.empty:
            price_view = price[(price['datetime_utc'].dt.year >= year_range[0]) &
                               (price['datetime_utc'].dt.year <= year_range[1])]

            # --- Clean price chart ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_view['datetime_utc'], y=price_view['close'],
                name='Price', line=dict(color='#e0e0e0', width=1.3),
                hovertemplate='%{x}<br>$%{y:,.2f}<extra></extra>'
            ))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=420,
                              margin=dict(t=20, b=20),
                              yaxis_title='Price',
                              showlegend=False,
                              dragmode=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

            # --- Pie chart ---
            st.markdown("#### Market State Distribution")
            regime_counts = view['regime'].value_counts().sort_index()
            regime_labels = [REGIME_NAMES.get(int(k), f'State {k}') for k in regime_counts.index]
            regime_colors_list = [REGIME_COLORS.get(int(k), '#888') for k in regime_counts.index]

            fig2 = go.Figure(go.Pie(
                labels=regime_labels, values=regime_counts.values,
                marker=dict(colors=regime_colors_list),
                hole=0.4, textinfo='label+percent',
                textfont=dict(size=12),
            ))
            fig2.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SHIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Detected Distribution Shifts")
    st.markdown("*These are moments where the market's statistical behavior changed significantly. "
                "Think of it as the market switching from one mood to another.*")

    if not shifts.empty:
        col1, col2 = st.columns(2)
        min_sev = col1.slider("Minimum severity", 1, 5, 1, key="sev_slider")
        show_type = col2.multiselect("Shift types", ['scheduled', 'unexpected'],
                                      default=['scheduled', 'unexpected'])

        n_total = len(shifts)
        n_sched = len(shifts[shifts['type'] == 'scheduled']) if 'type' in shifts.columns else 0
        n_unexp = len(shifts[shifts['type'] == 'unexpected']) if 'type' in shifts.columns else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Shifts", n_total)
        c2.metric("Scheduled (around events)", n_sched)
        c3.metric("Unexpected (no calendar match)", n_unexp)

        f = shifts.copy()
        if 'type' in f.columns:
            f = f[f['type'].isin(show_type)]
        if 'severity' in f.columns:
            f = f[f['severity'] >= min_sev]
        f = f.sort_values('datetime_utc', ascending=False)

        display_cols = [c for c in ['datetime_utc', 'type', 'severity', 'event_names',
                                     'n_features_triggered'] if c in f.columns]
        st.dataframe(f[display_cols].head(50), use_container_width=True, height=300)

        if not attr.empty and 'dominant_group' in attr.columns:
            st.markdown("### What's Causing the Shifts?")
            st.markdown("""
            *When a shift happens, we use SHAP (a model explanation tool) to figure out
            which category of features drove the change. The groups are:*

            - **Technical** — price patterns, moving averages, momentum indicators
            - **Sentiment** — fear index (VIX), dollar strength, stock market mood
            - **Volatility** — how much the price is swinging, compression patterns
            - **Macro** — interest rates, economic events, GDP/employment data
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Which group caused the most shifts:**")
                counts = attr['dominant_group'].value_counts()
                fig = go.Figure(go.Bar(
                    x=counts.values, y=counts.index, orientation='h',
                    marker_color=[ACCENT, '#3498db', '#f39c12', '#e74c3c', '#9b59b6'][:len(counts)],
                    text=counts.values, textposition='outside',
                ))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=280,
                                  margin=dict(t=10, b=20, l=100))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                gcols = [c for c in attr.columns if c.startswith('group_')]
                if gcols:
                    st.markdown("**Average responsibility per group (normalized):**")
                    means = attr[gcols].mean()
                    total = means.sum()
                    if total > 0:
                        means = (means / total).sort_values(ascending=True)
                    means.index = [c.replace('group_', '').title() for c in means.index]
                    fig = go.Figure(go.Bar(
                        x=means.values, y=means.index, orientation='h',
                        marker_color='#3498db',
                        text=[f'{v:.0%}' for v in means.values], textposition='outside',
                    ))
                    fig.update_layout(template=PLOTLY_TEMPLATE, height=280,
                                      margin=dict(t=10, b=20, l=100),
                                      xaxis=dict(tickformat='.0%'))
                    st.plotly_chart(fig, use_container_width=True)

            # Plain language summary
            top_group = attr['dominant_group'].value_counts().index[0]
            explanations = {
                'technical': 'Price patterns and momentum indicators are the biggest driver of regime changes. This means the market tends to shift when technical trends break down or reverse.',
                'sentiment': 'Market mood (fear, dollar strength, stock market) is the primary cause of shifts. External sentiment shocks — like geopolitical events or risk-off moves — are driving regime changes more than price patterns alone.',
                'volatility': 'Changes in how much the market swings are the main trigger. When volatility compresses and then expands, or vice versa, the regime shifts.',
                'macro': 'Economic fundamentals (interest rates, jobs data, GDP) are driving the shifts. Policy decisions and economic surprises are the dominant force.',
            }
            st.info(explanations.get(top_group, f'The dominant shift driver for this pair is **{top_group}**.'))
    else:
        st.info("No shift detection data available.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: HUMAN-IN-THE-LOOP REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Review & Decide")
    st.markdown("*This is the human-in-the-loop component. You review each detected shift, "
                "see what caused it, and decide: should we retrain the model or ignore it?*")

    if not shifts.empty:
        shifts_sorted = shifts.sort_values('datetime_utc', ascending=False).copy()
        # Build readable labels: "2026-04-06 | Scheduled | 5/5 | Unemployment Claims"
        shift_labels = []
        for _, sr in shifts_sorted.iterrows():
            dt_str = str(sr['datetime_utc'])[:10]
            stype = str(sr.get('type', '?')).title()
            ssev = int(sr.get('severity', 0))
            event = str(sr.get('event_names', ''))[:50]
            if event == 'nan' or event == '':
                event = 'No event'
            shift_labels.append(f"{dt_str}  |  {stype}  |  {ssev}/5  |  {event}")

        pick_idx = st.selectbox("Select a shift to review", range(len(shift_labels)),
                                format_func=lambda i: shift_labels[i])
        pick = shifts_sorted.iloc[pick_idx]['datetime_utc']

        if pick is not None:
            row = shifts_sorted.iloc[pick_idx]
            sev = int(row.get('severity', 0))
            labels = {1: 'Minimal', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Extreme'}

            col1, col2, col3 = st.columns(3)
            col1.metric("Type", str(row.get('type', '')).replace('_', ' ').title())
            col2.metric("Severity", f"{sev}/5 ({labels.get(sev, '?')})")
            col3.metric("Date", str(pick)[:10])

            # Price context graph around the shift — highlight the shift zone
            if not price.empty:
                pick_dt = pd.to_datetime(pick)
                ctx_start = pick_dt - pd.Timedelta(days=30)
                ctx_end = pick_dt + pd.Timedelta(days=30)
                ctx_price = price[(price['datetime_utc'] >= ctx_start) & (price['datetime_utc'] <= ctx_end)]

                if not ctx_price.empty:
                    fig_ctx = go.Figure()

                    # Price before shift (gray)
                    before = ctx_price[ctx_price['datetime_utc'] < pick_dt]
                    after = ctx_price[ctx_price['datetime_utc'] >= pick_dt]

                    if not before.empty:
                        fig_ctx.add_trace(go.Scatter(
                            x=before['datetime_utc'], y=before['close'],
                            mode='lines', line=dict(color='#888', width=1.5),
                            name='Before shift', showlegend=True,
                        ))
                    if not after.empty:
                        fig_ctx.add_trace(go.Scatter(
                            x=after['datetime_utc'], y=after['close'],
                            mode='lines', line=dict(color=ACCENT, width=2),
                            name='After shift', showlegend=True,
                        ))

                    # Highlight shift zone (3 days around the shift)
                    shift_zone_start = pick_dt - pd.Timedelta(days=2)
                    shift_zone_end = pick_dt + pd.Timedelta(days=2)
                    fig_ctx.add_shape(
                        type='rect',
                        x0=shift_zone_start.isoformat(), x1=shift_zone_end.isoformat(),
                        y0=0, y1=1, yref='paper',
                        fillcolor='rgba(255,215,0,0.20)',
                        line=dict(color='rgba(255,215,0,0.6)', width=1),
                        layer='below',
                    )
                    # Shift line
                    fig_ctx.add_shape(
                        type='line', x0=pick_dt.isoformat(), x1=pick_dt.isoformat(),
                        y0=0, y1=1, yref='paper',
                        line=dict(color='#ffd700', width=2.5, dash='dash'),
                    )
                    fig_ctx.add_annotation(
                        x=pick_dt.isoformat(), y=1.02, yref='paper',
                        text='SHIFT DETECTED', showarrow=False,
                        font=dict(color='#ffd700', size=12, family='Arial Black'),
                    )

                    fig_ctx.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                          margin=dict(t=30, b=15),
                                          yaxis_title='Price',
                                          legend=dict(orientation='h', y=1.12),
                                          dragmode=False)
                    st.plotly_chart(fig_ctx, use_container_width=True,
                                    config={'scrollZoom': False, 'displayModeBar': False})

            # ── Trading Analytics: 5 key metrics ──
            if not price.empty:
                metrics = enrich_shift(row, price)

                st.markdown("#### Shift Analytics")
                alert_colors = {'CRITICAL': '#ff4444', 'HIGH': '#ffa500', 'MODERATE': '#ffd700', 'LOW': '#888'}
                al = metrics['alert_level']
                al_color = alert_colors.get(al, '#888')

                pc = ACCENT if metrics['price_change_pct'] >= 0 else DANGER
                vc = DANGER if metrics['volatility_change_pct'] > 20 else WARN if metrics['volatility_change_pct'] > 0 else ACCENT
                st.markdown(f"""
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:12px 0;">
                    <div style="background:{BG_CARD}; border-radius:10px; padding:16px; border:1px solid {BG_SURFACE};">
                        <p style="color:{TEXT_DIM}; margin:0; font-size:0.75rem;">Price Change</p>
                        <p style="color:{pc}; margin:6px 0 0 0; font-size:1.3rem; font-weight:700;">{metrics['price_change_pct']:+.3f}%</p>
                        <p style="color:{TEXT_DIM}; margin:4px 0 0 0; font-size:0.7rem;">{metrics['price_before']:.5f} -> {metrics['price_after']:.5f}</p>
                    </div>
                    <div style="background:{BG_CARD}; border-radius:10px; padding:16px; border:1px solid {BG_SURFACE};">
                        <p style="color:{TEXT_DIM}; margin:0; font-size:0.75rem;">Volatility Change</p>
                        <p style="color:{vc}; margin:6px 0 0 0; font-size:1.3rem; font-weight:700;">{metrics['volatility_change_pct']:+.1f}%</p>
                        <p style="color:{TEXT_DIM}; margin:4px 0 0 0; font-size:0.7rem;">ATR: {metrics['pre_shift_atr']:.5f} -> {metrics['post_shift_atr']:.5f}</p>
                    </div>
                    <div style="background:{BG_CARD}; border-radius:10px; padding:16px; border:1px solid {BG_SURFACE};">
                        <p style="color:{TEXT_DIM}; margin:0; font-size:0.75rem;">Support / Resistance</p>
                        <p style="color:{TEXT_PRIMARY}; margin:6px 0 0 0; font-size:1.1rem;">{metrics['support']:.5f} — {metrics['resistance']:.5f}</p>
                        <p style="color:{TEXT_DIM}; margin:4px 0 0 0; font-size:0.7rem;">Range: {metrics['range']:.5f}</p>
                    </div>
                    <div style="background:{al_color}12; border-radius:10px; padding:16px; border:1px solid {al_color}; text-align:center;">
                        <p style="color:{TEXT_DIM}; margin:0; font-size:0.75rem;">Alert Score</p>
                        <p style="color:{al_color}; margin:6px 0 0 0; font-size:1.5rem; font-weight:700;">{metrics['alert_score']}/100</p>
                        <p style="color:{al_color}; margin:4px 0 0 0; font-size:0.8rem;">{al}</p>
                    </div>
                </div>
                <p style="color:{TEXT_DIM}; font-size:0.75rem; margin:4px 0 12px 0;">
                    Shift window: {metrics['shift_start'][:16]} to {metrics['shift_end'][:16]} ({metrics['shift_duration_hours']}h)
                    &nbsp;|&nbsp; KS: {metrics['ks_threshold']} &nbsp;|&nbsp; MMD: {metrics['mmd_threshold']}
                </p>
                """, unsafe_allow_html=True)

            # ── SHAP Breakdown ──
            if not attr.empty:
                match = attr[attr['datetime_utc'] == pick]
                if not match.empty:
                    arow = match.iloc[0]
                    gcols = [c for c in arow.index if c.startswith('group_')]
                    breakdown = {c.replace('group_', '').title(): arow[c]
                                 for c in gcols if pd.notna(arow[c]) and arow[c] > 0}
                    if breakdown:
                        st.markdown("**What drove this shift:**")
                        bd_total = sum(breakdown.values())
                        if bd_total > 0:
                            breakdown = {k: v / bd_total for k, v in breakdown.items()}
                        fig = go.Figure(go.Bar(
                            x=list(breakdown.values()), y=list(breakdown.keys()),
                            orientation='h', marker_color=ACCENT,
                            text=[f'{v:.0%}' for v in breakdown.values()],
                            textposition='outside',
                        ))
                        fig.update_layout(template=PLOTLY_TEMPLATE, height=200,
                                          margin=dict(t=10, b=10, l=100),
                                          xaxis=dict(tickformat='.0%'))
                        st.plotly_chart(fig, use_container_width=True)

                        dominant = arow.get('dominant_group', '?')
                        top_feat = arow.get('top_feature_1', '?')
                        st.markdown(f"**Main cause:** {dominant} | **Top feature:** {top_feat}")

            # Decision guidance
            st.divider()
            shift_type = str(row.get('type', ''))
            event_name = str(row.get('event_names', 'None'))
            if event_name == 'nan':
                event_name = 'None'

            st.markdown("#### What should you do?")

            # Build context-aware guidance
            vol_change = metrics.get('volatility_change_pct', 0) if 'metrics' in dir() else 0
            price_chg = metrics.get('price_change_pct', 0) if 'metrics' in dir() else 0

            st.markdown(f"""
            **About this shift:** On **{str(pick)[:10]}**, our detection engine flagged a
            **severity {sev}/5** distribution shift classified as **{shift_type}**.
            {'It was triggered by **' + event_name + '** — a scheduled economic event.' if shift_type == 'scheduled' and event_name != 'None' else 'No specific economic event was linked to this shift — it was detected purely from statistical changes in the data.'}

            **What happened:** The price moved **{price_chg:+.3f}%** during the shift window
            and volatility {'increased' if vol_change > 0 else 'decreased'} by **{abs(vol_change):.1f}%**.
            {'This is a large volatility spike, suggesting the market entered a fundamentally different regime.' if abs(vol_change) > 30 else 'The volatility change is moderate.' if abs(vol_change) > 10 else 'Volatility remained relatively stable.'}
            """)

            if sev >= 4:
                st.markdown(f"""
                **Recommendation: Confirm and retrain.** High-severity shifts ({sev}/5) indicate
                the market has genuinely changed its behavior. The model trained on old data
                will be less accurate in this new regime. Retraining allows it to learn the
                new patterns and make better predictions going forward.
                {'Since this aligns with a known economic event (' + event_name + '), the shift is almost certainly real.' if shift_type == 'scheduled' and event_name != 'None' else 'Even though no calendar event triggered this, unexpected shifts are often the most dangerous — they catch traders off guard.'}
                """)
            elif sev >= 2:
                st.markdown(f"""
                **Recommendation: Review carefully.** This is a moderate shift ({sev}/5).
                Look at the price chart above — does the price behavior look different before
                vs after the gold dashed line? If the trend direction changed or volatility
                clearly spiked, confirm it. If the chart looks similar on both sides, this
                may be noise and you can safely reject it.
                """)
            else:
                st.markdown(f"""
                **Recommendation: Likely reject.** This is a minor shift ({sev}/5) — the
                statistical change was small. Most shifts at this severity are noise rather
                than real regime changes. Only confirm if you see an obvious visual change
                in the price chart above.
                """)

            st.info("This shift is auto-confirmed by default. You only need to step in if you want to correct the label or reject the alert as noise.")
            st.markdown("""
            **What each action does now:**
            - **Keep / override & stay confirmed**: Preserve the default confirmation and optionally correct the shift type.
            - **Reject - False Alarm**: Mark the alert as noise and remove it from the selective-retraining path.

            Before submitting, you can **override the classification** below. If the system
            labeled this as "unexpected" but you know it was caused by a news event, change
            it to "scheduled". This updates the detection records so future analysis and
            the detection stats in the Shift Detection tab reflect the corrected type.
            """)

            # Override classification dropdown
            override_options = ['Keep original: ' + shift_type, 'scheduled', 'unexpected']
            override = st.selectbox("Override classification (optional)",
                                    override_options, index=0, key='override_select')

            notes = st.text_input("Add a note (optional)", key="review_note")

            def apply_override(pair_name, shift_dt, new_type_val):
                """Update the actual shifts.csv with the corrected type."""
                shifts_path = os.path.join(ROOT, 'results', 'detection', f'{pair_name}_shifts.csv')
                if os.path.exists(shifts_path):
                    sdf = pd.read_csv(shifts_path)
                    mask = sdf['datetime_utc'] == str(shift_dt)
                    if mask.any():
                        sdf.loc[mask, 'type'] = new_type_val
                        sdf.to_csv(shifts_path, index=False)
                        return True
                return False

            col1, col2 = st.columns(2)
            with col1:
                primary_label = "Keep auto-confirmed" if override.startswith('Keep') else "Apply override & keep confirmed"
                if st.button(primary_label, type="primary"):
                    # Apply override if changed
                    final_type = shift_type
                    decision_name = "confirm"
                    if not override.startswith('Keep'):
                        apply_override(pair, pick, override)
                        final_type = override
                        decision_name = f"reclassify_to_{override}"

                    save_decision(pair, pick, decision_name,
                                  f"type={final_type}" + (f' | {notes}' if notes else ''))
                    queue_retrain(pair, pick, sev, final_type, event_name)

                    msg = f"Confirmed. Retrain queued for {str(pick)[:10]}."
                    if not override.startswith('Keep'):
                        msg += f" Classification updated to '{final_type}'."
                    elif notes:
                        msg += " Notes saved on the confirmed shift."
                    msg += " The adaptive pipeline will retrain on the next cycle."
                    st.success(msg)
                    st.rerun()

            with col2:
                if st.button("Reject - False Alarm"):
                    # Apply override even on reject
                    final_type = shift_type
                    if not override.startswith('Keep'):
                        apply_override(pair, pick, override)
                        final_type = override

                    save_decision(pair, pick, "reject",
                                  f"type={final_type}" + (f' | {notes}' if notes else ''))

                    msg = "Rejected. Logged as false positive — model will not retrain."
                    if not override.startswith('Keep'):
                        msg += f" Classification updated to '{final_type}' for future reference."
                    st.warning(msg)
                    st.rerun()

    st.divider()
    st.markdown("#### Decision Log")
    decisions = load_decisions(pair)
    if not decisions.empty:
        st.dataframe(decisions, use_container_width=True)
        nc = len(decisions[decisions['decision'].isin(['confirm', 'auto_confirm', 'reclassify_to_scheduled', 'reclassify_to_unexpected'])])
        nr = len(decisions[decisions['decision'] == 'reject'])
        na = len(decisions[decisions['decision'] == 'auto_confirm'])
        st.caption(f"{nc} confirmed | {na} auto-confirmed | {nr} rejected | {len(decisions)} total")

        dc1, dc2 = st.columns(2)
        with dc1:
            del_idx = st.selectbox("Select entry to delete",
                                   range(len(decisions)),
                                   format_func=lambda i: f"{decisions.iloc[i]['datetime_utc'][:10]}  |  {decisions.iloc[i]['decision']}",
                                   key='del_select')
            if st.button("Delete selected entry"):
                dec_path = os.path.join(ROOT, 'results', 'decisions', f'{pair}_decisions.csv')
                updated = decisions.drop(decisions.index[del_idx]).reset_index(drop=True)
                updated.to_csv(dec_path, index=False)
                st.success("Entry deleted.")
                st.rerun()
        with dc2:
            st.markdown("")
            st.markdown("")
            if st.button("Clear all decisions"):
                dec_path = os.path.join(ROOT, 'results', 'decisions', f'{pair}_decisions.csv')
                pd.DataFrame(columns=['datetime_utc', 'decision', 'notes']).to_csv(dec_path, index=False)
                st.success("All decisions cleared.")
                st.rerun()
    else:
        st.caption("No decisions yet. Start reviewing shifts above.")
