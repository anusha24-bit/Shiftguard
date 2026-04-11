# ShiftGuard — Data Collection Instructions



---

## Overview

We need data across 5 categories for each currency pair. This is NOT just price data — we need every economic, fundamental, and sentiment factor that drives each pair. The goal is to build feature groups that our shift detection engine can trace back to (e.g., "this shift was caused by volatility factors" vs. "this shift was caused by macro news").

**Time Range**: January 2015 – March 2026 (covers Brexit, COVID-19, 2022 Fed hikes, 2023 banking crisis, 2024-25 rate cycles)
**Granularity**: Daily (primary), Hourly (required for session features — see Category 1B)

---

## Currency Pairs to Collect

### Primary (must have)
1. **EUR/USD** — Most liquid pair, driven by Fed vs ECB policy
2. **GBP/JPY** — High volatility cross, driven by BOE + BOJ + risk sentiment
3. **XAU/USD** — Gold, driven by completely different factors (safe haven, real yields, USD strength)

### Secondary (if time permits)
4. USD/JPY — BOJ policy divergence
5. GBP/USD — Brexit legacy, BOE policy

---

## CATEGORY 1: Price Data (OHLCV)

### 1A: Daily OHLCV (already collected ✅)
One CSV per pair: `EURUSD_ohlcv.csv`, `GBPJPY_ohlcv.csv`, `XAUUSD_ohlcv.csv`
Columns: `date, open, high, low, close, volume`

### 1B: Hourly OHLCV (NEW — needed for session features)
We need hourly bars to compute per-session features (Sydney/Asian/London/NY).

**Sources:**
- **yfinance**: `yf.download("EURUSD=X", interval="1h", period="730d")` — only last ~2 years
- **OANDA API** (free tier): Full historical hourly data back to 2015. **Best option.**
  - Sign up at https://www.oanda.com/us-en/trading/
  - Use `oandapyV20` Python library
- **Dukascopy** (free): https://www.dukascopy.com/swiss/english/marketwatch/historical/ — download hourly CSVs

**Columns needed:**
```
datetime_utc, open, high, low, close, volume
```
`datetime_utc` must include hour, e.g., `2020-03-09 14:00:00`

**Deliverable:**
One CSV per pair in `data/raw/price/hourly/`:
```
EURUSD_hourly.csv
GBPJPY_hourly.csv
XAUUSD_hourly.csv
```

**If only partial history available** (e.g., yfinance gives 2024-2026 only): That's OK. Collect what you can. Session features will be NaN for earlier years — the model handles this.

**Priority**: After all daily data is done. This is important but not blocking.

---

## CATEGORY 2: Economic Calendar Events

This is the most important non-price data. We need EVERY scheduled economic release that affects each currency in our pairs.

### Source
- **Investing.com Economic Calendar** (manual filter + export)
  - URL: https://www.investing.com/economic-calendar/
  - Filter by country + impact level (High and Medium)
- **ForexFactory Calendar** (https://www.forexfactory.com/calendar) — alternative
- **Python**: `investpy` library or scrape from the above

### What to collect per event
```
date, time_utc, currency, event_name, impact_level (high/medium/low), actual_value, forecast_value, previous_value
```

The **surprise** = actual - forecast. This is critical. Do NOT skip forecast values.

### Events needed PER CURRENCY:

#### USD (affects EUR/USD, XAU/USD, GBP/JPY indirectly)
- Non-Farm Payrolls (NFP)
- CPI (Consumer Price Index) — headline + core
- PPI (Producer Price Index)
- Fed Interest Rate Decision
- FOMC Meeting Minutes
- FOMC Press Conference dates
- Fed Chair Powell speeches
- GDP (Advance, Preliminary, Final)
- Retail Sales
- ISM Manufacturing PMI
- ISM Services PMI
- Unemployment Rate + Jobless Claims (weekly)
- Consumer Confidence (Conference Board)
- Michigan Consumer Sentiment
- Durable Goods Orders
- Housing Starts + Building Permits
- Existing Home Sales + New Home Sales
- Trade Balance
- Treasury Auction results (10Y, 30Y yields)
- ADP Employment Change

#### EUR (affects EUR/USD)
- ECB Interest Rate Decision
- ECB Press Conference dates
- ECB President Lagarde speeches
- Eurozone CPI (headline + core)
- Eurozone GDP
- German ZEW Economic Sentiment
- German IFO Business Climate
- Eurozone PMI (Manufacturing + Services + Composite)
- German Industrial Production
- Eurozone Retail Sales
- Eurozone Unemployment Rate
- Eurozone Trade Balance
- German CPI (often released before Eurozone CPI)

#### GBP (affects GBP/JPY)
- BOE Interest Rate Decision
- BOE Monetary Policy Summary
- BOE Governor speeches
- UK CPI (headline + core)
- UK GDP (monthly + quarterly)
- UK PMI (Manufacturing + Services)
- UK Retail Sales
- UK Unemployment Rate + Claimant Count
- UK Average Earnings
- UK Trade Balance
- UK Consumer Confidence (GfK)
- UK Housing Price Index

#### JPY (affects GBP/JPY, USD/JPY)
- BOJ Interest Rate Decision
- BOJ Monetary Policy Statement
- BOJ Governor speeches (Ueda / Kuroda pre-2023)
- Japan CPI (headline + core)
- Japan GDP
- Tankan Manufacturing Index (quarterly, very important)
- Japan PMI (Manufacturing + Services)
- Japan Trade Balance
- Japan Industrial Production
- Japan Retail Sales
- Japan Unemployment Rate
- Japan Machine Orders
- BOJ Yield Curve Control (YCC) policy changes (flag as binary events)

### Deliverable
One CSV per currency: `USD_economic_calendar.csv`, `EUR_economic_calendar.csv`, `GBP_economic_calendar.csv`, `JPY_economic_calendar.csv`

---

## CATEGORY 3: Macro Fundamental Data (Time Series)

These are continuous daily/monthly time series, NOT one-off events.

### Sources
- **FRED API** (free, Python: `fredapi`): Best for US data
- **World Bank / OECD**: International data
- **Investing.com**: Bond yields, rates

### Data needed

#### Interest Rates (daily or per meeting)
- US Federal Funds Rate (FRED: DFF)
- ECB Main Refinancing Rate
- BOE Bank Rate
- BOJ Policy Rate
- US 10-Year Treasury Yield (FRED: DGS10) — daily
- US 2-Year Treasury Yield (FRED: DGS2) — daily
- US 10Y-2Y Spread (yield curve, compute from above)
- German 10-Year Bund Yield — daily
- UK 10-Year Gilt Yield — daily
- Japan 10-Year JGB Yield — daily

#### Computed differentials (we'll compute these, but raw rates are needed)
- Fed Rate - ECB Rate (for EUR/USD)
- BOE Rate - BOJ Rate (for GBP/JPY)
- US 10Y - German 10Y (for EUR/USD)
- UK 10Y - Japan 10Y (for GBP/JPY)

#### Inflation (monthly)
- US CPI YoY (FRED: CPIAUCSL)
- Eurozone HICP YoY
- UK CPI YoY
- Japan CPI YoY
- US Core PCE (Fed's preferred measure, FRED: PCEPILFE)

#### Employment (monthly)
- US Unemployment Rate (FRED: UNRATE)
- Eurozone Unemployment Rate
- UK Unemployment Rate
- Japan Unemployment Rate

#### GDP Growth (quarterly)
- US GDP QoQ annualized
- Eurozone GDP QoQ
- UK GDP QoQ
- Japan GDP QoQ

### Deliverable
One CSV per data type: `interest_rates_daily.csv`, `inflation_monthly.csv`, `employment_monthly.csv`, `gdp_quarterly.csv`

All with columns: `date, metric_name, value`

---

## CATEGORY 4: Sentiment & Market Risk Data

### Sources
- **yfinance / FRED**: VIX, DXY
- **GDELT Project** (free): Global news event counts
- **NewsAPI** (free tier, 100 req/day): Headline counts

### Data needed (daily time series)

#### Risk / Fear Gauges
- **VIX** (CBOE Volatility Index) — `yf.download("^VIX")` — daily close
- **MOVE Index** (bond market volatility) — from FRED or investing.com
- **DXY** (US Dollar Index) — `yf.download("DX-Y.NYB")` — critical for all USD pairs

#### Gold-Specific Factors (for XAU/USD only)
- **US Real Yield** = US 10Y Yield - US CPI YoY (compute from Category 3 data)
- **Gold ETF Holdings** (SPDR GLD) — weekly, from World Gold Council or GLD website
- **CFTC Commitment of Traders (COT)** — Net long/short positions for gold futures, weekly
  - Source: https://www.cftc.gov/dea/futures/deacmxsf.htm
- **DXY** (already listed above, but inversely correlated with gold)
- **US M2 Money Supply** (FRED: M2SL) — monthly, monetary policy proxy
- **Central bank gold purchases** (quarterly, from World Gold Council)
- **Crude Oil prices** (WTI) — `yf.download("CL=F")` — inflation/commodity correlation

#### Cross-Asset (affects risk sentiment for all pairs)
- **S&P 500** — `yf.download("^GSPC")` — daily close, risk-on/risk-off proxy
- **US 10Y yield** (already in Category 3)
- **Crude Oil (WTI)** — `yf.download("CL=F")`
- **Bitcoin** (optional) — `yf.download("BTC-USD")` — has become a risk sentiment proxy

#### News Volume / Sentiment (simplified approach)
- **Option A (easier)**: Use GDELT to count daily "forex" or "economy" related articles per country. No NLP needed, just event counts.
  - GDELT API: https://api.gdeltproject.org/api/v2/doc/doc?query=forex&mode=timelinevol
- **Option B (better, more work)**: Use NewsAPI to pull headlines per day, we'll run FinBERT sentiment later.
  - Just save raw headlines + date + source, we'll process them.

### Deliverable
- `vix_dxy_daily.csv` — date, vix_close, dxy_close, sp500_close, oil_close
- `gold_specific_factors.csv` — date, gld_holdings, cot_net_long, us_real_yield, m2_money_supply
- `news_volume_daily.csv` — date, country, article_count (OR raw headlines CSV)

---

## CATEGORY 5: Geopolitical / Black Swan Event Log

This is a manually curated list of major unexpected events. We need this to validate that our "unexpected shift detector" catches them.

### Format
```
date, event_name, affected_currencies, severity (1-5), category
```

### Events to log (non-exhaustive, add any you find)

| Date | Event | Currencies | Severity |
|------|-------|-----------|----------|
| 2015-01-15 | SNB removes EUR/CHF floor | EUR, CHF | 5 |
| 2015-08-11 | China devalues Yuan | USD, JPY, AUD | 4 |
| 2016-06-23 | Brexit referendum | GBP, EUR | 5 |
| 2016-11-08 | Trump election | USD, JPY, MXN | 4 |
| 2018-02-05 | VIX spike / Volmageddon | All | 4 |
| 2019-08-05 | US-China trade war escalation | USD, CNY, JPY | 4 |
| 2020-03-09 | COVID crash / oil price war | All | 5 |
| 2020-03-15 | Fed emergency rate cut to 0% | USD, All | 5 |
| 2021-02-25 | US Treasury yield spike | USD, Gold | 3 |
| 2022-02-24 | Russia invades Ukraine | EUR, Gold, Oil | 5 |
| 2022-06-15 | Fed 75bps hike (first since 1994) | USD, All | 4 |
| 2022-09-22 | BOJ intervention in JPY | JPY | 5 |
| 2022-09-26 | UK mini-budget crisis (Truss) | GBP | 5 |
| 2023-03-10 | SVB collapse / banking crisis | USD, Gold | 4 |
| 2023-10-07 | Hamas-Israel conflict | Gold, Oil | 4 |
| 2024-07-31 | BOJ surprise rate hike | JPY | 5 |
| 2024-08-05 | JPY carry trade unwind | JPY, All | 5 |

Add any 2025-2026 events as needed.

### Deliverable
`geopolitical_events.csv`

---

## File Naming & Format Rules

1. **All files must be CSV**, UTF-8 encoded
2. **Date format**: `YYYY-MM-DD` (e.g., 2020-03-15)
3. **No missing date column** — every row must have a date
4. **NaN for missing values** — do not use empty strings, blanks, or "N/A"
5. **Column names**: lowercase, underscores, no spaces (e.g., `actual_value`, not `Actual Value`)
6. Put all files in a shared folder: `ShiftGuard/data/raw/`

## Folder Structure
```
ShiftGuard/
└── data/
    └── raw/
        ├── price/
        │   ├── EURUSD_ohlcv.csv
        │   ├── GBPJPY_ohlcv.csv
        │   ├── XAUUSD_ohlcv.csv
        │   └── hourly/
        │       ├── EURUSD_hourly.csv
        │       ├── GBPJPY_hourly.csv
        │       └── XAUUSD_hourly.csv
        ├── calendar/
        │   ├── USD_economic_calendar.csv
        │   ├── EUR_economic_calendar.csv
        │   ├── GBP_economic_calendar.csv
        │   └── JPY_economic_calendar.csv
        ├── macro/
        │   ├── interest_rates_daily.csv
        │   ├── inflation_monthly.csv
        │   ├── employment_monthly.csv
        │   └── gdp_quarterly.csv
        ├── sentiment/
        │   ├── vix_dxy_daily.csv
        │   ├── gold_specific_factors.csv
        │   └── news_volume_daily.csv
        └── events/
            └── geopolitical_events.csv
```

## Priority Order

If time is limited, collect in this order:
1. **Daily price data** (Category 1A) — ✅ Done
2. **Economic calendar** (Category 2) — ✅ Done (needs date format fix — Sohan will handle)
3. **Interest rates + bond yields** (Category 3) — ✅ Done
4. **VIX + DXY + S&P 500** (Category 4) — ✅ Done
5. **Gold-specific factors** (Category 4) — ✅ Done
6. **Geopolitical events** (Category 5) — ✅ Done
7. **News volume** (Category 4) — ✅ Done
8. **Hourly price data** (Category 1B) — 🔴 NEW, needed for session features. Use OANDA or Dukascopy.

## Questions?
Ping Sohan on Slack/Discord before making assumptions about any data format.
