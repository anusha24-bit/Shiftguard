# ShiftGuard

**A Human-in-the-Loop Framework for Distribution Shift Detection, Attribution, and Adaptive Retraining in Non-Stationary Forex Markets**

CS 6140 — Machine Learning | Prof. Smruthi Mukund | Northeastern University, Silicon Valley

**Team**: Sohan Mahesh · Anusha Ravi Kumar · Dishaben Manubhai Patel

---

## Overview

ShiftGuard is an ML pipeline that detects, explains, and adapts to distribution shifts in forex markets. It doesn't predict markets — it **monitors a model's understanding of short-term momentum and detects when that understanding becomes invalid due to distribution shifts**. The 4-hour momentum estimate is the vehicle for measuring model health, not a trading signal.

Core thesis: **the pipeline around the model (detection → attribution → selective retraining) matters more than model complexity.**

---

## Prediction Unit: 4-Hour Bars

The base dataframe uses **4-hour OHLCV bars** (6 per trading day, ~18,000 rows per pair). Each bar naturally aligns with a market session:

```
00:00-04:00  Late Sydney / Early Asian
04:00-08:00  Asian / Early London
08:00-12:00  London session
12:00-16:00  London-NY overlap (peak liquidity)
16:00-20:00  NY session
20:00-00:00  Late NY / Early Sydney
```

Target: `next-4H log return = ln(close_t+1 / close_t)` — estimated momentum for the next 4-hour window.

---

## The 4 Feature Groups (~50-65 features per pair)

All models use the same features. SHAP attribution traces shifts back to which group caused them.

### Group 1: Technical (from 4H OHLCV)
SMA(20/50), EMA(12/26), MACD line/signal/histogram, RSI(14), ADX(14), Stochastic %K/%D, Williams %R, CCI(20), ROC(10), Bollinger %B/Width, Ichimoku Tenkan/Kijun, 4H return, log return, gap, close-open range, session label (categorical)

### Group 2: Volatility (regime-level risk)
ATR(14), Garman-Klass vol, Parkinson vol, rolling std (5/20/60 bars), vol-of-vol, vol ratio (short/long), drawdown(20), high-low range

### Group 3: Macro (economic fundamentals)
Interest rate differential, yield spread (10Y), rate diff delta (30d), CPI/NFP/GDP surprise (actual - forecast), event proximity (bars to next event), binary event flags (is_rate_decision_day, is_nfp_day, is_cpi_day)

### Group 4: Sentiment (market risk & cross-asset)
VIX + VIX change, DXY + DXY change, S&P 500 return, oil return, rolling correlations (with S&P, with DXY), news volume/spike, gold-specific (real yield, GLD holdings, COT, M2 — XAU/USD only)

---

## Model Lineup

| Role | Model | Purpose |
|---|---|---|
| Baseline 1 | LSTM + Attention (all 4 groups) | Strong sequence baseline — proves temporal modeling alone isn't enough for adaptive retraining |
| Baseline 2 | Random Forest (all 4 groups) | Strong bagging baseline — proves bagging can't keep up with boosting under retraining |
| Baseline 3 | Stacked Ensemble: LSTM+XGBoost→XGBoost meta (all 4 groups) | Complexity baseline — proves stacking adds overhead without improving retraining speed |
| Baseline 4 | TFT / Temporal Fusion Transformer (all 4 groups) | Deep learning complexity baseline — proves transformers are too data-hungry for small post-shift windows |
| **Main Model** | **XGBoost (all 4 groups)** | **Best retraining efficiency, native TreeSHAP, fast on small windows** |

All 5 models get the **same features**. The only variable is the algorithm.

---

## Pipeline (7 Phases)

### Phase 1: Data Ingestion
- **4H OHLCV Bars** — EUR/USD, GBP/JPY, XAU/USD (resampled from hourly Dukascopy data, ~18,000 bars per pair)
- **Economic Calendar** — USD/EUR/GBP/JPY events with actual, forecast, previous (Investing.com)
- **Macro Time Series** — Interest rates, bond yields, CPI, unemployment, GDP (FRED API)
- **Sentiment Data** — VIX, DXY, S&P 500, Oil (yfinance). Gold: GLD holdings, COT, M2 (FRED)
- **Geopolitical Log** — 17+ known black swan events (manual curation)

### Phase 2: Feature Engineering
- Compute all 4 feature groups from 4H bars
- Merge macro/sentiment data onto 4H bars by date (forward-filled daily data applies to all 6 bars in that day)
- Target: next-4H log return (`target_return`) + direction (`target_direction`)
- Session label column included as a categorical feature
- Split: Train (2015-2019) | Val (2020) | Test (2021-2025)
- Chronological only — no shuffling

### Phase 3: Model Training & Comparison
- Train all 5 models on identical features and splits
- Hyperparameter tuning via TimeSeriesSplit(5) on training set
- XGBoost tuning: learning_rate, max_depth, n_estimators, reg_alpha, reg_lambda
- LSTM: hidden=128, layers=2, dropout=0.3, lookback=30 (4H bars = 5 days), attention head
- RF: GridSearchCV over n_estimators, max_depth, min_samples_leaf
- Evaluate all with: MAE, RMSE, Directional Accuracy, F1, AUC-ROC

### Phase 4: Dual-Mode Shift Detection Engine

**Scheduled Path**
- Cross-reference economic calendar
- On event day: KS test + MMD on pre-event window (10d) vs post-event window (10d)
- Significant → SCHEDULED SHIFT (with severity score)

**Unexpected Path**
- On non-event days: ADWIN (primary) sliding window on feature distributions
- BOCPD (secondary) — probability of shift
- Triggered without calendar match → UNEXPECTED SHIFT (with severity score)

**Performance Monitor**
- DDM / EDDM on XGBoost's error stream (residuals)
- If model error spikes but no shift detected → detection gap (logged)

### Phase 5: SHAP Attribution Layer
- TreeSHAP on XGBoost using post-shift data against pre-shift trained model
- Aggregate SHAP values by feature group → identifies which group drove the shift
- Per-feature KS drift magnitude as secondary attribution signal
- Example output: "Volatility 55% | Macro 28% | Technical 12% | Sentiment 5%"

### Phase 6: Human-in-the-Loop Dashboard (Streamlit)
- Shift alert panel: type, severity, SHAP waterfall plot, model confidence before/after
- User actions: CONFIRM | REJECT | OVERRIDE LABEL (reclassify scheduled ↔ unexpected)
- Confirmed shifts → retraining queue
- Rejected shifts → logged as false positive for analysis

### Phase 7: Selective Retraining
- Compare 3 strategies on confirmed shifts:
  - A) Full retrain (all historical data)
  - B) Window retrain (last 30/60 days only)
  - C) Weighted retrain (upweight recent samples)
- Measure recovery: rolling MAE → days to return to pre-shift baseline
- Model updated → loop back to Phase 4 (detection resumes)

---

## Experiments

### Experiment 1: Model Comparison
LSTM vs RF vs Stacked Ensemble vs TFT vs XGBoost — same features, different algorithms.
Table + bar chart. Proves complexity doesn't improve retraining recovery.

### Experiment 2: Ablation Study
No detection (blind retraining) → Detection only → +Attribution → +Human feedback.
Rolling MAE over time with vertical shift event lines. Progressive improvement.

### Experiment 3: Hyperparameter Tuning
XGBoost learning_rate × max_depth × n_estimators sweep.
Heatmap + top-5 param combos table.

### Experiment 4: Retraining Strategy
Full retrain vs Window(30d) vs Window(60d) vs Weighted — recovery curves per shift event.

### Experiment 5: Detection Evaluation
Dual-mode (scheduled + unexpected) vs single-mode (ADWIN only).
Precision / Recall / F1 against ground truth (geopolitical events log).

---

## Evaluation Metrics

**Regression**: MAE (primary), RMSE, Directional Accuracy
**Classification**: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
**Adaptation**: Rolling MAE (30-day window), Recovery time (days to baseline), Max error spike, Retraining cycles count

---

## Data Split

```
|<-------- Train ---------->|<-- Val -->|<------ Test ------->|
  Jan 2015 — Dec 2019         2020        2021 — Dec 2025
  (~7,800 4H bars)          (~1,500)     (~7,800 4H bars)
  Normal regimes            COVID stress   Fed hikes, BOJ, SVB
```

No shuffling. Chronological order preserved. TimeSeriesSplit(5) for hyperparameter CV on training set only.

---

## Project Structure

```
ShiftGuard/
├── src/
│   ├── features/
│   │   ├── technical.py          # Group 1 features
│   │   ├── volatility.py         # Group 2 features
│   │   ├── macro.py              # Group 3 features
│   │   ├── sentiment.py          # Group 4 features
│   │   └── build_dataset.py      # Merge all groups + targets
│   ├── models/
│   │   ├── baseline_lstm.py      # Baseline 1: LSTM + Attention
│   │   ├── baseline_rf.py        # Baseline 2: Random Forest
│   │   ├── baseline_stacked.py   # Baseline 3: Stacked Ensemble
│   │   ├── baseline_tft.py       # Baseline 4: TFT
│   │   └── main_xgboost.py       # Main: XGBoost
│   ├── detection/
│   │   ├── scheduled.py          # KS + MMD with calendar
│   │   ├── unexpected.py         # ADWIN + BOCPD
│   │   ├── performance.py        # DDM / EDDM on error stream
│   │   └── engine.py             # Dual-mode orchestrator
│   ├── attribution/
│   │   └── shap_analysis.py      # TreeSHAP + group aggregation
│   ├── retraining/
│   │   └── selective.py          # Full / Window / Weighted strategies
│   ├── dashboard/
│   │   └── app.py                # Streamlit HITL dashboard
│   ├── evaluation/
│   │   └── metrics.py            # Shared evaluation framework
│   └── utils/
│       └── data_loader.py        # CSV loading, splits, preprocessing
├── data/
│   ├── raw/
│   │   ├── price/                # EURUSD/GBPJPY/XAUUSD OHLCV
│   │   ├── calendar/             # Economic calendar per currency
│   │   ├── macro/                # Interest rates, CPI, GDP, employment
│   │   ├── sentiment/            # VIX, DXY, gold factors
│   │   └── events/               # Geopolitical events log
│   └── processed/                # Merged feature matrices per pair
├── results/
│   ├── predictions/              # Per-model prediction CSVs
│   ├── figures/                  # All plots for report
│   └── tables/                   # All result tables
├── notebooks/
│   └── EDA.ipynb                 # Exploratory data analysis
├── requirements.txt
└── README.md                     # This file
```

---

## Key Dependencies

```
pandas, numpy, scikit-learn, xgboost, lightgbm, catboost
torch (LSTM + TFT), pytorch-forecasting (TFT)
shap, ta (technical indicators), fredapi
river (ADWIN), streamlit
matplotlib, seaborn, plotly
optuna (hyperparameter tuning)
```

---

## How the Pipeline Components Connect

```
PREDICTION MODEL ──produces predictions──► DETECTION ENGINE
       ▲                                        │
       │                                        ▼
       └──── retrained using ◄──── SHAP ATTRIBUTION
              targeted data        (explains WHAT changed)
                   ▲                        │
                   │                        ▼
                   └──── confirmed ◄── HUMAN DASHBOARD
                         shifts        (CONFIRM / REJECT)
```

- **Prediction model** (XGBoost): Runs daily, predicts returns. Doesn't know about shifts.
- **Detection engine**: Watches feature distributions (KS/MMD/ADWIN) + model error (DDM). Flags shifts.
- **SHAP attribution**: On detected shift, traces to feature group. "Volatility drove this shift."
- **Human dashboard**: Shows alert + attribution. Human confirms or rejects.
- **Selective retraining**: Retrains XGBoost on confirmed shifts. Cycle restarts.

---

## Reproducibility

- `random_state=42` everywhere
- All model checkpoints saved
- Feature engineering is deterministic (no randomness)
- Standardization fit on train only, applied to val/test
- All results reproducible from `data/raw/` → `python src/features/build_dataset.py` → model training scripts

---

## References

1. Gama, J. et al. (2014). A survey on concept drift adaptation. ACM Computing Surveys.
2. Lu, J. et al. (2019). Learning under concept drift: A review. IEEE TKDE.
3. Ganin, Y. et al. (2016). Domain-adversarial training of neural networks. JMLR.
4. Monarch, R. (2021). Human-in-the-Loop Machine Learning. Manning.
5. Amershi, S. et al. (2014). Power to the people. AI Magazine.
6. Lundberg, S. & Lee, S. (2017). A unified approach to interpreting model predictions. NeurIPS.
