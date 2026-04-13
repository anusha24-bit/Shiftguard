# ShiftGuard

**Event-Aware Distribution Shift Detection, Attribution, and Human-in-the-Loop Adaptation for Non-Stationary Forex Markets**

CS 6140 - Machine Learning  
Northeastern University, Silicon Valley

Team: Sohan Mahesh, Anusha Ravi Kumar, Dishaben Manubhai Patel

## Overview

ShiftGuard is an ML systems project for handling **distribution shift in forex markets**. The central idea is that forex behavior changes around scheduled macro events, unexpected geopolitical shocks, volatility dislocations, and changing market structure. Instead of treating forecasting as the whole problem, ShiftGuard focuses on the pipeline around a monitored model:

1. train a monitored return model,
2. detect scheduled and unexpected shifts,
3. explain shifts with SHAP-based attribution,
4. auto-confirm alerts by default while keeping a human override layer,
5. selectively retrain after approved shifts,
6. adapt trading posture based on the detected shift context.

The project is therefore about **detecting, explaining, and adapting to non-stationarity**, not about building a pure trading bot.

## Canonical Project Goal

ShiftGuard asks:

- Can we detect distribution shifts in forex markets using event-aware and statistical methods?
- Can we explain those shifts using feature-group attribution?
- Can human review improve the quality of adaptation decisions while staying lightweight?
- Does selective retraining recover model performance better than no retraining or blind retraining?
- Can shift type and dominant attribution group drive a better downstream response than a single static policy?

## Canonical Pipeline

```text
Raw data (price + macro + sentiment + events)
    ->
Feature engineering
    technical / volatility / macro / sentiment / regime
    ->
Monitored model
    XGBoost regressor on next-bar return
    ->
Shift detection
    scheduled = calendar + KS/MMD
    unexpected = anomaly detection
    performance = drift on prediction errors
    ->
SHAP attribution
    what feature group drove the shift?
    ->
Dashboard
    auto-confirm by default
    human override / reject / relabel when needed
    ->
Adaptive response
    trading posture selection
    + retraining policy selection
    ->
Selective retraining
    full / window / weighted / adaptive
```

## Why XGBoost Regressor Is Canonical

The submission-facing monitored model is `XGBRegressor`, not `XGBClassifier`.

This choice is deliberate:

- the project needs a continuous error stream for drift monitoring,
- rolling MAE and recovery analysis are central evaluation tools,
- SHAP attribution is straightforward and fast,
- retraining experiments are easier to interpret with a return target.

Classification and trading-oriented experiments may still exist locally as supporting work, but they are not the canonical story of the project.

## Current System Behavior

The current local version of ShiftGuard now supports:

- a connected end-to-end pipeline from model outputs to detection, attribution, dashboard review, and retraining
- a Streamlit dashboard with **auto-confirm by default** and **human override only when necessary**
- selective retraining that consumes approved dashboard decisions
- an initial **adaptive ShiftGuard trading policy** for `EURUSD`
  - `technical` shift -> technical followthrough posture
  - `scheduled + macro/sentiment` shift -> event-aligned posture
  - `unexpected / volatility` shift -> defensive shock posture
- an initial **adaptive retraining experiment** where retrain policy can depend on shift type and dominant attribution group

This means attribution is no longer only descriptive; it is beginning to control downstream behavior.

## Data

The repo currently uses:

- 3 instruments: `EURUSD`, `GBPJPY`, `XAUUSD`
- multi-year forex OHLCV data
- macroeconomic calendar data
- macro time series
- sentiment / cross-asset context
- manually curated event context

Processed datasets are stored in `data/processed/` and raw inputs live in `data/raw/`.

## Feature Groups

The canonical submission path uses feature groups intended to capture multiple kinds of market change:

- `technical`
- `volatility`
- `macro`
- `sentiment`

Some legacy experiments also introduced regime-oriented or session-style features, but those are not required to understand the main ShiftGuard submission flow.

## Baseline Setup

The current project-facing comparison structure is:

- **Strong technical baseline**
  - rule-based trading benchmark
  - implemented in [src/models/baseline_technical.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/baseline_technical.py)
  - intended to be stronger than a minimal RSI/MACD toy rule and act as the main hand-crafted benchmark

- **ML direction baseline**
  - standard supervised benchmark
  - implemented in [src/models/baseline_ml_direction.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/baseline_ml_direction.py)
  - used to show what a plain predictive learner can do without explicit shift-aware control

- **Advanced supplementary baseline**
  - stacked ensemble in [src/models/baseline_stacked.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/baseline_stacked.py)
  - current structure: `RandomForest + BiLSTM-Attention -> LightGBM meta-learner`
  - included to test whether additional model complexity alone solves non-stationarity

- **Main system**
  - ShiftGuard
  - shift detection + attribution + dashboard review + adaptive response

## Submission-Facing Files

Canonical local pipeline:

- [src/models/main_xgboost.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/main_xgboost.py)
- [src/detection/engine.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/detection/engine.py)
- [src/attribution/shap_analysis.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/attribution/shap_analysis.py)
- [src/dashboard/app.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/dashboard/app.py)
- [src/retraining/selective.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/retraining/selective.py)
- [src/run_pipeline.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/run_pipeline.py)
- [src/models/baseline_technical.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/baseline_technical.py)
- [src/models/baseline_ml_direction.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/baseline_ml_direction.py)
- [src/models/winrate_experiment.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/winrate_experiment.py)
- [src/models/baseline_stacked.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/models/baseline_stacked.py)

Core feature/data pipeline:

- [src/features/build_dataset.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/features/build_dataset.py)
- [src/features/technical.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/features/technical.py)
- [src/features/volatility.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/features/volatility.py)
- [src/features/macro.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/features/macro.py)
- [src/features/sentiment.py](/C:/Users/Sohan%20M/Desktop/Shiftguard/src/features/sentiment.py)

## Legacy / Pivoted Work

Several files in `src/models/` reflect experiments from later pivots, including regime classification, live signal generation, and trading-oriented analysis. These are useful as local references, but they are **not** the canonical submission path.

In particular, the following should be treated as legacy or supporting work unless explicitly needed:

- `src/models/regime_classifier.py`
- `src/models/live_predict.py`
- `src/models/hpt_regime.py`
- `src/models/transition_predictor.py`
- profit-first analysis scripts

Likewise, `results/archive/` contains superseded experimental work and should not be treated as the source of truth for the final project narrative.

## How To Run Locally

### Full canonical flow

```bash
python src/run_pipeline.py
```

This runs:

1. `src/models/main_xgboost.py`
2. `src/detection/engine.py`
3. `src/attribution/shap_analysis.py`
4. `src/retraining/selective.py` if approved decisions already exist

If no review decisions are present yet, the pipeline pauses and expects:

```bash
streamlit run src/dashboard/app.py
```

After reviewing or overriding shifts in the dashboard, rerun:

```bash
python src/retraining/selective.py
```

### Step-by-step local flow

```bash
python src/features/build_dataset.py
python src/models/main_xgboost.py
python src/detection/engine.py
python src/attribution/shap_analysis.py
streamlit run src/dashboard/app.py
python src/retraining/selective.py
```

For a quicker local validation run on one pair:

```bash
python src/models/main_xgboost.py --pairs EURUSD --fast
```

## Adaptive Logic

ShiftGuard is moving toward an **adaptive policy system**, not just a detector.

Current intended adaptation logic:

- `technical`-dominant shift
  - loosen participation only when technical structure agrees
  - prefer technical-followthrough trading posture
  - favor shorter, more local retraining windows

- `scheduled + macro/sentiment` shift
  - require event-aware alignment and stronger confirmation
  - favor broader retraining on a fuller historical slice

- `unexpected / volatility` shock
  - become more defensive in trading
  - favor weighted retraining with recent data emphasized

The goal is for **shift type + dominant cause** to determine both:

- how ShiftGuard trades
- how the monitored model retrains

## Current Results Snapshot

### Shift detection / attribution

Current cross-pair shift outputs generated locally:

- `EURUSD`: `166` scheduled, `32` unexpected, `198` total
- `GBPJPY`: `118` scheduled, `118` unexpected, `236` total
- `XAUUSD`: `154` scheduled, `112` unexpected, `266` total

Sample attribution pattern:

- `EURUSD`: mostly technical / sentiment
- `GBPJPY`: mixed technical / sentiment
- `XAUUSD`: technical-heavy with meaningful sentiment contribution

### Selective retraining summary

Current saved retraining summaries:

- `EURUSD`
  - No retrain: `MAE 0.001220`, recovery `45.7` bars
  - Full retrain: `MAE 0.001207`, recovery `42.7` bars

- `GBPJPY`
  - No retrain: `MAE 0.001852`, recovery `60.7` bars
  - Full retrain: `MAE 0.001789`, recovery `57.0` bars

- `XAUUSD`
  - No retrain: `MAE 0.002997`, recovery `47.6` bars
  - Full retrain: `MAE 0.002947`, recovery `47.1` bars

Across all three pairs, **selective retraining outperforms no retraining**, and full retraining is currently the most reliable default baseline.

## Evaluation Story

The strongest evaluation framing for this project is:

- **Detection quality**
  scheduled vs unexpected shift coverage, alert counts, event alignment

- **Attribution quality**
  dominant feature groups and interpretable shift explanations

- **Adaptation quality**
  rolling MAE, recovery time, and retraining efficiency

- **Human-in-the-loop value**
  auto-confirm keeps the workflow efficient, while overrides and rejections improve downstream adaptation quality

- **Adaptive policy quality**
  whether shift type and dominant attribution group improve trading posture and retraining choice

The main academic claim is not that ShiftGuard maximizes raw trading profit. The claim is that it provides a more structured and interpretable way to handle shift in a non-stationary financial environment.

## Current Repo Notes

- The canonical monitored model writes outputs to `results/predictions/`.
- The dashboard reads from `results/detection/`, `results/attribution/`, `results/predictions/`, and writes decisions to `results/decisions/`.
- The dashboard currently auto-confirms new shifts by default and queues them for retraining unless a human overrides or rejects them.
- Selective retraining now consumes approved dashboard decisions when available.
- `src/models/winrate_experiment.py` now contains the first adaptive trading-policy implementation, currently explored on `EURUSD`.
- `results/archive/` contains historical experiments and should be treated as non-canonical.

## References

1. Gama, J. et al. (2014). A survey on concept drift adaptation.
2. Lu, J. et al. (2019). Learning under concept drift: A review.
3. Ganin, Y. et al. (2016). Domain-adversarial training of neural networks.
4. Monarch, R. (2021). Human-in-the-Loop Machine Learning.
5. Amershi, S. et al. (2014). Power to the people.
6. Lundberg, S. and Lee, S. (2017). A unified approach to interpreting model predictions.
