# ShiftGuard — Baseline Models & Feature Engineering Instructions

**Assigned to**: [Teammate Name]
**Deadline**: [FILL IN]
**Project**: ShiftGuard (CS 6140, Prof. Smruthi Mukund)
**Depends on**: Data collection (see DATA_COLLECTION_INSTRUCTIONS.md)

---

## Your Scope

You are responsible for:
1. Building the feature engineering pipeline (all 5 feature groups)
2. Implementing Baseline 1 (LSTM + Attention, all 5 feature groups)
3. Producing benchmark results that the main models must beat

**Not your scope** (Sohan will handle): Baseline 2 (Random Forest), Baselines 3-4 (Stacked Ensemble, TFT), main model (XGBoost), detection engine, SHAP, dashboard.

---

## PART 1: Feature Engineering Pipeline

### Input Data (from data collection)
You'll receive CSVs in `ShiftGuard/data/raw/`. Your job is to transform them into a single merged feature matrix per currency pair.

### Output Format
One CSV per pair in `ShiftGuard/data/processed/`:
```
EURUSD_features.csv
GBPJPY_features.csv
XAUUSD_features.csv
```

Each row = one trading day. Columns = date + all features below + target variable.

### Target Variable
```
target = next-day log return = ln(close_t+1 / close_t)
```
This is a regression target. We may also create a classification target later:
```
target_dir = 1 if next-day return > 0, else 0
```
Compute both. Put them as the last two columns: `target_return`, `target_direction`.

---

### Feature Group 1: Technical Indicators
**Source**: OHLCV data only
**Library**: `ta` (pip install ta) or `pandas_ta`

Compute ALL of the following. Use default periods unless specified.

#### Trend
- SMA(20) — 20-day Simple Moving Average
- SMA(50)
- EMA(12) — 12-day Exponential Moving Average
- EMA(26)
- MACD line = EMA(12) - EMA(26)
- MACD signal = EMA(9) of MACD line
- MACD histogram = MACD - Signal
- ADX(14) — Average Directional Index (trend strength, 0-100)
- Plus DI(14) / Minus DI(14) — Directional indicators
- Ichimoku Conversion Line (Tenkan-sen, period 9)
- Ichimoku Base Line (Kijun-sen, period 26)

#### Momentum
- RSI(14) — Relative Strength Index
- Stochastic %K(14), %D(3)
- Williams %R(14)
- CCI(20) — Commodity Channel Index
- ROC(10) — Rate of Change (10-day)
- MFI(14) — Money Flow Index (needs volume, skip if volume unavailable)

#### Volatility (price-derived)
- Bollinger Band Upper(20, 2std)
- Bollinger Band Lower(20, 2std)
- Bollinger Band Width = (Upper - Lower) / SMA(20)
- Bollinger %B = (Close - Lower) / (Upper - Lower)
- ATR(14) — Average True Range
- Keltner Channel Upper / Lower

#### Volume (if available, especially for XAU/USD)
- OBV — On Balance Volume
- VWAP — Volume Weighted Average Price (if intraday data available)
- Volume SMA(20) — to detect volume spikes

#### Derived Price Features
- Daily return = (close_t - close_t-1) / close_t-1
- Log return = ln(close_t / close_t-1)
- High-Low range = (high - low) / close
- Close-Open range = (close - open) / open
- Gap = (open_t - close_t-1) / close_t-1

#### Code Template
```python
import pandas as pd
import ta

def compute_technical_features(df):
    """
    Input: df with columns [date, open, high, low, close, volume]
    Output: df with all technical indicator columns appended
    """
    # Trend
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

    macd = ta.trend.MACD(df['close'])
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    # Momentum
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    df['cci_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    df['roc_10'] = ta.momentum.roc(df['close'], window=10)

    # Volatility
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    df['bb_pctb'] = bb.bollinger_pband()

    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

    # Derived
    df['daily_return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['co_range'] = (df['close'] - df['open']) / df['open']
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    return df
```

---

### Feature Group 2: Volatility Features (Advanced)
**Source**: OHLCV + computed

These go beyond ATR/Bollinger. They measure regime-level volatility changes.

```python
def compute_volatility_features(df):
    """Advanced volatility estimators using OHLC data."""

    # Garman-Klass volatility (uses OHLC, more efficient than close-to-close)
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    df['gk_vol'] = np.sqrt(
        (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(20).mean()
    )

    # Parkinson volatility (High-Low based)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low'])**2).rolling(20).mean()
    )

    # Rolling standard deviation of returns (multiple windows)
    df['rolling_std_5'] = df['log_return'].rolling(5).std()
    df['rolling_std_20'] = df['log_return'].rolling(20).std()
    df['rolling_std_60'] = df['log_return'].rolling(60).std()

    # Volatility of volatility (second-order)
    df['vol_of_vol'] = df['atr_14'].rolling(20).std()

    # Volatility ratio (short-term / long-term) — spike = regime change
    df['vol_ratio'] = df['rolling_std_5'] / df['rolling_std_60']

    # Max drawdown (rolling 20-day)
    rolling_max = df['close'].rolling(20).max()
    df['drawdown_20'] = (df['close'] - rolling_max) / rolling_max

    return df
```

---

### Feature Group 3: Macro Features
**Source**: `macro/` and `calendar/` CSVs from data collection

This requires merging external data onto the daily price dataframe.

```python
def compute_macro_features(df, rates_df, calendar_df):
    """
    rates_df: daily interest rates / bond yields
    calendar_df: economic calendar events with actual, forecast, previous
    """

    # --- Interest rate differentials (merge rates onto df by date) ---
    # For EUR/USD:
    df['rate_diff'] = rates_df['fed_rate'] - rates_df['ecb_rate']
    df['yield_spread_10y'] = rates_df['us_10y'] - rates_df['de_10y']
    # For GBP/JPY:
    # df['rate_diff'] = rates_df['boe_rate'] - rates_df['boj_rate']
    # df['yield_spread_10y'] = rates_df['uk_10y'] - rates_df['jp_10y']

    # Rate of change in differential
    df['rate_diff_delta_30d'] = df['rate_diff'] - df['rate_diff'].shift(30)

    # --- Event surprise features ---
    # For each high-impact event, compute surprise = actual - forecast
    # Then create columns: nfp_surprise, cpi_surprise, gdp_surprise, etc.
    # These will be 0 on non-event days and the surprise value on event days

    # --- Event proximity ---
    # Days until next high-impact event (from calendar)
    # This captures "pre-event tension"
    df['days_to_next_event'] = compute_days_to_next_event(df['date'], calendar_df)

    # --- Binary event flags ---
    df['is_rate_decision_day'] = df['date'].isin(rate_decision_dates).astype(int)
    df['is_nfp_day'] = df['date'].isin(nfp_dates).astype(int)
    df['is_cpi_day'] = df['date'].isin(cpi_dates).astype(int)

    return df
```

**Important**: Macro data is monthly/quarterly. Forward-fill to daily frequency. Never look ahead — use only data available on or before that date.

---

### Feature Group 4: Sentiment & Cross-Asset Features
**Source**: `sentiment/` CSVs

```python
def compute_sentiment_features(df, sentiment_df, gold_df=None):
    """
    sentiment_df: daily VIX, DXY, S&P 500, Oil
    gold_df: gold-specific factors (GLD holdings, COT, real yield) — only for XAU/USD
    """

    # --- Risk gauges ---
    df['vix'] = sentiment_df['vix_close']
    df['vix_change'] = sentiment_df['vix_close'].pct_change()
    df['vix_sma_10'] = sentiment_df['vix_close'].rolling(10).mean()
    df['vix_above_avg'] = (df['vix'] > df['vix_sma_10']).astype(int)

    df['dxy'] = sentiment_df['dxy_close']
    df['dxy_change'] = sentiment_df['dxy_close'].pct_change()

    df['sp500_return'] = sentiment_df['sp500_close'].pct_change()
    df['oil_return'] = sentiment_df['oil_close'].pct_change()

    # --- Cross-asset correlation (rolling 20-day) ---
    df['corr_with_sp500'] = df['log_return'].rolling(20).corr(df['sp500_return'])
    df['corr_with_dxy'] = df['log_return'].rolling(20).corr(df['dxy_change'])

    # --- Gold-specific (XAU/USD only) ---
    if gold_df is not None:
        df['real_yield'] = gold_df['us_10y'] - gold_df['us_cpi_yoy']
        df['gld_holdings_change'] = gold_df['gld_holdings'].pct_change()
        df['cot_net_long'] = gold_df['cot_net_long']
        df['m2_yoy_change'] = gold_df['m2_money_supply'].pct_change(12)  # 12-month

    # --- News volume (if available) ---
    if 'article_count' in sentiment_df.columns:
        df['news_volume'] = sentiment_df['article_count']
        df['news_spike'] = (df['news_volume'] > df['news_volume'].rolling(30).mean() + 
                            2 * df['news_volume'].rolling(30).std()).astype(int)

    return df
```

---

### Feature Group 5: Session Structure
**Source**: Hourly OHLCV from `data/raw/price/hourly/`
**Output**: Aggregated to daily — one row per trading day, same as all other groups

Forex trades 24hrs across 4 sessions:
```
SYDNEY    21:00 – 06:00 UTC   Thinnest liquidity, sets opening tone
ASIAN     00:00 – 09:00 UTC   Range-bound, JPY-driven
LONDON    07:00 – 16:00 UTC   Highest volume, breakouts
NEW YORK  12:00 – 21:00 UTC   News-driven, USD events
```

```python
SESSIONS = {
    'sydney': (21, 6),   # wraps midnight
    'asian':  (0, 9),
    'london': (7, 16),
    'ny':     (12, 21),
}

def compute_session_features(daily_df, hourly_df):
    """
    Aggregate hourly OHLCV into per-session features, merge onto daily df.
    hourly_df must have: datetime_utc, open, high, low, close
    """
    hourly_df['datetime_utc'] = pd.to_datetime(hourly_df['datetime_utc'])
    hourly_df['hour'] = hourly_df['datetime_utc'].dt.hour
    hourly_df['date'] = hourly_df['datetime_utc'].dt.date

    for session_name, (start_h, end_h) in SESSIONS.items():
        if start_h > end_h:  # wraps midnight (Sydney)
            mask = (hourly_df['hour'] >= start_h) | (hourly_df['hour'] < end_h)
        else:
            mask = (hourly_df['hour'] >= start_h) & (hourly_df['hour'] < end_h)

        session = hourly_df[mask].groupby('date').agg(
            s_open=('open', 'first'),
            s_high=('high', 'max'),
            s_low=('low', 'min'),
            s_close=('close', 'last')
        )

        # Per-session features
        daily_df[f'{session_name}_return'] = (session['s_close'] - session['s_open']) / session['s_open']
        daily_df[f'{session_name}_range'] = (session['s_high'] - session['s_low']) / session['s_open']

    # Session share: which session drove the day?
    for s in SESSIONS:
        daily_df[f'{s}_share'] = daily_df[f'{s}_range'] / daily_df['hl_range']

    # Cross-session features
    daily_df['session_range_ratio'] = daily_df['london_range'] / daily_df['asian_range'].replace(0, np.nan)
    daily_df['max_move_session'] = daily_df[['sydney_range', 'asian_range', 'london_range', 'ny_range']].idxmax(axis=1)

    # London-NY overlap (12:00-16:00 UTC)
    overlap = hourly_df[(hourly_df['hour'] >= 12) & (hourly_df['hour'] < 16)]
    overlap_agg = overlap.groupby('date').agg(o_high=('high', 'max'), o_low=('low', 'min'), o_open=('open', 'first'))
    daily_df['overlap_range'] = (overlap_agg['o_high'] - overlap_agg['o_low']) / overlap_agg['o_open']

    # Weekend gap (Friday close → Sunday Sydney open)
    daily_df['weekend_gap'] = np.where(
        pd.to_datetime(daily_df['date']).dt.dayofweek == 0,  # Monday
        (daily_df['open'] - daily_df['close'].shift(1)) / daily_df['close'].shift(1),
        0
    )

    return daily_df
```

**If hourly data is only available for recent years**: Session features will be NaN for earlier years. That's fine — XGBoost handles NaN natively. LSTM will need those NaNs filled with 0.

---

### Final Merge & Preprocessing

```python
def build_final_dataset(pair_name):
    """
    Merge all 5 feature groups into one dataframe.
    """
    # 1. Load OHLCV
    df = pd.read_csv(f'data/raw/price/{pair_name}_ohlcv.csv', parse_dates=['date'])

    # 2. Compute features (Groups 1-4)
    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, rates_df, calendar_df)
    df = compute_sentiment_features(df, sentiment_df)

    # 3. Compute session features (Group 5) — from hourly data
    try:
        hourly_df = pd.read_csv(f'data/raw/price/hourly/{pair_name}_hourly.csv')
        df = compute_session_features(df, hourly_df)
    except FileNotFoundError:
        print(f"Warning: No hourly data for {pair_name}, skipping session features")

    # 4. Target variable
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_direction'] = (df['target_return'] > 0).astype(int)

    # 5. Drop warmup rows (first 60 days will have NaNs from rolling windows)
    df = df.iloc[60:].reset_index(drop=True)

    # 6. Handle remaining NaNs
    # For macro features (monthly data), forward-fill BEFORE computing features
    # For session features with partial hourly data, leave as NaN (XGBoost handles it)
    # For any leftover NaN, fill with 0 or column median — document which you chose

    # 7. Save
    df.to_csv(f'data/processed/{pair_name}_features.csv', index=False)

    # 8. Log feature counts
    print(f"{pair_name}: {len(df)} rows, {len(df.columns)} columns")

    return df
```

**Expected feature count per pair**: ~65-80 features + 2 targets

---

## PART 2: Baseline Models

### Train/Test Split Strategy

DO NOT use random split. This is time series.

```
|<-------- Train -------->|<-- Val -->|<--- Test --->|
  Jan 2015 — Dec 2019       2020        2021 — 2025

```

- **Train**: 2015-01-01 to 2019-12-31 (~5 years)
- **Validation**: 2020-01-01 to 2020-12-31 (includes COVID — a deliberate stress test)
- **Test**: 2021-01-01 to 2025-12-31 (includes Fed hikes, BOJ intervention, banking crisis)

DO NOT shuffle. Maintain chronological order.

For hyperparameter tuning, use **TimeSeriesSplit** (sklearn) with 5 folds on the training set only.

---

### Evaluation Metrics

Compute ALL of the following for every model. This is the shared evaluation framework.

#### Regression (target_return)
- **MAE** — Mean Absolute Error (primary metric)
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error (if returns are non-zero)
- **Directional Accuracy** — % of days where predicted return sign matches actual sign

#### Classification (target_direction)
- **Accuracy**
- **Precision / Recall / F1** (for both classes)
- **AUC-ROC**
- **Confusion Matrix**

#### Adaptation-Specific (critical for ShiftGuard)
- **Rolling MAE (30-day window)** — Plot this over time. Spikes = model degradation after shifts.
- **Recovery time** — After a known shift event, how many days until rolling MAE returns to pre-shift level?
- **Max error spike** — Worst rolling MAE value after each shift event

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

def evaluate_model(y_true, y_pred, y_pred_proba=None, dates=None):
    """Standard evaluation. Run this for EVERY model."""
    results = {}

    # Regression
    results['mae'] = mean_absolute_error(y_true, y_pred)
    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    results['directional_acc'] = np.mean(np.sign(y_true) == np.sign(y_pred))

    # Classification (on direction)
    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    results['accuracy'] = accuracy_score(y_true_dir, y_pred_dir)
    p, r, f1, _ = precision_recall_fscore_support(y_true_dir, y_pred_dir, average='binary')
    results['precision'] = p
    results['recall'] = r
    results['f1'] = f1

    if y_pred_proba is not None:
        results['auc_roc'] = roc_auc_score(y_true_dir, y_pred_proba)

    # Rolling MAE (for adaptation analysis)
    if dates is not None:
        rolling_mae = pd.Series(np.abs(y_true - y_pred), index=dates).rolling(30).mean()
        results['rolling_mae'] = rolling_mae

    return results
```

---

### Baseline 1: LSTM + Attention (Upgraded — All 5 Feature Groups)

**Features used**: ALL 5 feature groups (~65-80 features)
**Why all features**: This is a strong baseline. Same features as XGBoost. The only difference is the algorithm. If LSTM+Attention still loses to XGBoost on retraining, it proves the architecture isn't the answer.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Feature selection: ALL features (same as XGBoost) ---
# Use every column in the processed CSV except: date, target_return, target_direction
LSTM_FEATURES = [col for col in df.columns if col not in ['date', 'target_return', 'target_direction']]
LOOKBACK = 30  # 30-day input sequence

# --- Data preparation ---
def create_sequences(df, feature_cols, target_col, lookback=30):
    """Convert flat dataframe to (samples, timesteps, features) for LSTM."""
    X, y = [], []
    data = df[feature_cols].values
    target = df[target_col].values

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target[i])

    return np.array(X), np.array(y)

# --- Normalize features (fit on train only, transform val/test) ---
# IMPORTANT: Fill NaN with 0 BEFORE scaling (LSTM can't handle NaN unlike XGBoost)
from sklearn.preprocessing import StandardScaler
train_filled = train_df[LSTM_FEATURES].fillna(0)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_filled)
val_scaled = scaler.transform(val_df[LSTM_FEATURES].fillna(0))
test_scaled = scaler.transform(test_df[LSTM_FEATURES].fillna(0))

# --- Model (with Attention) ---
class Attention(nn.Module):
    """Simple additive attention over LSTM timesteps."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        scores = self.attn(lstm_output)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)   # (batch, seq_len, 1)
        context = (lstm_output * weights).sum(dim=1)  # (batch, hidden_size)
        return context, weights

class LSTMAttentionBaseline(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)               # (batch, seq_len, hidden)
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden)
        return self.fc(context)

# --- Hyperparameters to report ---
# input_size: ~65-80 (all features)
# hidden_size: 128 (upgraded from 64)
# num_layers: 2
# dropout: 0.3
# lookback: 30 days
# learning_rate: 0.001
# batch_size: 32
# epochs: 100 (early stopping on val MAE, patience=15)
# optimizer: Adam
# loss: MSELoss
# attention: additive attention over timesteps
```

#### Training Notes
- Use **early stopping** on validation MAE with patience=15
- Save best model checkpoint
- Log training loss + validation MAE per epoch (we need the learning curve plot)
- **Save attention weights** for a few sample predictions — useful for the report to show which timesteps the model focused on
- After training, run `evaluate_model()` on val and test sets
- Save predictions to CSV: `results/baseline1_lstm_predictions.csv` with columns: `date, actual, predicted`
- **NaN handling**: Fill all NaN with 0 before scaling. Session features will be 0 for years without hourly data.

---

---

## PART 3: Deliverables Checklist

### Code Files
```
ShiftGuard/
├── src/
│   ├── features/
│   │   ├── technical.py        # compute_technical_features()
│   │   ├── volatility.py       # compute_volatility_features()
│   │   ├── macro.py            # compute_macro_features()
│   │   ├── sentiment.py        # compute_sentiment_features()
│   │   ├── sessions.py         # compute_session_features() — hourly → daily
│   │   └── build_dataset.py    # build_final_dataset() — merges all 5 groups
│   ├── models/
│   │   └── baseline_lstm.py    # LSTM model class + training script
│   ├── evaluation/
│   │   └── metrics.py          # evaluate_model() — shared by all models
│   └── utils/
│       └── data_loader.py      # load CSVs, handle splits
├── data/
│   ├── raw/                    # from data collection teammate
│   └── processed/              # your output feature CSVs
├── results/
│   ├── baseline1_lstm_predictions.csv
│   └── figures/                # plots (see below)
└── notebooks/
    └── EDA.ipynb               # exploratory data analysis (optional but useful)
```

### Figures to Produce
1. **Feature correlation heatmap** — top 30 features, check for multicollinearity
2. **Feature distributions** — histograms of key features, check for normality/skew
3. **LSTM training curve** — loss vs epoch (train + val)
4. **Rolling MAE over time** — for LSTM baseline, with vertical lines at known shift events
5. **Actual vs Predicted scatter** — for LSTM on test set
6. **Confusion matrix** — for directional accuracy on LSTM

### Numbers to Report
Fill in this table (we'll use this in the final report):

```
Model          | MAE    | RMSE   | Dir. Acc | F1    | AUC-ROC | Recovery (avg days)
-------------- | ------ | ------ | -------- | ----- | ------- | ------------------
LSTM (tech)    |        |        |          |       |         |
```

---

## PART 4: Important Rules

1. **No data leakage**: Never use future data. Fit scalers/encoders on train only. Forward-fill macro data (never backfill).
2. **No random splits**: Always chronological. Train < Val < Test in time.
3. **Document everything**: Every preprocessing choice (how you handled NaNs, which scaler, why). We need this for the Methods section of the report.
4. **Reproducibility**: Set `random_state=42` everywhere. Save model checkpoints.
5. **Feature names must be consistent**: Use the exact column names from this doc. The main model and SHAP pipeline will reference them.
6. **Tag features by group**: Maintain a mapping dict so we know which features belong to which group (needed for SHAP group attribution later):

```python
FEATURE_GROUPS = {
    'technical': ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd_line', 
                  'macd_signal', 'macd_hist', 'adx', 'rsi_14', 'stoch_k', 
                  'stoch_d', 'williams_r', 'cci_20', 'roc_10'],
    'volatility': ['bb_width', 'bb_pctb', 'atr_14', 'gk_vol', 'parkinson_vol',
                   'rolling_std_5', 'rolling_std_20', 'rolling_std_60',
                   'vol_of_vol', 'vol_ratio', 'drawdown_20', 'hl_range'],
    'macro': ['rate_diff', 'yield_spread_10y', 'rate_diff_delta_30d',
              'days_to_next_event', 'is_rate_decision_day', 'is_nfp_day',
              'is_cpi_day', 'nfp_surprise', 'cpi_surprise'],
    'sentiment': ['vix', 'vix_change', 'dxy', 'dxy_change', 'sp500_return',
                  'oil_return', 'corr_with_sp500', 'corr_with_dxy',
                  'news_volume', 'news_spike'],
    'session': ['sydney_return', 'asian_return', 'london_return', 'ny_return',
                'sydney_range', 'asian_range', 'london_range', 'ny_range',
                'sydney_share', 'asian_share', 'london_share', 'ny_share',
                'session_range_ratio', 'max_move_session',
                'overlap_range', 'weekend_gap'],
}
```

7. **Run this check before handing off**: Confirm that `data/processed/{pair}_features.csv` loads cleanly, has no NaN in feature columns, and the target column is correctly shifted (no leakage).

## Questions?
Ping Sohan before making assumptions about feature choices or evaluation changes.
