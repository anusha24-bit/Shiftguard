"""
Baseline 3: Stacked Ensemble — RF + BiLSTM-Attention → LightGBM Meta
Component 1: RandomForest (flat features, all 4 groups)
Component 2: BiLSTM + Attention (sequence features, all 4 groups)
Meta-learner: LightGBM combines RF + BiLSTM predictions + raw features.

Usage:
    python3 src/models/baseline_stacked.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import lightgbm as lgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'baseline3')
os.makedirs(RESULTS_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction']

# BiLSTM component config (lighter — it's a component, not the whole model)
LOOKBACK = 30
BILSTM_HIDDEN = 64
BILSTM_LAYERS = 1
BILSTM_DROPOUT = 0.2
BILSTM_EPOCHS = 50
BILSTM_PATIENCE = 10
BILSTM_LR = 0.001
BILSTM_BATCH = 64
DEVICE = 'cpu'

# RandomForest base config
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 12,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

# LightGBM meta-learner config
META_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
}


# ============================================================
# BiLSTM + Attention component (lightweight)
# ============================================================
class ComponentAttention(nn.Module):
    """Additive attention for BiLSTM component."""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        energy = torch.tanh(self.W(lstm_output))
        scores = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_output * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class BiLSTMComponent(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.attention = ComponentAttention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context, _ = self.attention(out)
        return self.fc(context)


def train_bilstm(X_train, y_train, X_val, y_val, input_size):
    model = BiLSTMComponent(input_size, BILSTM_HIDDEN, BILSTM_LAYERS, BILSTM_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=BILSTM_LR)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    best_state = model.state_dict().copy()
    patience_counter = 0

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BILSTM_BATCH, shuffle=False
    )

    for epoch in range(BILSTM_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(torch.FloatTensor(X_val)).squeeze(),
                                 torch.FloatTensor(y_val)).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= BILSTM_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def get_bilstm_predictions(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).squeeze().numpy()


# ============================================================
# Helpers
# ============================================================
def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def split_data(df):
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    train = df[df['datetime_utc'] < '2020-01-01'].copy()
    val = df[(df['datetime_utc'] >= '2020-01-01') & (df['datetime_utc'] < '2021-01-01')].copy()
    test = df[df['datetime_utc'] >= '2021-01-01'].copy()
    return train, val, test


def create_sequences(data, targets, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    acc = accuracy_score(y_true_dir, y_pred_dir)
    p, r, f1, _ = precision_recall_fscore_support(y_true_dir, y_pred_dir, average='binary', zero_division=0)

    print(f"\n  {label} Results:")
    print(f"    MAE:      {mae:.6f}")
    print(f"    RMSE:     {rmse:.6f}")
    print(f"    Dir Acc:  {dir_acc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1:       {f1:.4f}")
    print(f"    Prec:     {p:.4f}  Recall: {r:.4f}")

    return {'mae': mae, 'rmse': rmse, 'dir_acc': dir_acc, 'accuracy': acc, 'f1': f1, 'precision': p, 'recall': r}


# ============================================================
# Main
# ============================================================
def run_pair(pair_name):
    print(f"\n{'='*60}")
    print(f"BASELINE 3: Stacked (RF + BiLSTM → LightGBM) — {pair_name}")
    print(f"{'='*60}")

    # Load
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}, Rows: {len(df)}")

    # Split
    train_df, val_df, test_df = split_data(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Prepare flat features (for RF + meta)
    train_median = train_df[feature_cols].median()
    X_train_flat = train_df[feature_cols].fillna(train_median).replace([np.inf, -np.inf], 0).values
    X_val_flat = val_df[feature_cols].fillna(train_median).replace([np.inf, -np.inf], 0).values
    X_test_flat = test_df[feature_cols].fillna(train_median).replace([np.inf, -np.inf], 0).values

    y_train = train_df['target_return'].fillna(0).values
    y_val = val_df['target_return'].fillna(0).values
    y_test = test_df['target_return'].fillna(0).values

    # --- Component 1: RandomForest ---
    print("  Training RandomForest component...")
    rf_model = RandomForestRegressor(**RF_PARAMS)
    rf_model.fit(X_train_flat, y_train)

    rf_train_pred = rf_model.predict(X_train_flat)
    rf_val_pred = rf_model.predict(X_val_flat)
    rf_test_pred = rf_model.predict(X_test_flat)
    print(f"    RF done.")

    # --- Component 2: BiLSTM + Attention ---
    print("  Training BiLSTM + Attention component...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train_flat)
    val_scaled = scaler.transform(X_val_flat)
    test_scaled = scaler.transform(X_test_flat)

    X_train_seq, y_train_seq = create_sequences(train_scaled, y_train, LOOKBACK)
    X_val_seq, y_val_seq = create_sequences(val_scaled, y_val, LOOKBACK)
    X_test_seq, y_test_seq = create_sequences(test_scaled, y_test, LOOKBACK)

    bilstm_model = train_bilstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, len(feature_cols))

    bilstm_train_pred = get_bilstm_predictions(bilstm_model, X_train_seq)
    bilstm_val_pred = get_bilstm_predictions(bilstm_model, X_val_seq)
    bilstm_test_pred = get_bilstm_predictions(bilstm_model, X_test_seq)
    print(f"    BiLSTM done. Train pred shape: {bilstm_train_pred.shape}")

    # --- Meta-learner: RF + BiLSTM predictions + raw features → LightGBM ---
    # Align sizes (BiLSTM loses LOOKBACK rows)
    print("  Training LightGBM meta-learner...")
    train_offset = LOOKBACK

    meta_train_X = np.column_stack([
        rf_train_pred[train_offset:],
        bilstm_train_pred,
        X_train_flat[train_offset:],
    ])
    meta_train_y = y_train[train_offset:]

    meta_val_X = np.column_stack([
        rf_val_pred[LOOKBACK:],
        bilstm_val_pred,
        X_val_flat[LOOKBACK:],
    ])
    meta_val_y = y_val[LOOKBACK:]

    meta_test_X = np.column_stack([
        rf_test_pred[LOOKBACK:],
        bilstm_test_pred,
        X_test_flat[LOOKBACK:],
    ])
    meta_test_y = y_test[LOOKBACK:]

    meta_model = lgb.LGBMRegressor(**META_PARAMS)
    meta_model.fit(
        meta_train_X, meta_train_y,
        eval_set=[(meta_val_X, meta_val_y)],
    )

    # Predict
    val_pred = meta_model.predict(meta_val_X)
    test_pred = meta_model.predict(meta_test_X)

    # Evaluate
    val_results = evaluate(meta_val_y, val_pred, "Validation")
    test_results = evaluate(meta_test_y, test_pred, "Test")

    # Save predictions
    test_dates = test_df['datetime_utc'].iloc[LOOKBACK:].reset_index(drop=True)
    pred_df = pd.DataFrame({
        'datetime_utc': test_dates.values[:len(meta_test_y)],
        'actual': meta_test_y,
        'predicted': test_pred,
    })
    pred_path = os.path.join(RESULTS_DIR, f'baseline3_stacked_{pair_name}.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved: {pred_path}")

    return test_results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE 3: Stacked (RF + BiLSTM → LightGBM) — Summary")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'MAE':<10} {'RMSE':<10} {'Dir Acc':<10} {'F1':<10}")
    print("-" * 50)
    for pair, r in all_results.items():
        print(f"{pair:<10} {r['mae']:<10.6f} {r['rmse']:<10.6f} {r['dir_acc']:<10.4f} {r['f1']:<10.4f}")

    summary_path = os.path.join(RESULTS_DIR, 'baseline3_stacked_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
