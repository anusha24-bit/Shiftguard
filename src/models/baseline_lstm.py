"""
Baseline 1: LSTM + Attention
All 4 feature groups, 30-bar lookback (= 5 trading days).
Trains on 2015-2019, validates on 2020, tests on 2021-2025.

Usage:
    python src/models/baseline_lstm.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Config ---
LOOKBACK = 30       # 30 4H bars = 5 trading days
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']


# ============================================================
# Model
# ============================================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        scores = self.attn(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_output * weights).sum(dim=1)
        return context, weights


class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        return self.fc(context)


# ============================================================
# Data helpers
# ============================================================
def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def create_sequences(data, targets, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def split_data(df):
    """Chronological split: train 2015-2019, val 2020, test 2021+"""
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    train = df[df['datetime_utc'] < '2020-01-01']
    val = df[(df['datetime_utc'] >= '2020-01-01') & (df['datetime_utc'] < '2021-01-01')]
    test = df[df['datetime_utc'] >= '2021-01-01']
    return train, val, test


# ============================================================
# Evaluation
# ============================================================
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
# Training
# ============================================================
def train_model(model, train_loader, val_X, val_y, epochs, patience, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.to(DEVICE)).squeeze()
            val_loss = criterion(val_pred, val_y.to(DEVICE)).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


# ============================================================
# Main
# ============================================================
def run_pair(pair_name):
    print(f"\n{'='*60}")
    print(f"LSTM + Attention — {pair_name}")
    print(f"{'='*60}")

    # Load
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}, Rows: {len(df)}")

    # Split
    train_df, val_df, test_df = split_data(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Fill NaN with 0 (LSTM can't handle NaN)
    train_features = train_df[feature_cols].fillna(0).values
    val_features = val_df[feature_cols].fillna(0).values
    test_features = test_df[feature_cols].fillna(0).values

    train_targets = train_df['target_return'].fillna(0).values
    val_targets = val_df['target_return'].fillna(0).values
    test_targets = test_df['target_return'].fillna(0).values

    # Scale (fit on train only)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)

    # Create sequences
    print(f"  Creating {LOOKBACK}-bar sequences...")
    X_train, y_train = create_sequences(train_scaled, train_targets, LOOKBACK)
    X_val, y_val = create_sequences(val_scaled, val_targets, LOOKBACK)
    X_test, y_test = create_sequences(test_scaled, test_targets, LOOKBACK)
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # To tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=False  # time series — don't shuffle
    )

    # Model
    input_size = X_train.shape[2]
    model = LSTMAttention(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print(f"  Training (max {EPOCHS} epochs, patience={PATIENCE})...")
    model, train_losses, val_losses = train_model(
        model, train_loader, X_val_t, y_val_t, EPOCHS, PATIENCE, LEARNING_RATE
    )

    # Predict
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t.to(DEVICE)).squeeze().cpu().numpy()
        test_pred = model(X_test_t.to(DEVICE)).squeeze().cpu().numpy()

    # Evaluate
    val_results = evaluate(y_val, val_pred, "Validation")
    test_results = evaluate(y_test, test_pred, "Test")

    # Save predictions
    test_dates = test_df['datetime_utc'].iloc[LOOKBACK:].reset_index(drop=True)
    pred_df = pd.DataFrame({
        'datetime_utc': test_dates.values[:len(y_test)],
        'actual': y_test,
        'predicted': test_pred,
    })
    pred_path = os.path.join(RESULTS_DIR, f'baseline1_lstm_{pair_name}.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved: {pred_path}")

    # Save training curves
    curves_df = pd.DataFrame({'epoch': range(1, len(train_losses)+1), 'train_loss': train_losses, 'val_loss': val_losses})
    curves_path = os.path.join(RESULTS_DIR, f'baseline1_lstm_{pair_name}_curves.csv')
    curves_df.to_csv(curves_path, index=False)

    # Save model
    model_path = os.path.join(RESULTS_DIR, f'baseline1_lstm_{pair_name}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")

    return test_results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE 1: LSTM + Attention — Summary")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'MAE':<10} {'RMSE':<10} {'Dir Acc':<10} {'F1':<10}")
    print("-" * 50)
    for pair, r in all_results.items():
        print(f"{pair:<10} {r['mae']:<10.6f} {r['rmse']:<10.6f} {r['dir_acc']:<10.4f} {r['f1']:<10.4f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'baseline1_lstm_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
