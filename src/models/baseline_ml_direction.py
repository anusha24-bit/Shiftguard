"""
ML direction baseline for ShiftGuard.
"""
import numpy as np
import xgboost as xgb


DIR_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 500,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
    'eval_metric': 'logloss',
}


def select_direction_feature_cols(feature_cols):
    return [c for c in feature_cols if c not in [
        'atr_pct_short', 'atr_pct_long', 'vol_ratio_5_60', 'vol_ratio_5_20',
        'vol_compressed', 'compression_duration', 'range_contraction',
        'hurst_exponent', 'consecutive_dir_bars', 'vol_divergence',
        'event_vol_interaction', 'range_expansion', 'abs_gap', 'gap_expansion',
        'market_state', 'target_market_state', 'target_dir',
    ]]


def train_direction_model(df, feature_cols, start_idx, end_idx):
    base_feature_cols = select_direction_feature_cols(feature_cols)
    X_train = df.iloc[start_idx:end_idx][base_feature_cols].values
    y_train = df.iloc[start_idx:end_idx]['target_dir'].values

    model = xgb.XGBClassifier(**DIR_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model, base_feature_cols


def predict_direction_signals(model, df_chunk, base_feature_cols):
    X_chunk = df_chunk[base_feature_cols].values
    pred = model.predict(X_chunk)
    return np.where(pred == 1, 1, -1)
