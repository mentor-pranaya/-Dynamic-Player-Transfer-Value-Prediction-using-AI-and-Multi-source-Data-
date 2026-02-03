import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning (LSTM)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Gradient Boosting Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def generate_synthetic_dataset(
    num_players: int = 150,
    time_steps: int = 24,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic panel time-series dataset with player performance, market and sentiment features.

    Columns:
      - player_id, time_index
      - perf_0..perf_4 (performance stats)
      - market_0..market_2 (market trends)
      - sent_0..sent_1 (social sentiment)
      - target (transfer value next period)
    """
    rng = np.random.default_rng(random_seed)
    rows = []

    for player_id in range(num_players):
        # Player base skill and popularity influence dynamics
        base_skill = rng.normal(0.0, 1.0)
        popularity = rng.uniform(-1.0, 1.0)

        perf = rng.normal(loc=base_skill, scale=0.8, size=(time_steps, 5))
        market = rng.normal(loc=0.0, scale=0.6, size=(time_steps, 3))
        sent = rng.normal(loc=popularity, scale=0.7, size=(time_steps, 2))

        # latent transfer value follows AR(1) with exogenous inputs
        target = np.zeros(time_steps)
        target[0] = 10 + 2 * base_skill + 3 * popularity + rng.normal(0, 1)
        for t in range(1, time_steps):
            perf_signal = 0.5 * perf[t].mean()
            market_signal = 0.4 * market[t].mean()
            sent_signal = 0.6 * sent[t].mean()
            target[t] = (
                0.75 * target[t - 1]
                + 2.0 * perf_signal
                + 1.5 * market_signal
                + 1.8 * sent_signal
                + rng.normal(0, 1.2)
            )

        for t in range(time_steps):
            row = {
                "player_id": player_id,
                "time_index": t,
                **{f"perf_{i}": perf[t, i] for i in range(5)},
                **{f"market_{i}": market[t, i] for i in range(3)},
                **{f"sent_{i}": sent[t, i] for i in range(2)},
                "target": target[t],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def create_lstm_supervised(
    df: pd.DataFrame,
    sequence_length: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Turn panel data into supervised sequences for LSTM.

    For each player, create sequences of length `sequence_length` using the time-varying
    features (perf, market, sent) to predict target at the sequence end time.

    Returns:
      - X_seq: (n_samples, sequence_length, n_features_ts)
      - y_seq: (n_samples,)
      - X_tabular: (n_samples, n_features_tab) from last step of sequence
      - meta: DataFrame with columns [player_id, time_index] for the predicted time
    """
    feature_cols_ts = [
        *[f"perf_{i}" for i in range(5)],
        *[f"market_{i}" for i in range(3)],
        *[f"sent_{i}" for i in range(2)],
    ]

    # We'll use last-step raw features as tabular inputs
    feature_cols_tab = feature_cols_ts

    sequences = []
    targets = []
    tabular_last = []
    metas = []

    for player_id, g in df.sort_values(["player_id", "time_index"]).groupby("player_id"):
        g = g.reset_index(drop=True)
        values_ts = g[feature_cols_ts].values
        y_vals = g["target"].values

        if len(g) < sequence_length:
            continue

        for end_idx in range(sequence_length - 1, len(g)):
            start_idx = end_idx - (sequence_length - 1)
            X_seq = values_ts[start_idx : end_idx + 1]
            y = y_vals[end_idx]
            X_tab = values_ts[end_idx]
            sequences.append(X_seq)
            targets.append(y)
            tabular_last.append(X_tab)
            metas.append({"player_id": player_id, "time_index": int(g.loc[end_idx, "time_index"])})

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(targets, dtype=np.float32)
    X_tabular = np.array(tabular_last, dtype=np.float32)
    meta = pd.DataFrame(metas)
    return X_seq, y_seq, X_tabular, meta


def temporal_split_by_time(
    meta: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices by time_index thresholds to avoid leakage across time."""
    unique_times = np.sort(meta["time_index"].unique())
    t_train_max = unique_times[int(len(unique_times) * train_frac) - 1]
    t_val_max = unique_times[int(len(unique_times) * (train_frac + val_frac)) - 1]

    idx_train = meta.index[meta["time_index"] <= t_train_max].to_numpy()
    idx_val = meta.index[(meta["time_index"] > t_train_max) & (meta["time_index"] <= t_val_max)].to_numpy()
    idx_test = meta.index[meta["time_index"] > t_val_max].to_numpy()
    return idx_train, idx_val, idx_test


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def plot_lstm_loss(history: tf.keras.callbacks.History, out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predictions(y_true: np.ndarray, preds: Dict[str, np.ndarray], out_path: str, sample_n: int = 400) -> None:
    plt.figure(figsize=(9, 6))
    n = min(sample_n, len(y_true))
    x_axis = np.arange(n)
    plt.plot(x_axis, y_true[:n], label="Actual", linewidth=2)
    for name, yhat in preds.items():
        plt.plot(x_axis, yhat[:n], label=name, alpha=0.8)
    plt.xlabel("Sample")
    plt.ylabel("Transfer Value")
    plt.title("Predictions vs Actual (subset)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]], out_path: str) -> None:
    df_metrics = (
        pd.DataFrame(metrics_dict)
        .T
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    plt.figure(figsize=(9, 5))
    mlong = df_metrics.melt(id_vars=["Model"], value_vars=["RMSE", "MAE", "R2"], var_name="Metric", value_name="Value")
    sns.barplot(data=mlong, x="Metric", y="Value", hue="Model")
    plt.title("Model Performance Comparison (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    ensure_dir(output_dir)

    # 1) Data loading & preprocessing (synthetic)
    df = generate_synthetic_dataset(num_players=200, time_steps=24, random_seed=7)

    # Create supervised sequences
    seq_len = 6
    X_seq, y_seq, X_tab, meta = create_lstm_supervised(df, sequence_length=seq_len)

    # Split by time
    idx_train, idx_val, idx_test = temporal_split_by_time(meta, train_frac=0.6, val_frac=0.2)

    # Scale time-series features for LSTM
    n_features_ts = X_seq.shape[-1]
    scaler_ts = StandardScaler()
    # Fit on train windows only (flatten then reshape back)
    X_seq_train_flat = X_seq[idx_train].reshape(-1, n_features_ts)
    scaler_ts.fit(X_seq_train_flat)

    def transform_seq(X_seq_in: np.ndarray) -> np.ndarray:
        flat = X_seq_in.reshape(-1, n_features_ts)
        flat_scaled = scaler_ts.transform(flat)
        return flat_scaled.reshape(X_seq_in.shape)

    X_train_seq = transform_seq(X_seq[idx_train])
    X_val_seq = transform_seq(X_seq[idx_val])
    X_test_seq = transform_seq(X_seq[idx_test])

    y_train = y_seq[idx_train]
    y_val = y_seq[idx_val]
    y_test = y_seq[idx_test]

    # 2) LSTM model training
    tf.random.set_seed(123)
    np.random.seed(123)
    lstm = build_lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, verbose=0)

    history = lstm.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[es, rlrop],
        verbose=0,
    )

    plot_lstm_loss(history, os.path.join(output_dir, "lstm_loss_curve.png"))

    # LSTM predictions
    yhat_train_lstm = lstm.predict(X_train_seq, verbose=0).reshape(-1)
    yhat_val_lstm = lstm.predict(X_val_seq, verbose=0).reshape(-1)
    yhat_test_lstm = lstm.predict(X_test_seq, verbose=0).reshape(-1)

    lstm_metrics = {
        "Train": evaluate_regression(y_train, yhat_train_lstm),
        "Val": evaluate_regression(y_val, yhat_val_lstm),
        "Test": evaluate_regression(y_test, yhat_test_lstm),
    }

    # 3) Tabular models (XGB, LGB) with and without LSTM predictions
    # Scale tabular inputs (tree models generally do not need scaling, but consistent preprocessing helps)
    scaler_tab = StandardScaler()
    X_tab_train = X_tab[idx_train]
    X_tab_val = X_tab[idx_val]
    X_tab_test = X_tab[idx_test]

    scaler_tab.fit(X_tab_train)
    X_tab_train_s = scaler_tab.transform(X_tab_train)
    X_tab_val_s = scaler_tab.transform(X_tab_val)
    X_tab_test_s = scaler_tab.transform(X_tab_test)

    # Baselines (no LSTM feature)
    xgb_base = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=123,
        tree_method="hist",
    )
    xgb_base.fit(
        np.vstack([X_tab_train_s, X_tab_val_s]),
        np.hstack([y_train, y_val]),
        eval_set=[(X_tab_val_s, y_val)],
        verbose=False,
    )

    lgb_base = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=123,
    )
    lgb_base.fit(
        np.vstack([X_tab_train_s, X_tab_val_s]),
        np.hstack([y_train, y_val]),
        eval_set=[(X_tab_val_s, y_val)],
    )

    yhat_test_xgb_base = xgb_base.predict(X_tab_test_s)
    yhat_test_lgb_base = lgb_base.predict(X_tab_test_s)

    # Ensembles (append LSTM predictions)
    X_train_ens = np.column_stack([X_tab_train_s, yhat_train_lstm])
    X_val_ens = np.column_stack([X_tab_val_s, yhat_val_lstm])
    X_test_ens = np.column_stack([X_tab_test_s, yhat_test_lstm])

    xgb_ens = XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=123,
        tree_method="hist",
    )
    xgb_ens.fit(
        np.vstack([X_train_ens, X_val_ens]),
        np.hstack([y_train, y_val]),
        eval_set=[(X_val_ens, y_val)],
        verbose=False,
    )

    lgb_ens = LGBMRegressor(
        n_estimators=1400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=123,
    )
    lgb_ens.fit(
        np.vstack([X_train_ens, X_val_ens]),
        np.hstack([y_train, y_val]),
        eval_set=[(X_val_ens, y_val)],
    )

    yhat_test_xgb_ens = xgb_ens.predict(X_test_ens)
    yhat_test_lgb_ens = lgb_ens.predict(X_test_ens)

    # 4) Evaluation & Plots
    metrics_test = {
        "LSTM": evaluate_regression(y_test, yhat_test_lstm),
        "XGB Base": evaluate_regression(y_test, yhat_test_xgb_base),
        "LGB Base": evaluate_regression(y_test, yhat_test_lgb_base),
        "XGB + LSTM": evaluate_regression(y_test, yhat_test_xgb_ens),
        "LGB + LSTM": evaluate_regression(y_test, yhat_test_lgb_ens),
    }

    # Save metrics to a text file
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
        for model_name, md in metrics_test.items():
            f.write(f"{model_name}: RMSE={md['RMSE']:.4f}, MAE={md['MAE']:.4f}, R2={md['R2']:.4f}\n")

    # Plots
    plot_predictions(
        y_true=y_test,
        preds={
            "LSTM": yhat_test_lstm,
            "XGB Base": yhat_test_xgb_base,
            "LGB Base": yhat_test_lgb_base,
            "XGB + LSTM": yhat_test_xgb_ens,
            "LGB + LSTM": yhat_test_lgb_ens,
        },
        out_path=os.path.join(output_dir, "pred_vs_actual.png"),
    )

    plot_model_comparison(metrics_test, os.path.join(output_dir, "model_comparison.png"))

    # Print metrics
    print("Performance on Test Set:")
    for model_name, md in metrics_test.items():
        print(f"- {model_name}: RMSE={md['RMSE']:.4f}, MAE={md['MAE']:.4f}, R2={md['R2']:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()


