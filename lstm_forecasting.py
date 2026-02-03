import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================
# Data preprocessing utils
# ==========================

def fit_feature_target_scalers(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """
    Fit MinMax scalers for features and target separately.
    """
    feature_scaler = MinMaxScaler(feature_range=feature_range)
    target_scaler = MinMaxScaler(feature_range=feature_range)

    feature_scaler.fit(df[feature_columns].values)
    target_scaler.fit(df[[target_column]].values)
    return feature_scaler, target_scaler


def transform_features_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform features and target using provided scalers.
    Returns arrays: features_scaled, target_scaled.
    """
    X = feature_scaler.transform(df[feature_columns].values)
    y = target_scaler.transform(df[[target_column]].values).reshape(-1)
    return X, y


def generate_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for supervised learning on time series.

    - X: feature matrix of shape (num_samples, num_features)
    - y: target vector of shape (num_samples,)
    - lookback: number of past timesteps to use as input
    - horizon: number of future steps to predict (1 for single-step)
    - stride: step size for the sliding window

    Returns:
      - X_seq: (num_windows, lookback, num_features)
      - y_seq: (num_windows,) if horizon==1 else (num_windows, horizon)
    """
    num_samples = X.shape[0]
    num_features = X.shape[1]
    X_seq = []
    y_seq = []

    end = num_samples - lookback - horizon + 1
    for start_idx in range(0, end, stride):
        end_idx = start_idx + lookback
        X_window = X[start_idx:end_idx]
        if horizon == 1:
            y_window = y[end_idx]
        else:
            y_window = y[end_idx:end_idx + horizon]
        X_seq.append(X_window)
        y_seq.append(y_window)

    X_seq = np.array(X_seq).reshape(-1, lookback, num_features)
    if horizon == 1:
        y_seq = np.array(y_seq).reshape(-1)
    else:
        y_seq = np.array(y_seq).reshape(-1, horizon)
    return X_seq, y_seq


def train_val_test_split_sequences(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split of sequence data into train/val/test sets.
    """
    n = X_seq.shape[0]
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)

    X_train = X_seq[: n - n_test - n_val]
    y_train = y_seq[: n - n_test - n_val]
    X_val = X_seq[n - n_test - n_val: n - n_test]
    y_val = y_seq[n - n_test - n_val: n - n_test]
    X_test = X_seq[n - n_test:]
    y_test = y_seq[n - n_test:]
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================
# Model builders
# ==========================

def build_univariate_lstm(
    lookback: int,
    units: int = 64,
    dropout: float = 0.0,
    learning_rate: float = 1e-3
) -> Model:
    """
    Build a univariate single-step LSTM regression model.
    Input shape: (lookback, 1)
    Output: scalar next value.
    """
    inputs = Input(shape=(lookback, 1))
    x = LSTM(units, dropout=dropout, recurrent_dropout=0.0)(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_multivariate_lstm(
    lookback: int,
    num_features: int,
    units: int = 64,
    dropout: float = 0.0,
    learning_rate: float = 1e-3
) -> Model:
    """
    Build a multivariate single-step LSTM regression model.
    Input shape: (lookback, num_features)
    Output: scalar next target value.
    """
    inputs = Input(shape=(lookback, num_features))
    x = LSTM(units, dropout=dropout, recurrent_dropout=0.0)(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


essential_doc = """
Encoder-Decoder LSTM:
- Encoder processes input sequence and returns final hidden state.
- Decoder is initialized with encoder state and generates horizon-length outputs.
"""


def build_encoder_decoder_lstm(
    lookback: int,
    num_features: int,
    horizon: int,
    encoder_units: int = 64,
    decoder_units: int = 64,
    learning_rate: float = 1e-3
) -> Model:
    """
    Build an encoder-decoder LSTM for multi-step forecasting.
    Input: (lookback, num_features)
    Output: (horizon,) predictions.
    """
    encoder_inputs = Input(shape=(lookback, num_features))
    encoder_output, state_h, state_c = LSTM(encoder_units, return_state=True)(encoder_inputs)

    repeated_context = RepeatVector(horizon)(encoder_output)
    decoder_lstm = LSTM(decoder_units, return_sequences=True)
    decoder_outputs = decoder_lstm(repeated_context, initial_state=[state_h, state_c])

    decoder_dense = TimeDistributed(Dense(1))
    decoder_outputs = decoder_dense(decoder_outputs)

    outputs = tf.keras.layers.Flatten()(decoder_outputs)

    model = Model(encoder_inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


# ==========================
# Training, evaluation, plotting
# ==========================

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    model_dir: Optional[str] = None,
    model_name: str = "model"
) -> tf.keras.callbacks.History:
    """
    Train with early stopping and best checkpoint saving.
    """
    callbacks = []
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    callbacks.append(es)

    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        ckpt_path = os.path.join(model_dir, f"{model_name}.weights.h5")
        mc = ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
        callbacks.append(mc)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def plot_loss(history: tf.keras.callbacks.History, title: str, save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    save_path: Optional[str] = None
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Transfer Value")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ==========================
# Example CLI to run models
# ==========================

def try_infer_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "transfer_value",
        "market_value",
        "value_eur",
        "next_transfer_value",
        "valuation"
    ]
    for name in candidates:
        if name in df.columns:
            return name
    return None


def run_univariate_pipeline(
    df: pd.DataFrame,
    target_column: str,
    lookback: int,
    val_ratio: float,
    test_ratio: float,
    epochs: int,
    batch_size: int,
    out_dir: str
) -> Dict[str, float]:
    feature_columns = [target_column]
    feature_scaler, target_scaler = fit_feature_target_scalers(df, feature_columns, target_column)
    X_all, y_all = transform_features_target(df, feature_columns, target_column, feature_scaler, target_scaler)
    X_seq, y_seq = generate_sliding_windows(X_all, y_all, lookback=lookback, horizon=1)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(
        X_seq, y_seq, val_ratio=val_ratio, test_ratio=test_ratio
    )

    # Reshape to (samples, lookback, 1)
    X_train_u = X_train.reshape((-1, lookback, 1))
    X_val_u = X_val.reshape((-1, lookback, 1))
    X_test_u = X_test.reshape((-1, lookback, 1))

    model = build_univariate_lstm(lookback=lookback)

    history = train_model(
        model,
        X_train_u,
        y_train,
        X_val_u,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_dir=out_dir,
        model_name="univariate_lstm"
    )

    plot_loss(history, title="Univariate LSTM Loss", save_path=os.path.join(out_dir, "univariate_loss.png"))

    y_pred_scaled = model.predict(X_test_u).reshape(-1)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred_inv = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    metrics = evaluate_regression(y_test_inv, y_pred_inv)

    plot_predictions(
        y_test_inv,
        y_pred_inv,
        title="Univariate LSTM: Predictions vs Actuals (Test)",
        save_path=os.path.join(out_dir, "univariate_predictions.png")
    )
    return metrics


def run_multivariate_pipeline(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    lookback: int,
    val_ratio: float,
    test_ratio: float,
    epochs: int,
    batch_size: int,
    out_dir: str
) -> Dict[str, float]:
    feature_scaler, target_scaler = fit_feature_target_scalers(df, feature_columns, target_column)
    X_all, y_all = transform_features_target(df, feature_columns, target_column, feature_scaler, target_scaler)
    X_seq, y_seq = generate_sliding_windows(X_all, y_all, lookback=lookback, horizon=1)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(
        X_seq, y_seq, val_ratio=val_ratio, test_ratio=test_ratio
    )

    model = build_multivariate_lstm(lookback=lookback, num_features=len(feature_columns))

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_dir=out_dir,
        model_name="multivariate_lstm"
    )

    plot_loss(history, title="Multivariate LSTM Loss", save_path=os.path.join(out_dir, "multivariate_loss.png"))

    y_pred_scaled = model.predict(X_test).reshape(-1)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred_inv = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    metrics = evaluate_regression(y_test_inv, y_pred_inv)
    plot_predictions(
        y_test_inv,
        y_pred_inv,
        title="Multivariate LSTM: Predictions vs Actuals (Test)",
        save_path=os.path.join(out_dir, "multivariate_predictions.png")
    )
    return metrics


def run_encoder_decoder_pipeline(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    lookback: int,
    horizon: int,
    val_ratio: float,
    test_ratio: float,
    epochs: int,
    batch_size: int,
    out_dir: str
) -> Dict[str, float]:
    feature_scaler, target_scaler = fit_feature_target_scalers(df, feature_columns, target_column)
    X_all, y_all = transform_features_target(df, feature_columns, target_column, feature_scaler, target_scaler)
    X_seq, y_seq = generate_sliding_windows(X_all, y_all, lookback=lookback, horizon=horizon)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(
        X_seq, y_seq, val_ratio=val_ratio, test_ratio=test_ratio
    )

    model = build_encoder_decoder_lstm(
        lookback=lookback,
        num_features=len(feature_columns),
        horizon=horizon
    )

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_dir=out_dir,
        model_name="encoder_decoder_lstm"
    )

    plot_loss(history, title="Encoder-Decoder LSTM Loss", save_path=os.path.join(out_dir, "encoder_decoder_loss.png"))

    y_pred_scaled = model.predict(X_test)
    # Inverse transform each step independently using the target scaler
    y_pred_inv = target_scaler.inverse_transform(y_pred_scaled)
    y_test_inv = target_scaler.inverse_transform(y_test)

    # Evaluate using last step and average as examples
    metrics_last = evaluate_regression(y_test_inv[:, -1], y_pred_inv[:, -1])
    metrics_avg = evaluate_regression(y_test_inv.mean(axis=1), y_pred_inv.mean(axis=1))

    # Plot a sample of sequences
    sample_idx = min(200, y_test_inv.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_inv[:sample_idx, -1], label="Actual (last step)")
    plt.plot(y_pred_inv[:sample_idx, -1], label="Predicted (last step)")
    plt.title("Encoder-Decoder LSTM: Last-step Predictions vs Actuals (Test)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Transfer Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "encoder_decoder_predictions_last_step.png"), dpi=150)
    plt.close()

    # Multi-step trajectory plot for first example
    if y_test_inv.shape[0] > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, horizon + 1), y_test_inv[0], marker="o", label="Actual")
        plt.plot(range(1, horizon + 1), y_pred_inv[0], marker="x", label="Predicted")
        plt.title("Encoder-Decoder LSTM: Multi-step Trajectory (1st Test Sample)")
        plt.xlabel("Forecast Step")
        plt.ylabel("Transfer Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "encoder_decoder_predictions_multistep_sample.png"), dpi=150)
        plt.close()

    return {"rmse_last": metrics_last["rmse"], "mae_last": metrics_last["mae"], "rmse_avg": metrics_avg["rmse"], "mae_avg": metrics_avg["mae"]}


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="LSTM forecasting for player transfer values")
    parser.add_argument("--csv", type=str, default=os.path.join("processed", "features_final.csv"), help="Path to CSV with features/target")
    parser.add_argument("--target", type=str, default=None, help="Target column name (transfer value)")
    parser.add_argument("--features", type=str, nargs="*", default=None, help="Explicit list of feature columns (space-separated)")
    parser.add_argument("--lookback", type=int, default=12, help="Past timesteps for input")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast steps ahead for encoder-decoder")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default=os.path.join("processed", "lstm_outputs"))

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_dataframe(args.csv)

    target_col = args.target or try_infer_target_column(df)
    if target_col is None:
        raise ValueError("Could not infer target column. Please pass --target explicitly.")

    if args.features is None:
        feature_cols = [c for c in df.columns if c != target_col]
    else:
        feature_cols = args.features

    # Ensure numeric only
    df = df[feature_cols + [target_col]]
    df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)

    # 1) Univariate
    uni_metrics = run_univariate_pipeline(
        df=df[[target_col]].copy(),
        target_column=target_col,
        lookback=args.lookback,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_dir=args.out_dir
    )
    with open(os.path.join(args.out_dir, "univariate_metrics.txt"), "w") as f:
        f.write(str(uni_metrics))

    # 2) Multivariate
    multi_metrics = run_multivariate_pipeline(
        df=df.copy(),
        feature_columns=[c for c in df.columns if c != target_col],
        target_column=target_col,
        lookback=args.lookback,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_dir=args.out_dir
    )
    with open(os.path.join(args.out_dir, "multivariate_metrics.txt"), "w") as f:
        f.write(str(multi_metrics))

    # 3) Encoder-Decoder multi-step
    ed_metrics = run_encoder_decoder_pipeline(
        df=df.copy(),
        feature_columns=[c for c in df.columns if c != target_col],
        target_column=target_col,
        lookback=args.lookback,
        horizon=args.horizon,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_dir=args.out_dir
    )
    with open(os.path.join(args.out_dir, "encoder_decoder_metrics.txt"), "w") as f:
        f.write(str(ed_metrics))

    print("Univariate metrics:", uni_metrics)
    print("Multivariate metrics:", multi_metrics)
    print("Encoder-Decoder metrics:", ed_metrics)


if __name__ == "__main__":
    main()
