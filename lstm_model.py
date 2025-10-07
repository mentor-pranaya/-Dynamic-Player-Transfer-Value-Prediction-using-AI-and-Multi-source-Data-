"""
lstm_model.py
-------------
Builds and trains an LSTM model on engineered player performance data.
Predicts player market value (value_euro) based on ratings, sentiment, and age-related factors.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------------------
# Setup logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------
# Paths
# ------------------------------
DATA_PATH = Path("data/processed/processed_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lstm_model.h5"

# ------------------------------
# Load and preprocess data
# ------------------------------
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Processed dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logging.info(f"✅ Loaded dataset with shape: {df.shape}")
    return df


def prepare_data(df):
    """Prepares data for LSTM model input."""
    features = [
        "overall_rating",
        "potential",
        "normalized_sentiment",
        "age_factor",
        "skill_index",
        "physical_index",
        "consistency_index",
        "age_adjusted_performance",
    ]

    # Handle missing values
    df[features] = df[features].fillna(df[features].mean())

    X = df[features].values
    y = df["value_euro"].values

    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape input to [samples, timesteps, features]
    X_scaled = np.expand_dims(X_scaled, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# ------------------------------
# Build LSTM Model
# ------------------------------
def build_lstm_model(input_shape):
    """Creates a sequential LSTM model."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ------------------------------
# Train and Evaluate
# ------------------------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    logging.info(f"📈 Model Performance:")
    logging.info(f"   MAE:  {mae:.4f}")
    logging.info(f"   RMSE: {rmse:.4f}")
    logging.info(f"   R²:   {r2:.4f}")

    return history, (mae, rmse, r2)


# ------------------------------
# Main
# ------------------------------
def main():
    logging.info("🚀 Starting LSTM model training pipeline...")

    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test, _, _ = prepare_data(df)

    # Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary(print_fn=lambda x: logging.info(x))

    # Train and evaluate
    history, metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    # Save final model
    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)

    logging.info(f"✅ LSTM model trained and saved at {MODEL_PATH}")
    logging.info("🎯 Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
