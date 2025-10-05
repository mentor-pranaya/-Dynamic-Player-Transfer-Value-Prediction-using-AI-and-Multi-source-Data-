import numpy as np
import os

MODEL_DIR = "/content/drive/MyDrive/TransferIQ_Models"

# Load validation sets
val_data = np.load(os.path.join(MODEL_DIR, "val_data.npz"))

X_val = val_data["X_val"]
y_val = val_data["y_val"]

print("✅ Loaded X_val and y_val")
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# -----------------------------
# Milestone : Model Evaluation, Hyperparameter Tuning, and Testing
# -----------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import joblib

print("\n=== Milestone 6: Model Evaluation, Hyperparameter Tuning, and Testing ===")

# -----------------------------
# 1) Evaluation helper (RMSE, MAE, R2)
# -----------------------------
def evaluate_full(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    mae = mean_absolute_error(y_t, y_pred)
    rmse = np.sqrt(mean_squared_error(y_t, y_pred))
    r2 = r2_score(y_t, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []


# Ensemble Models
results.append(evaluate_full("XGBoost", xgb_model, X_val_sample, y_val_sample))
results.append(evaluate_full("LightGBM", lgb_model, X_val_sample, y_val_sample))
results.append(evaluate_full("Stacking Ensemble", stack_model, X_val_sample, y_val_sample))


# LSTM (multivariate)
y_pred_val_scaled = multi_model.predict(X_val_scaled)
y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1,1)).reshape(-1)
y_true_val = scaler_y.inverse_transform(y_val_scaled.reshape(-1,1)).reshape(-1)

mae = mean_absolute_error(y_true_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
r2 = r2_score(y_true_val, y_pred_val)
results.append({"Model": "Multivariate LSTM", "MAE": mae, "RMSE": rmse, "R2": r2})

# Print comparison
report_df = pd.DataFrame(results)
print("\n=== Model Evaluation Report ===")
print(report_df)


# 2) Hyperparameter tuning — Ensemble (Randomized Search)
# -----------------------------
print("\nHyperparameter tuning for XGBoost and LightGBM...")

xgb_params = {
    "n_estimators": [200, 500, 800],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1]
}
xgb_search = RandomizedSearchCV(xgb.XGBRegressor(random_state=RANDOM_SEED),
                                xgb_params, n_iter=5, scoring="neg_mean_squared_error",
                                cv=3, verbose=1, n_jobs=-1)
xgb_search.fit(X_train_tree, y_train_tree)
xgb_best = xgb_search.best_estimator_
print("Best XGBoost params:", xgb_search.best_params_)

lgb_params = {
    "n_estimators": [200, 500, 800],
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1]
}
lgb_search = RandomizedSearchCV(lgb.LGBMRegressor(random_state=RANDOM_SEED),
                                lgb_params, n_iter=5, scoring="neg_mean_squared_error",
                                cv=3, verbose=1, n_jobs=-1)
lgb_search.fit(X_train_tree, y_train_tree)
lgb_best = lgb_search.best_estimator_
print("Best LightGBM params:", lgb_search.best_params_)



# 3) Hyperparameter tuning — LSTM with KerasTuner
# -----------------------------
print("\nHyperparameter tuning for LSTM (using KerasTuner)...")

!pip install keras-tuner -q
import keras_tuner as kt
from tensorflow.keras import layers, models

def build_lstm_model(hp):
    model = models.Sequential()
    model.add(layers.LSTM(units=hp.Int("units", min_value=32, max_value=128, step=32),
                          input_shape=(N_STEPS_IN, n_feat)))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("lr", [1e-2, 1e-3, 1e-4])
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model

tuner = kt.RandomSearch(
    build_lstm_model,
    objective="val_loss",
    max_trials=4,
    executions_per_trial=1,
    directory="tuner_logs",
    project_name="lstm_tuning"
)

tuner.search(X_train_scaled, y_train_scaled.reshape(-1,1),
             epochs=5, validation_split=0.2, verbose=1)

best_lstm = tuner.get_best_models(1)[0]
print("Best LSTM hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)


# 4) Final Testing & Save tuned models
# -----------------------------
joblib.dump(xgb_best, os.path.join(MODEL_DIR, "xgb_best.save"))
joblib.dump(lgb_best, os.path.join(MODEL_DIR, "lgb_best.save"))
best_lstm.save(os.path.join(MODEL_DIR, "lstm_best.h5"))

print("\n=== Final tuned models saved in", MODEL_DIR, "===")

