from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error # type: ignore
import sys
print(sys.executable)

# Get LSTM predictions as features
lstm_train_pred = model.predict(X_train)
lstm_test_pred = model.predict(X_test)

# Use last time step features for each sample
X_train_flat = X_train[:, -1, :]
X_test_flat = X_test[:, -1, :]

# Combine LSTM output with original features
X_train_ensemble = np.hstack([X_train_flat, lstm_train_pred])
X_test_ensemble = np.hstack([X_test_flat, lstm_test_pred])

# XGBoost Ensemble
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_ensemble, y_train)
ensemble_pred = xgb.predict(X_test_ensemble)

print("LSTM RMSE:", np.sqrt(mean_squared_error(y_test, lstm_test_pred)))
print("Ensemble RMSE:", np.sqrt(mean_squared_error(y_test, ensemble_pred)))