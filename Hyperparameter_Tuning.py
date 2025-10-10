import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# This 'if' statement is still best practice, so we keep it.
if __name__ == '__main__':

    # --- Load Your Model-Ready Data ---
    data_folder = 'data'
    X_path = os.path.join(data_folder, 'X_scaled_features.csv')
    y_path = os.path.join(data_folder, 'y_target.csv')

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    print("--- Model-ready data loaded successfully ---")

    # --- Split Data for Training and Testing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("
--- Data split into training and testing sets ---")

    # --- Define the Hyperparameter Grid ---
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    print("
--- Hyperparameter grid created ---")

    # --- Set Up and Run GridSearchCV ---
    model = XGBRegressor(random_state=42)

    # THE FIX: Changed n_jobs=-1 to n_jobs=1 to disable parallel processing and avoid the error.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=1  # This change prevents the multiprocessing error on Windows
    )

    print("
--- Starting hyperparameter tuning... ---")
    grid_search.fit(X_train, y_train)

    print("
--- Hyperparameter tuning complete! ---")
    print("Best hyperparameters found: ", grid_search.best_params_)

    # --- Evaluate the Tuned Model ---
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)

    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    r2_tuned = r2_score(y_test, y_pred_tuned)

    print("
--- Tuned Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae_tuned:.2f} million €")
    print(f"Root Mean Squared Error (RMSE): {rmse_tuned:.2f} million €")
    print(f"R-squared (R²): {r2_tuned:.2f}")