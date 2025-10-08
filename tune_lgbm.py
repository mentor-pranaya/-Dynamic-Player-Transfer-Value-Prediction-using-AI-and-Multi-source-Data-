import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, make_scorer  # <-- CHANGED IMPORT


def tune_lightgbm(file_path="enhanced_player_stats.csv"):
    """
    Performs Grid Search Cross-Validation using Mean Absolute Error (MAE)
    to find optimal hyperparameters for LightGBM.
    """
    # --- 1. Data Preparation (Reuse Logic) ---
    target = 'current_value'

    cols_to_drop = [
        target, 'player', 'team', 'name', 'highest_value', 'position',
        'minutes played', 'days_injured'
    ]

    df = pd.read_csv(file_path)
    features_df = df.drop(columns=cols_to_drop)

    # Log Transformation on target
    y = np.log1p(df[target])

    # One-Hot Encoding and feature cleaning
    X = pd.get_dummies(features_df, columns=['Age_Group'], drop_first=True)
    new_columns = X.columns
    new_columns = new_columns.str.replace(r'[\[\]<()]', '_', regex=True)
    X.columns = new_columns

    # --- 2. Hyperparameter Tuning Setup ---
    print("Starting Grid Search Hyperparameter Tuning for LightGBM (Using MAE Score)...")

    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'max_depth': [3, 5, 7]
    }

    # Initialize the model
    lgbm = LGBMRegressor(random_state=42, n_jobs=-1, objective='regression')

    # Define the cross-validation strategy
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    # 🛑 CORRECTED SCORER: Use Mean Absolute Error (MAE) for stability
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Setup the GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        scoring=mae_scorer,  # <-- USING MAE SCORER
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    # --- 3. Run Search ---
    grid_search.fit(X, y)

    # --- 4. Results ---
    print("-" * 50)
    print("Grid Search Complete.")
    # Note: Score is reported as Negative MAE, so we negate it for interpretation
    print(f"Best MAE (Log-Transformed): {-grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Store the best model to a CSV for easy retrieval
    best_params_df = pd.DataFrame([grid_search.best_params_])
    best_params_df.to_csv('best_lgbm_params.csv', index=False)
    print("\nBest parameters saved to 'best_lgbm_params.csv'.")

    return grid_search.best_params_


if __name__ == "__main__":
    tune_lightgbm()