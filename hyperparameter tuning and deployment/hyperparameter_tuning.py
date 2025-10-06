import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    base_dir = r"C:/Users/Abhinav/Desktop/Project"
    X_file = os.path.join(base_dir, "X_scaled_features.csv")
    y_file = os.path.join(base_dir, "y_target.csv")

    X = pd.read_csv(X_file)
    y = pd.read_csv(y_file)

    print("✅ Data successfully loaded for model training.")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )
    print("\n📊 Train/Test split complete.")

    param_candidates = {
        "n_estimators": [150, 250],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6, 8]
    }
    print("🔎 Grid search parameters defined.")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=123,
        n_jobs=1
    )

    print("\n🚀 Beginning hyperparameter tuning...")
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_candidates,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("\n✅ Grid search complete.")
    print(f"Best parameters: {grid.best_params_}")

    tuned_model = grid.best_estimator_
    predictions = tuned_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n📈 Tuned Model Performance:")
    print(f"MAE:  {mae:.2f} million €")
    print(f"RMSE: {rmse:.2f} million €")
    print(f"R²:   {r2:.2f}")

if __name__ == "__main__":
    main()
