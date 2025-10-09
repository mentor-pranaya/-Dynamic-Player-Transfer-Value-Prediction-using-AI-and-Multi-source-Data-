# XGBoost Model Hyperparameter Tuning Results

## Overview
This report documents the results of the Randomized Search Cross-Validation used to optimize the XGBoost Regressor for predicting transfer market values.

- **Baseline R-squared (Before Tuning):** 0.6384 (from initial run)
- **Tuning Method:** Randomized Search Cross-Validation (30 iterations, 5 folds) - Speed Optimized
- **Time of Analysis:** 2025-10-09 08:04:40

## Best Parameters Found
| Parameter | Optimal Value |
| :--- | :--- |
| `colsample_bytree` | `0.8156249507619748` |
| `gamma` | `0.007983126110107097` |
| `learning_rate` | `0.0792681476866447` |
| `max_depth` | `6` |
| `n_estimators` | `466` |
| `subsample` | `0.9049790556476374` |

## Final Evaluation Metrics (on Test Set)
Using the best parameters found by the search:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R-squared (R²)** | **0.6414** | Percentage of the target variance explained by the model. |
| **Root Mean Squared Error (RMSE)** | 0.7882 EUR | Standard deviation of the prediction errors (should be minimized). |
| **Mean Absolute Error (MAE)** | 0.2147 EUR | Average magnitude of errors (less sensitive to outliers than RMSE). |

## Next Steps
The next step is to perform **Feature Importance Analysis** to understand which player attributes are most critical to the final market value prediction.
