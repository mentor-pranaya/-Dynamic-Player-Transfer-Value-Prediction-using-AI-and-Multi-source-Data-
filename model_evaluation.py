import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Assuming you have the unscaled arrays from Step 1:
# y_test_unscaled, y_pred_uni_unscaled, y_pred_multi_unscaled
# We'll reload them from the results_df for robustness:

# Reload the results DataFrame saved in Step 1
results_df = pd.read_csv(r'D:\Pythonproject\datasets\pythonProject\lstm_prediction_results.csv')


# Convert the string currency back to numbers for calculation
def parse_currency(value):
    return float(str(value).replace('€', '').replace(',', ''))


y_actual = results_df['Actual_Value'].apply(parse_currency).values.reshape(-1, 1)
y_pred_uni = results_df['Uni_Predicted_Value'].apply(parse_currency).values.reshape(-1, 1)
y_pred_multi = results_df['Multi_Predicted_Value'].apply(parse_currency).values.reshape(-1, 1)


# --- 1. Metric Calculation Function ---
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {model_name} Performance on Test Set ---")
    print(f"RMSE (Root Mean Squared Error): €{rmse:,.0f}")
    print(f"MAE (Mean Absolute Error):      €{mae:,.0f}")
    print(f"R-squared (R2 Score):           {r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


# Calculate Metrics
uni_metrics = evaluate_model(y_actual, y_pred_uni, "Univariate LSTM (Goals)")
multi_metrics = evaluate_model(y_actual, y_pred_multi, "Multivariate LSTM (10 Features)")

# --- 2. Visualization (Actual vs. Predicted) ---

# Set plot style
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Determine the max value for plotting limit
max_val = max(y_actual.max(), y_pred_multi.max(), y_pred_uni.max()) * 1.05
plot_range = [0, max_val]

# Univariate Plot
axes[0].scatter(y_actual, y_pred_uni, alpha=0.6, color='skyblue')
axes[0].plot(plot_range, plot_range, color='red', linestyle='--', label='Ideal Prediction')
axes[0].set_title('Univariate LSTM: Actual vs. Predicted Value')
axes[0].set_xlabel('Actual Value (€)')
axes[0].set_ylabel('Predicted Value (€)')
axes[0].ticklabel_format(style='plain', axis='both')
axes[0].legend()

# Multivariate Plot
axes[1].scatter(y_actual, y_pred_multi, alpha=0.6, color='lightcoral')
axes[1].plot(plot_range, plot_range, color='red', linestyle='--', label='Ideal Prediction')
axes[1].set_title('Multivariate LSTM: Actual vs. Predicted Value')
axes[1].set_xlabel('Actual Value (€)')
axes[1].set_ylabel('Predicted Value (€)')
axes[1].ticklabel_format(style='plain', axis='both')
axes[1].legend()

fig.suptitle('Model Performance Evaluation', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- 3. Summary Table (Optional, for presentation) ---
summary_data = {
    'Metric': ['RMSE', 'MAE', 'R2 Score'],
    'Univariate Model': [f"€{uni_metrics['RMSE']:,.0f}", f"€{uni_metrics['MAE']:,.0f}", f"{uni_metrics['R2']:.4f}"],
    'Multivariate Model': [f"€{multi_metrics['RMSE']:,.0f}", f"€{multi_metrics['MAE']:,.0f}",
                           f"{multi_metrics['R2']:.4f}"]
}
summary_df = pd.DataFrame(summary_data)

print("\n--- Final Performance Summary ---")
print(summary_df.to_string(index=False))