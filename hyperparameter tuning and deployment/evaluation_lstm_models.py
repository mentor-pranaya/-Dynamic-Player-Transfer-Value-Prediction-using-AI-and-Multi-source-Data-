import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------------
# 1️⃣ Load and Prepare Data
# -----------------------------------------------------------
print("🔹 Loading model prediction results...")

data_path = r"C:\Users\Abhinav\TransferIQ\data\lstm_prediction_results.csv"
results = pd.read_csv(data_path, encoding='utf-8')

# Helper function to safely convert Euro values back to floats
def convert_euro_to_float(euro_str):
    """Remove euro symbol and commas from market value strings."""
    try:
        return float(str(euro_str).replace("€", "").replace(",", "").strip())
    except:
        return np.nan

# Convert string-formatted values back to numeric
y_actual = results['Actual_Value'].apply(convert_euro_to_float).to_numpy().reshape(-1, 1)
y_pred_uni = results['Uni_Predicted_Value'].apply(convert_euro_to_float).to_numpy().reshape(-1, 1)
y_pred_multi = results['Multi_Predicted_Value'].apply(convert_euro_to_float).to_numpy().reshape(-1, 1)

print(f"✅ Data loaded successfully — {len(y_actual)} samples found.\n")

# -----------------------------------------------------------
# 2️⃣ Define Metric Computation
# -----------------------------------------------------------
def get_model_scores(true_vals, pred_vals, model_title):
    """Calculate and print evaluation metrics."""
    rmse_val = np.sqrt(mean_squared_error(true_vals, pred_vals))
    mae_val = mean_absolute_error(true_vals, pred_vals)
    r2_val = r2_score(true_vals, pred_vals)

    print(f"📊 {model_title} Evaluation:")
    print(f"   • RMSE : €{rmse_val:,.2f}")
    print(f"   • MAE  : €{mae_val:,.2f}")
    print(f"   • R²   :  {r2_val:.4f}")
    return {"RMSE": rmse_val, "MAE": mae_val, "R2": r2_val}

# Compute metrics for both models
metrics_uni = get_model_scores(y_actual, y_pred_uni, "Univariate LSTM (Single Feature)")
metrics_multi = get_model_scores(y_actual, y_pred_multi, "Multivariate LSTM (Multi-Feature)")

# -----------------------------------------------------------
# 3️⃣ Visualization — Actual vs Predicted Scatter Plots
# -----------------------------------------------------------
sns.set_style("whitegrid")

fig, axarr = plt.subplots(1, 2, figsize=(14, 6))
plt.suptitle("⚽ LSTM Market Value Prediction Comparison", fontsize=16, weight="bold")

# Determine plotting bounds dynamically
plot_max = max(y_actual.max(), y_pred_multi.max(), y_pred_uni.max()) * 1.05
lims = [0, plot_max]

# Univariate plot
axarr[0].scatter(y_actual, y_pred_uni, alpha=0.6, color="#3fa7d6", edgecolor='k')
axarr[0].plot(lims, lims, 'r--', lw=1.5, label="Ideal Prediction")
axarr[0].set_title("Univariate LSTM")
axarr[0].set_xlabel("Actual Value (€)")
axarr[0].set_ylabel("Predicted Value (€)")
axarr[0].legend()

# Multivariate plot
axarr[1].scatter(y_actual, y_pred_multi, alpha=0.6, color="#e86a33", edgecolor='k')
axarr[1].plot(lims, lims, 'r--', lw=1.5, label="Ideal Prediction")
axarr[1].set_title("Multivariate LSTM")
axarr[1].set_xlabel("Actual Value (€)")
axarr[1].set_ylabel("Predicted Value (€)")
axarr[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# -----------------------------------------------------------
# 4️⃣ Summary Table
# -----------------------------------------------------------
summary = pd.DataFrame({
    "Metric": ["RMSE (€)", "MAE (€)", "R² Score"],
    "Univariate LSTM": [
        f"{metrics_uni['RMSE']:,.2f}",
        f"{metrics_uni['MAE']:,.2f}",
        f"{metrics_uni['R2']:.4f}"
    ],
    "Multivariate LSTM": [
        f"{metrics_multi['RMSE']:,.2f}",
        f"{metrics_multi['MAE']:,.2f}",
        f"{metrics_multi['R2']:.4f}"
    ]
})

print("\n📘 Final Performance Summary:")
print(summary.to_string(index=False))

# Optional: save results
save_path = r"C:\Users\Abhinav\TransferIQ\results\model_comparison_summary.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
summary.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\n📂 Summary saved to: {save_path}")
