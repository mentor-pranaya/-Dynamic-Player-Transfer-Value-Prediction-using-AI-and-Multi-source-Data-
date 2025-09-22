# =======================================
# 1. Load Player Features
# =======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector

# --- Connect to DB ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yahoonet",
    database="AIProject"
)

df = pd.read_sql("SELECT * FROM player_features", db)
db.close()

print("✅ Loaded player_features shape:", df.shape)
df.head()

# =======================================
# 2. Basic EDA
# =======================================
# Missing values
print(df.isnull().sum())

# Histograms of key features
num_cols = ["latest_market_value", "market_value_growth",
            "total_injuries", "avg_days_out",
            "total_transfers", "total_transfer_fees",
            "sentiment_mean", "sentiment_positive_ratio",
            "sentiment_trend", "avg_cards_per_match"]
df[num_cols].hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig('feature_histograms.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations")
plt.savefig('feature_correlation_heatmap.png')
plt.show()
