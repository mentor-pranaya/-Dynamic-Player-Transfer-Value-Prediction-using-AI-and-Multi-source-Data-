

import pandas as pd
import numpy as np
import os

INPUT_PATH = "/Users/ghans/OneDrive/Desktop/filemanaging/master_list_cleaned.csv"
OUTPUT_FOLDER = "/Users/ghans/OneDrive/Desktop/filemanaging"
OUTPUT_FILE = "master_list_final_features.csv"

try:
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset loaded successfully from: {INPUT_PATH}")
except FileNotFoundError:
    print(f"Error: File not found at {INPUT_PATH}")
    exit()

print(f"Initial dataset shape: {df.shape}\n")

df = df.fillna({
    'age': df['age'].mean() if 'age' in df.columns else 25,
    'overall_rating': df['overall_rating'].mean() if 'overall_rating' in df.columns else 70,
    'potential': df['potential'].mean() if 'potential' in df.columns else 75,
    'sentiment_score': 0,
    'total_injuries': 0,
    'total_days_missed': 0,
})

# Ensure numeric conversions
numeric_cols = ['market_value_in_eur', 'overall_rating', 'potential', 'sentiment_score', 
                'total_injuries', 'total_days_missed', 'age']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print("Creating performance-related features...")

# Trend-like metric showing growth from potential and rating
if {'overall_rating', 'potential'}.issubset(df.columns):
    df['growth_potential'] = (df['potential'] - df['overall_rating']).round(2)

# Combine match stats (if they exist)
if {'goals', 'assists'}.issubset(df.columns):
    df['goal_contribution'] = df['goals'] + df['assists']
else:
    df['goal_contribution'] = 0

# Performance index – weighted formula
df['performance_index'] = (
    0.4 * df['overall_rating'] +
    0.3 * df['potential'] +
    0.2 * df['goal_contribution'] +
    0.1 * (100 - df['total_days_missed'].clip(0, 200))
)

print("✅ Performance metrics added.\n")


print("Adding sentiment-based features...")

if 'sentiment_score' in df.columns:
    # Weighted sentiment impact based on recent performance
    df['sentiment_impact'] = (
        df['sentiment_score'] * (df['performance_index'] / 100)
    ).round(3)
else:
    df['sentiment_impact'] = 0


df['player_influence_score'] = (
    0.6 * df['performance_index'] + 
    0.4 * (100 * df['sentiment_impact'])
)

print("✅ Sentiment and influence features added.\n")

print("Computing injury and age-related factors...")

df['injury_severity'] = np.where(
    df['total_injuries'] > 0,
    df['total_days_missed'] / df['total_injuries'],
    0
).round(2)

df['age_factor'] = np.exp(-0.05 * (df['age'] - 27).abs())

df['fitness_index'] = (
    0.7 * df['age_factor'] + 
    0.3 * (1 / (1 + df['injury_severity']))
).round(3)

print("✅ Injury and age features added.\n")

print("Calculating final player value predictor feature...")

df['value_prediction_index'] = (
    0.45 * df['performance_index'] +
    0.25 * df['player_influence_score'] +
    0.15 * df['fitness_index'] +
    0.15 * df['growth_potential']
).round(3)

print("✅ Final value prediction index computed.\n")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

df.to_csv(output_path, index=False)
print(f"Feature-engineered dataset saved successfully at:\n{output_path}\n")

print("Final dataset shape:", df.shape)
print("Feature engineering completed ✅")
