import pandas as pd
import numpy as np

# Load the dataset from the specified file path.

try:
    df = pd.read_csv('/content/Injury_data.csv', na_values=[''])
except FileNotFoundError:
    print("Error: The file 'Injury_data.csv' was not found.")
    print("Please make sure the file is in the correct directory.")
    exit()

print("Initial data information:")
df.info()
print("\nFirst 5 rows of the raw data:")
print(df.head())
print("-" * 50)

# --- Data Cleaning and Preprocessing ---

# 1. Handle missing values.

df['cumulative_minutes_played'] = df['cumulative_minutes_played'].fillna(0)
df['cumulative_games_played'] = df['cumulative_games_played'].fillna(0)
df['minutes_per_game_prev_seasons'] = df['minutes_per_game_prev_seasons'].fillna(0)
df['avg_days_injured_prev_seasons'] = df['avg_days_injured_prev_seasons'].fillna(0)
df['avg_games_per_season_prev_seasons'] = df['avg_games_per_season_prev_seasons'].fillna(0)
df['significant_injury_prev_season'] = df['significant_injury_prev_season'].fillna(0)
df['cumulative_days_injured'] = df['cumulative_days_injured'].fillna(0)
df['season_days_injured_prev_season'] = df['season_days_injured_prev_season'].fillna(0)

# 2. Convert 'dob' (date of birth) to datetime objects for accurate age calculation.
df['dob'] = pd.to_datetime(df['dob'])

# 3. Create a more accurate 'age_at_start_of_season' column.

df['age_at_start_of_season'] = (df['start_year'] - df['dob'].dt.year).astype(int)

# 4. Convert categorical variables to numerical representations.

work_rate_mapping = {
    'High/High': 3.0, 'High/Medium': 2.5, 'High/Low': 2.0,
    'Medium/High': 2.5, 'Medium/Medium': 2.0, 'Medium/Low': 1.5,
    'Low/High': 1.5, 'Low/Medium': 1.0, 'Low/Low': 0.5
}
df['work_rate_numeric'] = df['work_rate'].map(work_rate_mapping)

position_mapping = {
    'Forward': 1, 'Midfielder': 2, 'Defender': 3, 'Goalkeeper': 4,
    'RW': 1, 'LW': 1, 'ST': 1, 'CF': 1, 'CAM': 2, 'CDM': 2, 'CM': 2,
    'LM': 2, 'RM': 2, 'CB': 3, 'LB': 3, 'RB': 3, 'GK': 4, 'RWB': 3, 'LWB': 3
}
df['position_numeric'] = df['position'].map(position_mapping)

# Check for any remaining non-numeric values that couldn't be mapped

df['work_rate_numeric'].fillna(df['work_rate_numeric'].mode()[0], inplace=True)
df['position_numeric'].fillna(df['position_numeric'].mode()[0], inplace=True)

# 5. Drop original categorical columns and unnecessary columns.

df.drop(columns=['work_rate', 'position', 'dob', 'age'], inplace=True)


print("\nProcessed data information:")
df.info()
print("\nFirst 5 rows of the preprocessed data:")
print(df.head())
print("-" * 50)

