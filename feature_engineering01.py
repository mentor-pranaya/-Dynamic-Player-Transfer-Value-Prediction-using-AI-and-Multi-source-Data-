import pandas as pd
import os
from unidecode import unidecode

# --- Load Your Datasets ---
data_folder = 'data'

# Load your main dataset that already includes sentiment scores
main_df = pd.read_csv(os.path.join(data_folder, 'final_top_10_player_data.csv'))

# The Fix: Add the encoding='latin-1' parameter to the read_csv function
injury_df = pd.read_csv(
    os.path.join(data_folder, 'injury_history.csv'),
    encoding='latin-1'
)

print("Successfully loaded injury_df using 'latin-1' encoding!")

# Re-use the name cleaning function for consistent merging
def clean_player_name(name):
    clean_name = unidecode(str(name).lower())
    parts = clean_name.split()
    return ' '.join(parts[:2])

print("--- Data loaded successfully ---")
print("Main DataFrame head:")
print(main_df.head())
print("\nInjury DataFrame head:")
print(injury_df.head())

# --- Calculate Injury Metrics ---
print("\n--- Calculating injury metrics per player ---")

# Group by player name and aggregate the data
injury_features = injury_df.groupby('player_name').agg(
    total_days_injured=('days_missed', 'sum'),
    injury_count=('player_name', 'count')
).reset_index()

print("Calculated injury metrics:")
print(injury_features.head())

# --- Merge Injury Features ---
print("\n--- Merging injury features into main DataFrame ---")

# Create a clean merge key on both dataframes for a reliable match
main_df['merge_key'] = main_df['player_name'].apply(clean_player_name)
injury_features['merge_key'] = injury_features['player_name'].apply(clean_player_name)

# Perform a left merge to add the injury data
main_df = pd.merge(main_df, injury_features[['merge_key', 'total_days_injured', 'injury_count']], on='merge_key', how='left')

# For players with no injuries, the merge creates NaN. We must fill these with 0.
main_df['total_days_injured'].fillna(0, inplace=True)
main_df['injury_count'].fillna(0, inplace=True)

# Clean up by dropping the temporary merge key
main_df.drop('merge_key', axis=1, inplace=True)

print("Injury data added successfully:")
print(main_df[['player_name', 'total_days_injured', 'injury_count']])

# --- Save DataFrame with Injury Features ---

# Define the path to your main CSV file
output_path = os.path.join(data_folder, 'final_top_10_player_data.csv')

# Save the updated DataFrame, overwriting the old file
# index=False prevents adding an extra, unnamed column to your CSV
main_df.to_csv(output_path, index=False)

print(f"\nSuccessfully updated '{output_path}' with the new injury data columns.")