# --- IMPORTS ---
from unidecode import unidecode
import pandas as pd
import os
import numpy as np # Import numpy to use np.nan for null values

# --- STEP 1: LOAD DATA ---
print("--- Step 1: Loading Data ---")
data_folder = 'data'
performance_file_path = os.path.join(data_folder, 'La_Liga_2015-2016_all_events.csv')
market_value_file = os.path.join(data_folder, 'laliga_2015_market_values.csv')

performance_df = pd.read_csv(performance_file_path, low_memory=False)
market_value_df = pd.read_csv(market_value_file)
print("Successfully loaded data from the 'data' subfolder!")

# --- STEP 2: AGGREGATE PERFORMANCE DATA (WITH DEFENSIVE CHECKS) ---
print("\n--- Step 2: Aggregating Performance Data ---")

# Always calculate goals and assists as they are critical
goals_df = performance_df[performance_df['shot_outcome'] == 'Goal']
player_goals = goals_df.groupby('player').size().reset_index(name='goals')
assists_df = performance_df[performance_df['pass_assisted_shot_id'].notna()]
player_assists = assists_df.groupby('player').size().reset_index(name='assists')

# --- DEFENSIVE CHECK for Successful Passes ---
if 'pass_outcome' in performance_df.columns:
    successful_passes_df = performance_df[performance_df['pass_outcome'].isna()]
    player_passes = successful_passes_df.groupby('player').size().reset_index(name='successful_passes')
else:
    print("Warning: 'pass_outcome' column not found. 'successful_passes' will be set to 0.")
    player_passes = pd.DataFrame(columns=['player', 'successful_passes'])

# --- DEFENSIVE CHECK for Tackles Won ---
if 'type' in performance_df.columns and 'duel_outcome' in performance_df.columns:
    tackles_won_df = performance_df[(performance_df['type'] == 'Duel') & (performance_df['duel_outcome'] == 'Won')]
    player_tackles = tackles_won_df.groupby('player').size().reset_index(name='tackles_won')
else:
    print("Warning: Columns for calculating tackles not found. 'tackles_won' will be set to 0.")
    player_tackles = pd.DataFrame(columns=['player', 'tackles_won'])

# --- DEFENSIVE CHECK for Primary Position ---
if 'position' in performance_df.columns:
    player_positions = performance_df.groupby('player')['position'].agg(lambda x: x.mode().iloc[0]).reset_index(name='position')
else:
    print("Warning: 'position' column not found. 'position' will be set to null.")
    player_positions = pd.DataFrame(columns=['player', 'position'])

# Combine all aggregated stats
performance_summary_df = pd.merge(player_goals, player_assists, on='player', how='outer')
performance_summary_df = pd.merge(performance_summary_df, player_passes, on='player', how='outer')
performance_summary_df = pd.merge(performance_summary_df, player_tackles, on='player', how='outer')
performance_summary_df = pd.merge(performance_summary_df, player_positions, on='player', how='outer')

# Fill any missing numeric stats with 0
numeric_cols = ['goals', 'assists', 'successful_passes', 'tackles_won']
for col in numeric_cols:
    if col in performance_summary_df.columns:
        performance_summary_df[col] = performance_summary_df[col].fillna(0)
        performance_summary_df[col] = performance_summary_df[col].astype(int)

print("Performance data aggregated successfully.")

# --- STEP 3: STANDARDIZE NAMES FOR MERGE ---
print("\n--- Step 3: Standardizing Player Names ---")
def clean_player_name(name):
    clean_name = unidecode(str(name).lower())
    parts = clean_name.split()
    return ' '.join(parts[:2])

performance_summary_df['merge_key'] = performance_summary_df['player'].apply(clean_player_name)
market_value_df['merge_key'] = market_value_df['Player Name'].apply(clean_player_name)
print("Created a standardized 'merge_key' in both DataFrames.")

# --- STEP 4: MERGE DATAFRAMES ---
print("\n--- Step 4: Merging DataFrames ---")
final_df = pd.merge(performance_summary_df, market_value_df, on='merge_key', how='right')
# Fill missing performance data that resulted from the right merge
final_df['goals'] = final_df['goals'].fillna(0)
final_df['assists'] = final_df['assists'].fillna(0)
print("DataFrames merged successfully.")

# --- STEP 5: CLEAN UP AND ENSURE ALL COLUMNS EXIST (CORRECTED) ---
print("\n--- Step 5: Cleaning Up and Finalizing Columns ---")
# --- DEFENSIVE CHECK for all required columns before final selection ---
# For any column that might be missing, we create it here with a default value.
if 'successful_passes' not in final_df.columns:
    final_df['successful_passes'] = 0
if 'tackles_won' not in final_df.columns:
    final_df['tackles_won'] = 0
if 'position' not in final_df.columns:
    final_df['position'] = np.nan # Use np.nan for null
if 'Nationality' not in final_df.columns:
    final_df['Nationality'] = np.nan # Use np.nan for null

# Now we can safely clean up and select our final columns
final_df.drop(columns=['player', 'merge_key'], inplace=True)
final_df.rename(columns={'Player Name': 'player_name'}, inplace=True)

# The final column selection is now guaranteed to work
final_df = final_df[[
    'player_name',
    'position',
    'Nationality',
    'goals',
    'assists',
    'successful_passes',
    'tackles_won',
    'Market Value 2015 (in millions €)'
]]
print("Cleanup successful and all feature columns are present.")

# --- STEP 6: FILTER FOR TOP 10 PLAYERS ---
print("\n--- Step 6: Filtering for Top 10 Players ---")
top_10_players_list = [
    'lionel messi', 'cristiano ronaldo', 'neymar', 'luis suarez',
    'gareth bale', 'james rodriguez', 'sergio busquets',
    'karim benzema', 'luka modric', 'toni kroos'
]
clean_func = lambda name: unidecode(str(name).lower())
mask = final_df['player_name'].apply(clean_func).isin(top_10_players_list)
top_10_df = final_df[mask].copy()
print(f"Successfully found {len(top_10_df)} players after filtering.")

# --- STEP 7: SAVE THE FINAL DATAFRAME ---
print("\n--- Step 7: Saving the Final CSV File ---")
output_path = os.path.join(data_folder, 'final_top_10_player_data.csv')
top_10_df.to_csv(output_path, index=False)
print(f"DataFrame successfully saved to: {output_path}")

print("\n--- Final Data Preview ---")
print(top_10_df)