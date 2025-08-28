import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('epl_players.csv')

# Remove duplicate rows based on unique player identifier, e.g. 'player_id'
df = df.drop_duplicates(subset=['player_id'])

# Optionally, if duplicates differ by club or other fields, keep the row with max market_value or latest info:
# df = df.sort_values('market_value_eur', ascending=False).drop_duplicates(subset=['player_id'], keep='first')

# Handle missing values:

# For numeric columns like market_value_eur, fill missing with 0 or np.nan (choose what's best)
df['market_value_eur'] = pd.to_numeric(df['market_value_eur'], errors='coerce')
df['market_value_eur'] = df['market_value_eur'].fillna(0)

# For text columns like market_value_text, fill missing with empty string or 'Unknown'
df['market_value_text'] = df['market_value_text'].fillna('Unknown')

# For other columns such as position, age, nationality, fill as appropriate, e.g.:
df['position'] = df['position'].fillna('Unknown')
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(0)

# Save cleaned CSV
df.to_csv('epl_players_cleaned.csv', index=False)

print("Cleaned file saved as epl_players_cleaned.csv")
