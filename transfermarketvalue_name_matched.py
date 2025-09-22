import pandas as pd
import difflib

# Load CSV files
players_df = pd.read_csv("players.csv")
fifa_df = pd.read_csv("fifa_players_cleaned.csv")

# Take only the names that exist in fifa_player_cleaned
# This avoids matching against "extra" players from players.csv
valid_names = players_df[players_df["name"].isin(players_df["name"])]["name"].tolist()
# Actually just take all players.csv names, difflib will only use them for matching

# Function to match full_name with closest short name
def match_name(fifa_fullname, choices):
    best_match = difflib.get_close_matches(fifa_fullname, choices, n=1, cutoff=0.5)
    return best_match[0] if best_match else None   # return None if no good match

# Match each fifa full_name to closest players.csv name
fifa_df["short_name"] = fifa_df["full_name"].apply(lambda x: match_name(x, players_df["name"].tolist()))

# Drop rows where no match was found (optional)
fifa_df = fifa_df.dropna(subset=["short_name"])

# Save final file
fifa_df.to_csv("fifa_player_final.csv", index=False)

print("✅ New file saved as fifa_player_final.csv")
