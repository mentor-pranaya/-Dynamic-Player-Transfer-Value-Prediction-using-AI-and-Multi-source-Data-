import os
import pandas as pd
from functools import reduce

# Paths
base_path = os.path.expanduser("~/football-data-analysis")
data_path = os.path.join(base_path, "data")

players_file = os.path.join(data_path, "Complete_players_list_unique.csv")
sentiment_file = os.path.join(data_path, "sentiment_report.csv")
injury_path = os.path.join(data_path, "week1_injury")

# Load players + sentiment
players = pd.read_csv(players_file)
sentiment = pd.read_csv(sentiment_file)

# Merge all injury CSVs
injury_files = [os.path.join(injury_path, f) for f in os.listdir(injury_path) if f.endswith(".csv")]
injury_dfs = [pd.read_csv(file) for file in injury_files]
injury_merged = pd.concat(injury_dfs, ignore_index=True)

# --- Align keys ---
# Players & Sentiment share "Player"
# Injury does NOT → we add Player = "Unknown" (for now)
injury_merged["Player"] = "Unknown"

# Merge everything
merged = reduce(
    lambda left, right: pd.merge(left, right, on="Player", how="outer"),
    [players, sentiment, injury_merged]
)

# Save output
output_file = os.path.join(data_path, "week1_final.csv")
merged.to_csv(output_file, index=False)

print(f" Week1 data merged into {output_file}")
