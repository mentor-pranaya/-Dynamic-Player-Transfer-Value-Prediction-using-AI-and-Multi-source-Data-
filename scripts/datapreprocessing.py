import os
import pandas as pd
import numpy as np
from unidecode import unidecode

print("=== Player Data Preprocessing Pipeline ===")
BASE_DIR = r"C:\Users\Abhinav\Desktop\Project\data"
PERF_FILE = os.path.join(BASE_DIR, "La_Liga_2015-2016_all_events.csv")
VALUE_FILE = os.path.join(BASE_DIR, "laliga_2015_market_values.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "final_top10_players.csv")

print("[1/7] Loading datasets...")
perf_df = pd.read_csv(PERF_FILE, low_memory=False)
values_df = pd.read_csv(VALUE_FILE)
print(f"✔ Loaded performance ({len(perf_df):,} rows) and market value ({len(values_df):,} rows) data")

print("[2/7] Aggregating performance metrics...")

# Goals
goals = perf_df.loc[perf_df.get("shot_outcome") == "Goal"].groupby("player").size().reset_index(name="goals")

# Assists
assists = perf_df.loc[perf_df["pass_assisted_shot_id"].notna()].groupby("player").size().reset_index(name="assists")

# Successful passes
if "pass_outcome" in perf_df.columns:
    succ_pass = perf_df.loc[perf_df["pass_outcome"].isna()].groupby("player").size().reset_index(name="succ_passes")
else:
    print("⚠ 'pass_outcome' missing → defaulting passes to 0")
    succ_pass = pd.DataFrame(columns=["player", "succ_passes"])

# Tackles won
if {"type", "duel_outcome"}.issubset(perf_df.columns):
    tackles = perf_df.loc[(perf_df["type"] == "Duel") & (perf_df["duel_outcome"] == "Won")] \
                     .groupby("player").size().reset_index(name="tackles_won")
else:
    print("⚠ Duel columns missing → defaulting tackles to 0")
    tackles = pd.DataFrame(columns=["player", "tackles_won"])

# Player position (mode)
if "position" in perf_df.columns:
    positions = perf_df.groupby("player")["position"].agg(lambda x: x.mode().iloc[0]).reset_index(name="position")
else:
    positions = pd.DataFrame(columns=["player", "position"])

# Merge all aggregated stats
summary = goals.merge(assists, on="player", how="outer") \
               .merge(succ_pass, on="player", how="outer") \
               .merge(tackles, on="player", how="outer") \
               .merge(positions, on="player", how="outer")

for col in ["goals", "assists", "succ_passes", "tackles_won"]:
    summary[col] = summary[col].fillna(0).astype(int)

print("✔ Aggregation complete")

print("[3/7] Standardizing player names for merging...")

def normalize_name(name):
    name = unidecode(str(name).lower())
    parts = name.split()
    return " ".join(parts[:2])  # use first two parts to reduce mismatch

summary["merge_key"] = summary["player"].map(normalize_name)
values_df["merge_key"] = values_df["Player Name"].map(normalize_name)

print("[4/7] Merging datasets...")
merged = pd.merge(summary, values_df, on="merge_key", how="right")
merged["goals"] = merged["goals"].fillna(0).astype(int)
merged["assists"] = merged["assists"].fillna(0).astype(int)

# Ensure required cols exist
if "succ_passes" not in merged: merged["succ_passes"] = 0
if "tackles_won" not in merged: merged["tackles_won"] = 0
if "position" not in merged: merged["position"] = np.nan
if "Nationality" not in merged: merged["Nationality"] = np.nan

# Drop helper cols and rename
merged.drop(columns=["player", "merge_key"], inplace=True, errors="ignore")
merged.rename(columns={"Player Name": "player_name"}, inplace=True)

final = merged[[
    "player_name", "position", "Nationality",
    "goals", "assists", "succ_passes", "tackles_won",
    "Market Value 2015 (in millions €)"
]]
print("✔ Final dataset structured")

print("[5/7] Extracting top 10 stars...")
top10_names = [
    "lionel messi", "cristiano ronaldo", "neymar", "luis suarez",
    "gareth bale", "james rodriguez", "sergio busquets",
    "karim benzema", "luka modric", "toni kroos"
]
mask = final["player_name"].map(lambda n: unidecode(str(n).lower())).isin(top10_names)
top10 = final[mask].copy()
print(f"✔ Selected {len(top10)} players")

print("[6/7] Saving processed file...")
top10.to_csv(OUTPUT_FILE, index=False)
print(f"✔ Final CSV saved to {OUTPUT_FILE}")

print("\n=== Preview of Final Data ===")
print(top10.head())
