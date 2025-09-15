import pandas as pd
import numpy as np
import ast

# ===========================
# 1. Load & Clean Data
# ===========================
players_df = pd.read_csv(r"D:\Pythonproject\datasets\model_training/epl_players_basic_info_cleaned.csv")
injuries_df = pd.read_csv(r"D:\Pythonproject\datasets\model_training/players_injuries_cleaned.csv")
competitions_df = pd.read_csv(r"D:\Pythonproject\stats_bomb\data\csv_output/competitions.csv")
lineups_df = pd.read_csv(r"D:\Pythonproject\stats_bomb\data\csv_output/lineups.csv")
matches_df = pd.read_csv(r"D:\Pythonproject\stats_bomb\data\csv_output/matches.csv")

for df in [players_df, injuries_df, competitions_df, lineups_df, matches_df]:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("%", "pct")

# ===========================
# 2. Parse Lineups
# ===========================
parsed_rows = []
for _, row in lineups_df.iterrows():
    try:
        lineup_data = ast.literal_eval(row['lineup']) if isinstance(row['lineup'], str) else row['lineup']
        for player in lineup_data:
            parsed_rows.append({
                "match_id": player.get("match_id", row.get("match_id")),
                "player_id": player.get("player_id"),
                "player_name": player.get("player_name"),
                "minutes_played": player.get("minutes_played", 0),
                "goals": player.get("goals", 0),
                "assists": player.get("assists", 0),
                "shots": player.get("shots", 0),
                "shots_on_target": player.get("shots_on_target", 0),
                "tackles": player.get("tackles", 0),
                "interceptions": player.get("interceptions", 0),
                "pressures": player.get("pressures", 0),
                "blocks": player.get("blocks", 0),
                "touches": player.get("touches", 0),
                "passes_completed": player.get("passes_completed", 0),
                "passes_attempted": player.get("passes", 0)
            })
    except Exception as e:
        print(f"⚠️ Could not parse lineup row: {e}")

parsed_lineups = pd.DataFrame(parsed_rows)
parsed_lineups.dropna(subset=["match_id", "player_id"], inplace=True)

# ===========================
# 3. Merge with Matches & Filter PL
# ===========================
merged_df = pd.merge(parsed_lineups, matches_df, on="match_id", how="left")
pl_matches = merged_df[merged_df["competition_competition_name"].str.contains("Premier League", case=False, na=False)]

# Create 90s column
pl_matches["90s"] = pl_matches["minutes_played"] / 90

# ===========================
# 4. Per-90 Features
# ===========================
per90_features = {
    "goals": "goals_per90",
    "assists": "assists_per90",
    "shots": "shots_per90",
    "shots_on_target": "shots_on_target_per90",
    "tackles": "tackles_per90",
    "interceptions": "interceptions_per90",
    "pressures": "pressures_per90",
    "blocks": "blocks_per90",
    "touches": "touches_per90",
    "passes_completed": "passes_completed_per90",
    "passes_attempted": "passes_attempted_per90"
}

for col, new_col in per90_features.items():
    if col in pl_matches.columns:
        pl_matches[new_col] = pl_matches[col] / pl_matches["90s"].replace(0, np.nan)

# ===========================
# 5. Composite Features
# ===========================
pl_matches["goal_contrib_per90"] = (pl_matches["goals"] + pl_matches["assists"]) / pl_matches["90s"].replace(0, np.nan)
pl_matches["shot_accuracy"] = pl_matches["shots_on_target"] / pl_matches["shots"].replace(0, np.nan)
pl_matches["pass_accuracy"] = pl_matches["passes_completed"] / pl_matches["passes_attempted"].replace(0, np.nan)
pl_matches["defensive_index"] = (pl_matches["tackles"] + pl_matches["interceptions"] + pl_matches["blocks"]) / pl_matches["90s"].replace(0, np.nan)

# ===========================
# 6. Aggregate Per Player Per Season
# ===========================
player_season_features = pl_matches.groupby(["player_id", "season_season_name"]).agg({
    "goals_per90": "mean",
    "assists_per90": "mean",
    "goal_contrib_per90": "mean",
    "shots_per90": "mean",
    "shot_accuracy": "mean",
    "tackles_per90": "mean",
    "interceptions_per90": "mean",
    "pressures_per90": "mean",
    "blocks_per90": "mean",
    "defensive_index": "mean",
    "pass_accuracy": "mean",
    "minutes_played": "sum",
    "match_id": "count"
}).reset_index()

player_season_features.rename(columns={"match_id": "matches_played"}, inplace=True)

# ===========================
# 7. Merge Player Info
# ===========================
features_df = pd.merge(players_df, player_season_features, on="player_id", how="left")

# ===========================
# 8. Merge Injuries Per Season
# ===========================
if "season" in injuries_df.columns:
    injury_features = injuries_df.groupby(["player_name", "season"]).agg(
        total_injuries=("injury_type", "count"),
        total_days_missed=("days", "sum"),
        avg_days_per_injury=("days", "mean")
    ).reset_index()
    features_df = pd.merge(
        features_df,
        injury_features,
        left_on=["player_name", "season_season_name"],
        right_on=["player_name", "season"],
        how="left"
    )
else:
    injury_features = injuries_df.groupby("player_name").agg(
        total_injuries=("injury_type", "count"),
        total_days_missed=("days", "sum"),
        avg_days_per_injury=("days", "mean")
    ).reset_index()
    features_df = pd.merge(features_df, injury_features, on="player_name", how="left")

features_df.fillna({
    "matches_played": 0,
    "minutes_played": 0,
    "total_injuries": 0,
    "total_days_missed": 0,
    "avg_days_per_injury": 0
}, inplace=True)

# ===========================
# 9. Position Encoding & Age Features
# ===========================
def map_position(pos):
    if not isinstance(pos, str):
        return "Other"
    pos = pos.upper()
    if pos.startswith("F") or pos in ["RW", "LW", "CF", "ST"]:
        return "Attacker"
    elif pos.startswith("M"):
        return "Midfielder"
    elif pos.startswith("D"):
        return "Defender"
    elif pos in ["GK", "G"]:
        return "Goalkeeper"
    else:
        return "Other"

if "position" in features_df.columns:
    features_df["position_group"] = features_df["position"].apply(map_position)
    features_df = pd.get_dummies(features_df, columns=["position_group"], prefix="pos")

if "age" in features_df.columns:
    features_df["age_squared"] = features_df["age"] ** 2
    features_df["age_bucket"] = pd.cut(
        features_df["age"],
        bins=[15, 20, 24, 28, 32, 36, 40, 50],
        labels=["<20", "20-24", "25-28", "29-32", "33-36", "37-40", "40+"]
    )

# ===========================
# 10. Save Final Dataset
# ===========================
output_file = r"D:\Pythonproject\datasets\model_training\player_features_engineered_PL_seasonwise.csv"
features_df.to_csv(output_file, index=False)
print(f"✅ Feature engineering completed (Premier League, per season). Saved to {output_file}, shape={features_df.shape}")
