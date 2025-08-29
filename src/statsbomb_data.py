import os
import json
import pandas as pd
from glob import glob

# Paths to data folders
BASE_PATH = r"C:\Users\asing\PycharmProjects\-Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data-\data\statsbomb\open-data"
OUTPUT_FOLDER = r"C:\Users\asing\PycharmProjects\-Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data-\processed_data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "player_performance.csv")

PATH_EVENTS = os.path.join(BASE_PATH, "events")
PATH_LINEUPS = os.path.join(BASE_PATH, "lineups")
PATH_MATCHES = os.path.join(BASE_PATH, "matches")
PATH_COMPETITION = os.path.join(BASE_PATH, "competitions.json")

# Load matches & competition
matches_data = []
for subfolder in glob(os.path.join(PATH_MATCHES, "*")):
    if os.path.isdir(subfolder):
        for file in glob(os.path.join(subfolder, "*.json")):
            with open(file, "r", encoding="utf-8") as f:
                matches_data.extend(json.load(f))

matches_df = pd.DataFrame([
    {
        "match_id": str(m.get("match_id")),  # ensure string
        "competition_id": m.get("competition", {}).get("competition_id"),
        "season": m.get("season", {}).get("season_name"),
        "home_team_id": m.get("home_team", {}).get("home_team_id"),
        "away_team_id": m.get("away_team", {}).get("away_team_id"),
        "date": m.get("match_date"),
    }
    for m in matches_data
])

# Process lineups → build player dictionary
lineups_list = []
player_lookup = {}

for file in glob(os.path.join(PATH_LINEUPS, "*.json")):
    with open(file, "r", encoding="utf-8") as f:
        lineups_data = json.load(f)

    match_id = os.path.splitext(os.path.basename(file))[0]  # from filename

    for team in lineups_data:
        team_id = team["team_id"]
        for player in team["lineup"]:
            player_id = player["player_id"]
            player_name = player.get("player_name", f"Unknown_{player_id}")

            # store mapping
            player_lookup[player_id] = player_name

            for pos in player.get("positions", []):
                lineups_list.append({
                    "player_id": player_id,
                    "player_name": player_name,  # also store here
                    "team_id": team_id,
                    "match_id": str(match_id),
                    "position": pos.get("position"),
                })

lineups_df = pd.DataFrame(lineups_list)

# Helper function
def safe_get(row, keys):
    """Safely navigate nested dicts."""
    for key in keys:
        if row is None or row.get(key) is None:
            return None
        row = row[key]
    return row

# Process events in chunks
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

first_chunk = True

for json_file in glob(os.path.join(PATH_EVENTS, "*.json")):
    match_id = os.path.splitext(os.path.basename(json_file))[0]  # from filename
    match_id = str(match_id)

    print(f"Processing {os.path.basename(json_file)} (match_id={match_id}) ...")

    with open(json_file, "r", encoding="utf-8") as f:
        events_data = json.load(f)

    events_list = []

    for e in events_data:
        player_id = safe_get(e, ["player", "id"])
        if player_id is None:
            continue  # skip non-player events

        player_name = player_lookup.get(player_id, f"Unknown_{player_id}")

        events_list.append({
            "player_name": player_name,
            "match_id": match_id,
            "event_type": safe_get(e, ["type", "name"]),
            "minute": e.get("minute"),
            "x": e.get("location", [None, None])[0] if e.get("location") else None,
            "y": e.get("location", [None, None])[1] if e.get("location") else None,
            "outcome": safe_get(e, ["shot", "outcome", "name"]) or safe_get(e, ["outcome", "name"]),
            "xG": safe_get(e, ["shot", "statsbomb_xg"]),
            "pass_type": safe_get(e, ["pass", "pass_type", "name"]),
            "pass_assisted_shot_id": safe_get(e, ["pass", "assisted_shot_id"]),
        })

    if not events_list:
        continue

    df_chunk = pd.DataFrame(events_list)
    df_chunk = df_chunk.merge(matches_df[["match_id", "season"]], on="match_id", how="left")

    # Define correct flags
    df_chunk["is_goal"] = (
        (df_chunk["event_type"] == "Shot") &
        (df_chunk["outcome"] == "Goal")
    )

    df_chunk["is_shot_on_target"] = (
        (df_chunk["event_type"] == "Shot") &
        (df_chunk["outcome"].isin(["Goal", "Saved", "Saved To Post"]))
    )

    df_chunk["is_assist"] = (
        (df_chunk["event_type"] == "Pass") &
        (df_chunk["pass_assisted_shot_id"].notnull())
    )

    # Aggregate per chunk → group by player_name
    agg_chunk = df_chunk.groupby(["player_name", "season"]).agg(
        total_goals=pd.NamedAgg(column="is_goal", aggfunc="sum"),
        total_assists=pd.NamedAgg(column="is_assist", aggfunc="sum"),
        shots_on_target=pd.NamedAgg(column="is_shot_on_target", aggfunc="sum"),
        xG=pd.NamedAgg(column="xG", aggfunc="sum"),
        matches_played=pd.NamedAgg(column="match_id", aggfunc=pd.Series.nunique),
        minutes_played=pd.NamedAgg(column="minute", aggfunc="max"),
    ).reset_index()

    # Merge position distribution
    if not lineups_df.empty:
        pos_distribution = lineups_df.groupby(["player_id", "position"]).size().unstack(fill_value=0)

        # Map player_id → player_name
        pos_distribution = pos_distribution.rename(index=player_lookup)

        agg_chunk = agg_chunk.merge(pos_distribution, left_on="player_name", right_index=True, how="left")

    # Append to CSV
    if first_chunk:
        agg_chunk.to_csv(OUTPUT_CSV, index=False, mode="w")
        first_chunk = False
    else:
        agg_chunk.to_csv(OUTPUT_CSV, index=False, mode="a", header=False)

print(f"player_performance.csv saved successfully at {OUTPUT_CSV}")