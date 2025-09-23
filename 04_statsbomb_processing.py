import pandas as pd
import json
import os

STATS_BOMB_DATA_PATH = "open-data-master/data/"
COMPETITION_ID = "37"
SEASON_ID = "90"

matches_file = os.path.join(STATS_BOMB_DATA_PATH, f"matches/{COMPETITION_ID}/{SEASON_ID}.json")

try:
    with open(matches_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    match_ids = [match['match_id'] for match in matches]
    print(f"Found {len(match_ids)} matches for the season.")

    all_player_stats = []
    processed_count = 0

    for match_id in match_ids:
        events_file = os.path.join(STATS_BOMB_DATA_PATH, f"events/{match_id}.json")
        
        with open(events_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        df_events = pd.json_normalize(events, sep='_')

        player_events = df_events[df_events['player_name'].notna()].copy()
        
        if player_events.empty:
            continue

        match_date = pd.to_datetime(player_events['timestamp'].iloc[0]).date()

        # --- THE FIX: Build the aggregation dynamically ---
        # Start with stats that are always present
        agg_dict = {
            'shots': ('type_name', lambda x: (x == 'Shot').sum()),
            'goals': ('shot_outcome_name', lambda x: (x == 'Goal').sum())
        }

        # Conditionally add stats for columns that might be missing
        if 'pass_goal_assist' in player_events.columns:
            agg_dict['assists'] = ('pass_goal_assist', 'count')
        
        if 'pass_shot_assist' in player_events.columns:
            agg_dict['key_passes'] = ('pass_shot_assist', 'count')

        player_stats = player_events.groupby('player_name').agg(**agg_dict).reset_index()
        
        # --- Ensure all columns exist after aggregation, filling missing ones with 0 ---
        for col in ['assists', 'key_passes']:
            if col not in player_stats.columns:
                player_stats[col] = 0

        player_stats['match_id'] = match_id
        player_stats['match_date'] = match_date
        
        all_player_stats.append(player_stats)
        processed_count += 1
        print(f"Processed match {processed_count}/{len(match_ids)}: ID {match_id}")

    if all_player_stats:
        df_performance = pd.concat(all_player_stats, ignore_index=True)
        
        print("\nSuccessfully processed all matches.")
        print("Generated Performance DataFrame:")
        df_performance.info()
        print(df_performance.head())
        
        df_performance.to_csv('statsbomb_match_performance.csv', index=False)
        print("\n✅ Performance data saved to statsbomb_match_performance.csv")
    else:
        print("❌ No player stats were generated.")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find the file at '{matches_file}'.")
    print("Please make sure the 'open-data-master' folder is in your project directory.")