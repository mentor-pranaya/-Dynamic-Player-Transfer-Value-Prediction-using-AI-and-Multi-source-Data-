from statsbombpy import sb
import pandas as pd

# --- Configuration ---
TARGET_COMPETITION = "La Liga"
TARGET_SEASON = "2015/2016"

print(f"Starting data fetch for {TARGET_COMPETITION} - Season {TARGET_SEASON}")

try:
    # --- Step 1: Find the Competition and Season IDs ---
    competitions = sb.competitions()
    our_competition = competitions[
        (competitions['competition_name'] == TARGET_COMPETITION) &
        (competitions['season_name'] == TARGET_SEASON)
    ]
    if our_competition.empty:
        raise ValueError("This competition/season is not available in the free data.")
    competition_id = our_competition['competition_id'].iloc[0]
    season_id = our_competition['season_id'].iloc[0]
    print(f"Found Competition ID: {competition_id}, Season ID: {season_id}")

    # --- Step 2: Get All Match IDs for that Season ---
    print("Finding all matches in the season...")
    matches_in_season = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches_in_season['match_id'].tolist()
    print(f"Found {len(match_ids)} matches.")

    # --- Step 3 (FIXED): Loop Through Each Match ID to Fetch Events ---
    print("Fetching event data for all matches one by one... This will take several minutes.")
    
    list_of_match_events = []
    # Loop through each match ID
    for i, match_id in enumerate(match_ids):
        # Print progress so you know it's working
        print(f"--> Fetching data for match {i+1}/{len(match_ids)} (ID: {match_id})")
        
        # Fetch events for the single match
        events = sb.events(match_id=match_id)
        
        # Add the events DataFrame to our list
        list_of_match_events.append(events)

    print("\nAll event data has been fetched. Now combining into a single file...")
    
    # Combine all the individual DataFrames in the list into one big DataFrame
    all_events = pd.concat(list_of_match_events)

    # --- Step 4: Save the Combined Data to a CSV File ---
    output_filename = f"{TARGET_COMPETITION.replace(' ', '_')}_{TARGET_SEASON.replace('/', '-')}_all_events.csv"
    print(f"Saving data to {output_filename}...")
    all_events.to_csv(output_filename, index=False)
    
    print(f"\n✅ Success! Data for the entire season saved to {output_filename}")
    print("\n--- Data Summary ---")
    print(all_events.info())

except Exception as e:
    print(f"\nAn error occurred: {e}")