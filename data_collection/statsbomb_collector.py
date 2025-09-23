import pandas as pd
import requests

# 1. We've updated the match_id to a valid one that exists in the repository.
# This ID is for a match from the FA Women's Super League.
match_id = '3788741'
url = f'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json'

# Fetch the data from the URL
response = requests.get(url)

# 2. Add a check to ensure the request was successful (status code 200 means "OK")
if response.status_code == 200:
    try:
        # The data is a list of event dictionaries. We can load it into a pandas DataFrame.
        df_events = pd.json_normalize(response.json(), sep='_')

        # Print the first 5 events to see what the data looks like
        print(f"✅ Successfully loaded data for match ID: {match_id}")
        print("First 5 events:")
        print(df_events.head())

        # You can also explore the columns available
        print("\nAvailable columns:")
        print(list(df_events.columns))

    except Exception as e:
        print(f"❌ Failed to parse JSON. Error: {e}")
else:
    print(f"❌ Failed to fetch data from URL. Status code: {response.status_code}")
    print(f"Please check if the match_id '{match_id}' is correct.")