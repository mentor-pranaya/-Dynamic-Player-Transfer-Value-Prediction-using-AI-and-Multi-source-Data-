import pandas as pd
import json

# 1. Load competitions
def load_competitions(path="./data/raw/statsbomb/data/competitions.json"):
    return pd.read_json(path)

# 2. Load matches for a given competition & season
def load_matches(competition_id, season_id, base="./data/raw/statsbomb/data/matches/"):
    file_path = f"{base}{competition_id}/{season_id}.json"
    return pd.read_json(file_path)

# 3. Load events for a given match
def load_events(match_id, base="./data/raw/statsbomb/data/events/"):
    file_path = f"{base}{match_id}.json"
    return pd.read_json(file_path)

# 4. Load lineups
def load_lineups(match_id, base="./data/raw/statsbomb/data/lineups/"):
    file_path = f"{base}{match_id}.json"
    return pd.read_json(file_path)

# Example usage:
competitions = load_competitions()
print("Competitions:\n", competitions.head())

# Pick first competition/season and load matches
first_comp = competitions.iloc[0]
matches = load_matches(first_comp['competition_id'], first_comp['season_id'])
print("\nMatches sample:\n", matches.head())

# Pick one match and load events
if not matches.empty:
    sample_match_id = matches['match_id'].iloc[0]
    events = load_events(sample_match_id)
    print("\nEvents sample:\n", events.head())

# 1. Competitions summary
print("Total competitions:", competitions.shape[0])
print(competitions[['competition_id','competition_name','season_name']].head(10))

# 2. Matches summary
print("Matches shape:", matches.shape)
print("Columns:", matches.columns.tolist())

print("\nGoals Summary:")
print("Home goals mean:", matches['home_score'].mean())
print("Away goals mean:", matches['away_score'].mean())

# 3. Events summary
print("\nEvents shape:", events.shape)
print("Event types sample:", events['type'].head(10))

# Count events by type
event_type_counts = events['type'].apply(lambda x: x['name']).value_counts()
print("\nTop event types:\n", event_type_counts.head(15))

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Distribution of goals
plt.figure(figsize=(6,4))
sns.histplot(matches['home_score'], bins=range(0,6), kde=False, color="blue", label="Home Goals")
sns.histplot(matches['away_score'], bins=range(0,6), kde=False, color="red", label="Away Goals")
plt.legend()
plt.title("Distribution of Goals (Home vs Away)")
plt.show()

# 2. Top 10 event types
event_type_counts = events['type'].apply(lambda x: x['name']).value_counts().head(10)

plt.figure(figsize=(8,4))
sns.barplot(x=event_type_counts.values, y=event_type_counts.index, palette="viridis")
plt.title("Top 10 Event Types")
plt.xlabel("Count")
plt.ylabel("Event Type")
plt.show()

# 3. Matches per competition
comp_match_counts = competitions.groupby('competition_name').size().sort_values(ascending=False).head(10)
comp_match_counts.plot(kind="barh", figsize=(8,4), color="teal")
plt.title("Top Competitions by Dataset Coverage")
plt.show()


