# -----------------------------------------
# STEP 0: Install Required Packages
# -----------------------------------------
# statsbombpy: Library to access StatsBomb open data
# pandas: For data handling
# matplotlib & seaborn: For data visualization
# !pip install statsbombpy pandas matplotlib seaborn

# -----------------------------------------
# STEP 1: Import Required Libraries
# -----------------------------------------
import pandas as pd              # For handling tabular data (like spreadsheets)
import matplotlib.pyplot as plt  # For creating charts
import seaborn as sns            # For prettier charts (built on top of matplotlib)
from statsbombpy import sb       # StatsBomb API for football (soccer) data

# -----------------------------------------
# STEP 2: Load Match Data from a Competition
# -----------------------------------------
# Here we fetch all matches from FIFA World Cup 2018
# competition_id = 43 (FIFA World Cup), season_id = 3 (2018)
matches = sb.matches(competition_id=43, season_id=3)

# Display the first few rows of match data to understand its structure
print("Matches dataset sample:")
print(matches.head())

# Save match dataset to CSV file (deliverable for Week 1)
matches.to_csv("matches.csv", index=False)

# -----------------------------------------
# STEP 3: Load Events for a Specific Match
# -----------------------------------------
# Each match has a unique match_id. We'll pick the first match from our matches dataset.
match_id = matches.loc[0, "match_id"]  # Access the first match_id
events = sb.events(match_id=match_id)  # Fetch all events (passes, shots, fouls, etc.) for that match

# Display first few rows of events data
print("\nEvents dataset sample:")
print(events.head())

# Save raw events dataset for this match (deliverable for Week 1)
events.to_csv(f"match_{match_id}_events.csv", index=False)

# -----------------------------------------
# STEP 4: Basic Data Exploration (EDA)
# -----------------------------------------

# 4.1: View general information about the dataset (columns, datatypes, etc.)
print("\nEvents Data Info:")
print(events.info())

# 4.2: Check for missing values (important for later preprocessing)
print("\nMissing values in each column:")
print(events.isnull().sum())

# 4.3: Analyze event types (e.g., how many passes, shots, fouls, etc.)
event_counts = events["type"].value_counts()
print("\nEvent Type Counts:")
print(event_counts)

# Plot top 10 event types for quick understanding of match dynamics
plt.figure(figsize=(10,5))
sns.barplot(x=event_counts.index[:10], y=event_counts.values[:10])
plt.title("Top 10 Event Types in Match")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 4.4: Focus on Player Performance - Example: Shots Taken
shots = events[events["type"] == "Shot"]  # Filter events to only include "Shot"
print("\nShots Data Sample (Player, Team, Outcome):")
print(shots[["player", "team", "shot_outcome"]].head())

# Count number of shots per player (Top 10)
shots_per_player = shots["player"].value_counts().head(10)

# Visualize top 10 players by shots
plt.figure(figsize=(8,5))
sns.barplot(x=shots_per_player.index, y=shots_per_player.values)
plt.title("Top 10 Players by Shots in Match")
plt.ylabel("Shots")
plt.xticks(rotation=45)
plt.show()

