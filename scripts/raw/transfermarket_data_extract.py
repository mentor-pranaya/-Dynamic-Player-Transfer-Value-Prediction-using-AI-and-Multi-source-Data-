# Create folder if needed
!mkdir -p ./data/raw/

# Download the dataset
!kaggle datasets download -d davidcariboo/player-scores -p ./data/raw/

# Unzip it into a subfolder
!unzip -q ./data/raw/player-scores.zip -d ./data/raw/player_scores/


import os
files = os.listdir("./data/raw/player_scores/")
print("Available files:", files)
# You should see CSVs like: competitions.csv, players.csv, games.csv, appearances.csv, clubs.csv

import pandas as pd

players_df = pd.read_csv("./data/raw/player_scores/players.csv")
print(players_df.shape)
print(players_df.columns.tolist())
print(players_df.head())

appearances_df = pd.read_csv("./data/raw/player_scores/appearances.csv")
print("Appearances shape:", appearances_df.shape)
print(appearances_df.columns.tolist())
print(appearances_df.head())

games_df = pd.read_csv("./data/raw/player_scores/games.csv")
print("Games shape:", games_df.shape)
print(games_df.columns.tolist())
print(games_df.head())

competitions_df = pd.read_csv("./data/raw/player_scores/competitions.csv")
print("Competitions shape:", competitions_df.shape)
print(competitions_df.columns.tolist())
print(competitions_df.head())

