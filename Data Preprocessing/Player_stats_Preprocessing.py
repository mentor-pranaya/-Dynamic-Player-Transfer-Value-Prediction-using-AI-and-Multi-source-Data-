
import pandas as pd
import numpy as np

player_stats = pd.read_csv('Player_Stats.csv', encoding='latin1', sep=';')
player_stats.columns = player_stats.columns.str.strip()

player_stats.drop_duplicates(inplace=True)

num_cols = player_stats.select_dtypes(include=[np.number]).columns
player_stats[num_cols] = player_stats[num_cols].fillna(player_stats[num_cols].mean())

cat_cols = player_stats.select_dtypes(include=['object']).columns
player_stats[cat_cols] = player_stats[cat_cols].fillna(player_stats[cat_cols].mode().iloc[0])

player_stats = player_stats.assign(
    Performance_Index=(
        (player_stats["Goals"].astype(float) + player_stats["Assists"].astype(float)) /
        player_stats["90s"].replace(0, np.nan)
    ),
    Shot_Accuracy=(
        player_stats["SoT"].astype(float) / player_stats["Shots"].replace(0, np.nan)
    ),
    Pass_Efficiency=(
        player_stats["PasTotCmp"].astype(float) / player_stats["PasTotAtt"].replace(0, np.nan)
    )
).copy()

print("Player Stats cleaned & engineered:")
print(player_stats.head())
