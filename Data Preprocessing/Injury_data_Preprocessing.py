import pandas as pd
import numpy as np

injury_data = pd.read_csv('Injury data.csv')
injury_data.drop_duplicates(inplace=True)

num_cols = injury_data.select_dtypes(include=[np.number]).columns
injury_data[num_cols] = injury_data[num_cols].fillna(injury_data[num_cols].mean())

injury_data["Injury_Risk_Score"] = np.log1p(injury_data["season_days_injured"])

injury_data["Games_Missed_Ratio"] = (
    1 - (injury_data["season_games_played"] / injury_data["season_matches_in_squad"].replace(0, np.nan))
)

injury_data["Injury_Severity_Index"] = (
    injury_data["total_days_injured"] / injury_data["total_games_played"].replace(0, np.nan)
)

print("Injury Data cleaned & engineered:")
print(injury_data.head())
