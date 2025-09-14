import pandas as pd
import numpy as np

# load dataset
data = pd.read_csv("final_player_dataset.csv")

# date related features
data["dob"] = pd.to_datetime(data["dob"], errors="coerce")
data["birth_year"] = data["dob"].dt.year
data["birth_month"] = data["dob"].dt.month
data["age_start"] = data["start_year"] - data["birth_year"]

# physical features
data["bmi_new"] = data["weight_kg"] / ((data["height_cm"]/100)**2)
data["bmi_diff"] = data["bmi_new"] - data["bmi"]

# performance features
data["minutes_per_game"] = data["season_minutes_played"] / (data["season_games_played"] + 1)
data["injury_days_per_game"] = data["season_days_injured"] / (data["season_games_played"] + 1)
data["injury_days_per_minute"] = data["season_days_injured"] / (data["season_minutes_played"] + 1)

# rolling avg features
data["rolling_minutes"] = data.groupby("player_id")["season_minutes_played"].transform(lambda x: x.rolling(3, 1).mean())
data["rolling_injuries"] = data.groupby("player_id")["season_days_injured"].transform(lambda x: x.rolling(3, 1).mean())

# market value ratios
data["fee_per_rating"] = data["Fee"] / (data["fifa_rating"] + 1)
data["fee_per_age"] = data["Fee"] / (data["age"] + 1)
data["fee_per_game"] = data["Fee"] / (data["season_games_played"] + 1)

# sentiment interactions
data["sentiment_fee"] = data["Sentiment_Score"] * data["Fee"]
data["engagement_norm"] = data["Engagement"] / (data["Engagement"].max() + 1)
data["weighted_sent_ratio"] = data["Weighted_Sentiment"] / (data["Engagement"] + 1)

# encode categorical columns
data["club_code"] = data["Club"].astype("category").cat.codes
data["position_code"] = data["position"].astype("category").cat.codes
data["nation_code"] = data["nationality"].astype("category").cat.codes

# target for modeling
data["target_fee"] = data["Log_Fee"]

print("feature engineering done")
print(data.shape)
print(data.head())
