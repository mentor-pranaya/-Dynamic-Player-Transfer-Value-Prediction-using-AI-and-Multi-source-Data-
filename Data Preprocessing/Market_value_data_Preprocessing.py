import pandas as pd
import numpy as np

market_value = pd.read_csv('Market value data.csv')

market_value.rename(columns={"Name": "Player", "Fee": "Market_Value"}, inplace=True)
market_value["Player"] = market_value["Player"].str.replace("Ã©", "é").str.replace("Ã±", "ñ")

market_value["Market_Value"] = pd.to_numeric(market_value["Market_Value"], errors="coerce")

market_value.dropna(subset=["Player", "Market_Value"], inplace=True)
market_value["Market_Value_Log"] = np.log1p(market_value["Market_Value"])

print("Market Value cleaned & engineered:")
print(market_value.head())
