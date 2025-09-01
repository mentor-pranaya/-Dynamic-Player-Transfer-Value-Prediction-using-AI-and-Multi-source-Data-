import pandas as pd
import numpy as np
import re

# Load dataset with utf-8 encoding fix
df = pd.read_csv("serie-a_players_basic_info_23_24.csv", encoding="utf-8", on_bad_lines="skip")


# 1. Remove duplicates

df = df.drop_duplicates(subset=["player_id"])

# -----------------------
# 2. Clean player name
# -----------------------
# Keep only first part (remove position appended by mistake)
df["player_name"] = df["player_name"].astype(str).str.split().str[:2].str.join(" ")


# -----------------------
# 3. Fix nationality
# -----------------------
df["nationality"] = df["nationality"].astype(str).str.split("|").str[0].str.strip()

# -----------------------
# 4. Fix age
# -----------------------
# Drop unrealistic ages (<15 or >50)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.loc[(df["age"] < 15) | (df["age"] > 50), "age"] = np.nan

# -----------------------
# 5. Fix market value
# -----------------------
def parse_market_val(val):
    if pd.isna(val): 
        return np.nan
    val = str(val).replace("â‚¬", "").replace("€", "").strip().lower()
    if "m" in val:
        return float(re.sub(r"[^0-9.]", "", val)) * 1_000_000
    elif "k" in val:
        return float(re.sub(r"[^0-9.]", "", val)) * 1_000
    else:
        return pd.to_numeric(val, errors="coerce")

df["market_value_cleaned"] = df["market_value_text"].apply(parse_market_val)

# Prefer cleaned value over existing eur col if available
df["market_value_eur"] = df["market_value_cleaned"].fillna(df["market_value_eur"])

# -----------------------
# 6. Drop unnecessary cols
# -----------------------
df = df.drop(columns=["age", "market_value_text", "market_value_cleaned"])

# Save cleaned dataset
df.to_csv("serie-a_players_cleaned.csv", index=False)

print("✅ serie-a** dataset cleaned and saved!")
