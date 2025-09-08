import pandas as pd
import mysql.connector
import math
from datetime import datetime

# ------------------------------
# DB Connection
# ------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",
    password="yahoonet",  
    database="AIProject"
)
cursor = db.cursor()

# Transfermarkt Player Market Values
cursor.execute(""" CREATE TABLE IF NOT EXISTS transfermrkt (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player VARCHAR(100),
    club VARCHAR(100),
    competition VARCHAR(100),
    market_value BIGINT
); """)

# Player Injuries Impact
cursor.execute(""" CREATE TABLE IF NOT EXISTS player_injuries_impact (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    team_name VARCHAR(100),
    position VARCHAR(50),
    age INT,
    season VARCHAR(20),
    fifa_rating INT,
    injury VARCHAR(100),
    date_of_injury DATE,
    date_of_return DATE,
    match1_before_injury_result VARCHAR(20),
    match1_before_injury_opposition VARCHAR(100),
    match1_before_injury_gd INT,
    match1_before_injury_player_rating VARCHAR(10),
    match2_before_injury_result VARCHAR(20),
    match2_before_injury_opposition VARCHAR(100),
    match2_before_injury_gd INT,
    match2_before_injury_player_rating VARCHAR(10),
    match3_before_injury_result VARCHAR(20),
    match3_before_injury_opposition VARCHAR(100),
    match3_before_injury_gd INT,
    match3_before_injury_player_rating VARCHAR(10),
    match1_missed_match_result VARCHAR(20),
    match1_missed_match_opposition VARCHAR(100),
    match1_missed_match_gd INT,
    match2_missed_match_result VARCHAR(20),
    match2_missed_match_opposition VARCHAR(100),
    match2_missed_match_gd INT,
    match3_missed_match_result VARCHAR(20),
    match3_missed_match_opposition VARCHAR(100),
    match3_missed_match_gd INT,
    match1_after_injury_result VARCHAR(20),
    match1_after_injury_opposition VARCHAR(100),
    match1_after_injury_gd INT,
    match1_after_injury_player_rating VARCHAR(10),
    match2_after_injury_result VARCHAR(20),
    match2_after_injury_opposition VARCHAR(100),
    match2_after_injury_gd INT,
    match2_after_injury_player_rating VARCHAR(10),
    match3_after_injury_result VARCHAR(20),
    match3_after_injury_opposition VARCHAR(100),
    match3_after_injury_gd INT,
    match3_after_injury_player_rating VARCHAR(10)
);""")

# ------------------------------
# Helper: Convert market values like €110.00m → 110000000
# ------------------------------
def parse_market_value(value):
    if pd.isna(value) or value == "N.A.":
        return None
    value = value.replace("€", "").replace(",", "").strip()
    if value.endswith("m"):   # millions
        return int(float(value[:-1]) * 1_000_000)
    elif value.endswith("k"): # thousands
        return int(float(value[:-1]) * 1_000)
    else:
        try:
            return int(float(value))
        except:
            return None

# ------------------------------
# Import Transfermarkt Market Values
# ------------------------------
market_file = "/home/gubsend/Infosys Springboard/market_values_all_competitions.csv"
df_market = pd.read_csv(market_file)

# Convert market value to numeric
df_market["Market Value"] = df_market["Market Value"].apply(parse_market_value)

for _, row in df_market.iterrows():
    cursor.execute("""
        INSERT INTO transfermrkt (player, club, competition, market_value)
        VALUES (%s, %s, %s, %s)
    """, (row["Player"], row["Club"], row["Competition"], row["Market Value"]))

db.commit()
print(f"✅ Imported {len(df_market)} rows into transfermrkt")

# ------------------------------
# Import Player Injuries Impact
# ------------------------------
injuries_file = "/home/gubsend/Infosys Springboard/player_injuries_impact.csv"
df_injuries = pd.read_csv(injuries_file)

# Convert dates safely
def parse_date(d):
    if pd.isna(d) or d == "N.A.":
        return 0
    try:
        return datetime.strptime(d.strip('"'), "%b %d, %Y").date()
    except:
        return None

df_injuries["Date of Injury"] = df_injuries["Date of Injury"].apply(parse_date)
df_injuries["Date of return"] = df_injuries["Date of return"].apply(parse_date)

# Insert into MySQL
cols = df_injuries.columns.tolist()
placeholders = ",".join(["%s"] * len(cols))

# ------------------------------
# Clean Numeric Columns
# ------------------------------
numeric_cols = [c for c in df_injuries.columns if "gd" in c.lower() or c.lower() == "age" or "fifa rating" in c.lower()]

def safe_int(x):
    try:
        if pd.isna(x) or math.isnan(x) or str(x).strip().upper() == "N.A.":
            return 0
        return int(x)
    except:
        return 0

for col in numeric_cols:
    df_injuries[col] = df_injuries[col].apply(safe_int)

# ------------------------------
# Insert into MySQL
# ------------------------------
cols = df_injuries.columns.tolist()
placeholders = ",".join(["%s"] * len(cols))
#print(f"INSERT INTO player_injuries_impact ({','.join(c.lower().replace(' ', '_') for c in cols)}) VALUES ({placeholders})")
query = f"INSERT INTO player_injuries_impact ({','.join(c.lower().replace(' ', '_') for c in cols)}) VALUES ({placeholders})"
#print(query)
for _, row in df_injuries.iterrows():
    #print(row)
    cursor.execute(query, tuple(row))

db.commit()
print(f"✅ Imported {len(df_injuries)} rows into player_injuries_impact")

# ------------------------------
# Close connection
# ------------------------------
cursor.close()
db.close()
