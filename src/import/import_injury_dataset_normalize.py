import pandas as pd
import mysql.connector
from datetime import datetime

# ------------------------------
# DB Connection
# ------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",
    password="yahoonet",  # change to your MySQL password
    database="AIProject"
)
cursor = db.cursor()


# Players & injuries (main table)
cursor.execute(""" CREATE TABLE IF NOT EXISTS player_injuries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    team_name VARCHAR(100),
    position VARCHAR(50),
    age INT,
    season VARCHAR(20),
    fifa_rating INT,
    injury VARCHAR(100),
    date_of_injury DATE,
    date_of_return DATE
); """)

# Match performances related to injuries
# 'phase' distinguishes: "before", "missed", "after"
cursor.execute(""" CREATE TABLE IF NOT EXISTS injury_matches (
    id INT AUTO_INCREMENT PRIMARY KEY,
    injury_id INT,
    phase ENUM('before','missed','after'),
    match_number INT,
    result VARCHAR(20),
    opposition VARCHAR(100),
    gd INT,
    player_rating VARCHAR(10),
    FOREIGN KEY (injury_id) REFERENCES player_injuries(id) ON DELETE CASCADE
); """)

# ------------------------------
# Helper: parse dates
# ------------------------------
def parse_date(d):
    if pd.isna(d) or d == "N.A.":
        return None
    try:
        return datetime.strptime(d.strip('"'), "%b %d, %Y").date()
    except:
        return None

# ------------------------------
# Load CSV
# ------------------------------
injuries_file = "/home/gubsend/Infosys Springboard/player_injuries_impact.csv"
df = pd.read_csv(injuries_file)

# Normalize schema
for _, row in df.iterrows():
    # Insert into player_injuries
    cursor.execute("""
        INSERT INTO player_injuries
        (name, team_name, position, age, season, fifa_rating, injury, date_of_injury, date_of_return)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        row["Name"],
        row["Team Name"],
        row["Position"],
        int(row["Age"]) if not pd.isna(row["Age"]) else None,
        row["Season"],
        int(row["FIFA rating"]) if not pd.isna(row["FIFA rating"]) else None,
        row["Injury"],
        parse_date(row["Date of Injury"]),
        parse_date(row["Date of return"])
    ))
    injury_id = cursor.lastrowid  # get auto id
    
    # Function to insert related matches
    def insert_matches(phase, prefix):
        for i in range(1, 4):
            result = row.get(f"{prefix}{i}_{phase}_injury_Result", None)
            opposition = row.get(f"{prefix}{i}_{phase}_injury_Opposition", None)
            gd = row.get(f"{prefix}{i}_{phase}_injury_GD", None)
            player_rating = row.get(f"{prefix}{i}_{phase}_injury_Player_rating", None)

            # Some phases (missed) don’t have player ratings
            if pd.isna(result) and pd.isna(opposition):
                continue

            cursor.execute("""
                INSERT INTO injury_matches
                (injury_id, phase, match_number, result, opposition, gd, player_rating)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                injury_id, phase, i,
                None if pd.isna(result) else result,
                None if pd.isna(opposition) else opposition,
                None if pd.isna(gd) or gd == "N.A." else int(str(gd).replace("(S)", "").strip()),
                None if pd.isna(player_rating) or player_rating == "N.A." else str(player_rating)
            ))

    # Insert matches for each phase
    insert_matches("before", "Match")
    insert_matches("missed", "Match")
    insert_matches("after", "Match")

db.commit()
print("✅ Data successfully imported into normalized schema")

cursor.close()
db.close()

