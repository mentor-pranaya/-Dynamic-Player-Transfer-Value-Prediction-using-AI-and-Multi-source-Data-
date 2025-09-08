import os
import json
import pandas as pd
import mysql.connector

# ------------------------------
# Database Connection Setup
# ------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",
    password="yahoonet",
    database="AIProject"
)
cursor = db.cursor()

# Competitions Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS competitions (
    file_name varchar(100),
    id INT PRIMARY KEY,
    country_name VARCHAR(100),
    competition_name VARCHAR(200),
    competition_gender VARCHAR(20),    
    competition_youth boolean, 
    competition_international boolean, 
    match_updated datetime, 
    match_updated_360 datetime, 
    match_available_360 datetime, 
    match_available datetime,
    season_id INT,
    season_name VARCHAR(50)
);""")

# Matches Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS matches (
    file_name varchar(100),
    match_id INT PRIMARY KEY,
    competition_id INT,
    season_id INT,
    match_date DATE,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    home_score INT,
    away_score INT,
    stadium VARCHAR(100),
    referee VARCHAR(100)
);""")


# ------------------------------
# Helper function to insert safely
# ------------------------------
def insert_many(query, data):
    cursor.executemany(query, data)
    db.commit()

# ------------------------------
# Base Path for StatsBomb Data
# ------------------------------
base_path = "/home/gubsend/Infosys Springboard/open-data-master/data"

# ------------------------------
# 1. Import Competitions
# ------------------------------
with open(os.path.join(base_path, "competitions.json"), "r", encoding="utf-8") as f:
    competitions = json.load(f)
print(competitions)
comp_data = []
for comp in competitions:
    comp_data.append((
        "competitions.json",
        comp["competition_id"],
        comp["country_name"],
        comp["competition_name"],
        comp["competition_gender"],
        comp["competition_youth"],
        comp["competition_international"],
        comp["match_updated"],
        comp["match_updated_360"],
        comp["match_available_360"],
        comp["season_id"],
        comp["season_name"]
    ))

insert_many("""
INSERT INTO competitions (file_name, id, country_name, competition_name, competition_gender, competition_youth, competition_international, match_updated, match_updated_360,match_available_360, season_id, season_name)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE country_name=VALUES(country_name);
""", comp_data)

print(f"✅ Inserted {len(comp_data)} competitions")

# ------------------------------
# 2. Import Matches
# ------------------------------
matches_path = os.path.join(base_path, "matches")
match_data = []

for comp_folder in os.listdir(matches_path):
    comp_folder_path = os.path.join(matches_path, comp_folder)
    if os.path.isdir(comp_folder_path):
        for file in os.listdir(comp_folder_path):
            with open(os.path.join(comp_folder_path, file), "r", encoding="utf-8") as f:
                matches = json.load(f)
                for match in matches:
                    match_data.append((
                        f"{comp_folder}/{file}",
                        match["match_id"],
                        match["competition"]["competition_id"],
                        match["season"]["season_id"],
                        match["match_date"],
                        match["home_team"]["home_team_name"],
                        match["away_team"]["away_team_name"],
                        match["home_score"],
                        match["away_score"],
                        match.get("stadium", {}).get("name"),
                        match.get("referee", {}).get("name")
                    ))

insert_many("""
INSERT INTO matches (file_name, match_id, competition_id, season_id, match_date, home_team, away_team, home_score, away_score, stadium, referee)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE home_score=VALUES(home_score), away_score=VALUES(away_score);
""", match_data)

print(f"✅ Inserted {len(match_data)} matches")



# ------------------------------
# Close Connection
# ------------------------------
cursor.close()
db.close()
print("🎉 All datasets imported successfully into MySQL!")


