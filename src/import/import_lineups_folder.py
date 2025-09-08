import os
import json
import mysql.connector

# ------------------------------
# Database Connection
# ------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",  # replace with your MySQL password
    database="statsbomb"
)
cursor = db.cursor()
USE statsbomb;

# Teams
cursor.execute("""
CREATE TABLE IF NOT EXISTS teams (
    
    team_id INT PRIMARY KEY,
    team_name VARCHAR(100)
);""")

# Players
cursor.execute(""" CREATE TABLE IF NOT EXISTS players (
    player_id INT PRIMARY KEY,
    player_name VARCHAR(100),
    player_nickname VARCHAR(100),
    jersey_number INT,
    country_id INT,
    country_name VARCHAR(100)
); """)

# Lineup Positions (captures substitutions, tactical shifts, etc.)
cursor.execute(""" CREATE TABLE IF NOT EXISTS lineup_positions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT,
    team_id INT,
    position_id INT,
    position_name VARCHAR(100),
    from_time VARCHAR(10),
    to_time VARCHAR(10),
    from_period INT,
    to_period INT,
    start_reason VARCHAR(100),
    end_reason VARCHAR(100),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
); """)

# Player Cards (yellow/red cards, reasons)
cursor.execute(""" CREATE TABLE IF NOT EXISTS player_cards (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT,
    team_id INT,
    card_time VARCHAR(10),
    card_type VARCHAR(50),
    reason VARCHAR(200),
    period INT,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
); """)

def insert_many(query, data):
    if data:
        cursor.executemany(query, data)
        db.commit()

# ------------------------------
# Paths
# ------------------------------
lineups_path = "/home/gubsend/Infosys Springboard/open-data-master/data/lineups"
filecnt=1
# ------------------------------
# Process Lineup JSON Files
# ------------------------------
for file in os.listdir(lineups_path):
    if file.endswith(".json"):
        with open(os.path.join(lineups_path, file), "r", encoding="utf-8") as f:
            teams_data = json.load(f)

        team_records, player_records, position_records, card_records = [], [], [], []

        for team in teams_data:
            team_id = team["team_id"]
            team_name = team["team_name"]

            team_records.append((team_id, team_name))

            for player in team["lineup"]:
                player_id = player["player_id"]
                player_name = player["player_name"]
                player_nickname = player.get("player_nickname")
                jersey = player.get("jersey_number")
                country_id = player["country"]["id"] if "country" in player else None
                country_name = player["country"]["name"] if "country" in player else None

                player_records.append((
                    player_id, player_name, player_nickname, jersey,
                    country_id, country_name
                ))

                # Handle positions
                for pos in player.get("positions", []):
                    position_records.append((
                        player_id, team_id,
                        pos["position_id"], pos["position"],
                        pos.get("from"), pos.get("to"),
                        pos.get("from_period"), pos.get("to_period"),
                        pos.get("start_reason"), pos.get("end_reason")
                    ))

                # Handle cards
                for card in player.get("cards", []):
                    card_records.append((
                        player_id, team_id,
                        card.get("time"), card.get("card_type"),
                        card.get("reason"), card.get("period")
                    ))

        # Insert into DB
        insert_many("""
            INSERT IGNORE INTO teams (team_id, team_name)
            VALUES (%s, %s)
        """, team_records)

        insert_many("""
            INSERT IGNORE INTO players (player_id, player_name, player_nickname, jersey_number, country_id, country_name)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, player_records)

        insert_many("""
            INSERT INTO lineup_positions (player_id, team_id, position_id, position_name, from_time, to_time, from_period, to_period, start_reason, end_reason)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, position_records)

        insert_many("""
            INSERT INTO player_cards (player_id, team_id, card_time, card_type, reason, period)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, card_records)

        print(f"✅ Processed {file}: {len(player_records)} players, {len(position_records)} positions, {len(card_records)} cards. File# {filecnt}")
        filecnt=filecnt+1

# ------------------------------
# Close Connection
# ------------------------------
cursor.close()
db.close()
print("🎉 Lineups imported successfully!")

