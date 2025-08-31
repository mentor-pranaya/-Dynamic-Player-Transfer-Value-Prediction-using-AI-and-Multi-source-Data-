import pandas as pd
import mysql.connector

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


# Master list of players_trfrmrkt
cursor.execute(""" CREATE TABLE IF NOT EXISTS players_trfrmrkt (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE
); """)

# Clubs
cursor.execute(""" CREATE TABLE IF NOT EXISTS clubs_trfrmrkt (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE
); """)

# Competitions
cursor.execute(""" CREATE TABLE IF NOT EXISTS competitions_trfrmrkt (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE
); """)

# Market values (history table)
cursor.execute(""" CREATE TABLE IF NOT EXISTS market_values_trfrmrkt (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT,
    club_id INT,
    competition_id INT,
    market_value BIGINT,
    snapshot_date DATETIME DEFAULT current_timestamp,
    FOREIGN KEY (player_id) REFERENCES players_trfrmrkt(id) ON DELETE CASCADE,
    FOREIGN KEY (club_id) REFERENCES clubs_trfrmrkt(id) ON DELETE CASCADE,
    FOREIGN KEY (competition_id) REFERENCES competitions_trfrmrkt(id) ON DELETE CASCADE
); """)

# ------------------------------
# Helper: convert market values
# ------------------------------
def parse_market_value(value):
    if pd.isna(value): 
        return None
    value = value.replace("€", "").replace(",", "").strip()
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    elif value.endswith("k"):
        return int(float(value[:-1]) * 1_000)
    else:
        return int(value)

# ------------------------------
# Load CSV
# ------------------------------
file_path = "/home/gubsend/Infosys Springboard/market_values_all_competitions.csv"
df = pd.read_csv(file_path)

for _, row in df.iterrows():
    # Insert player if not exists
    cursor.execute("INSERT IGNORE INTO players_trfrmrkt (name) VALUES (%s)", (row["Player"],))
    cursor.execute("SELECT id FROM players_trfrmrkt WHERE name=%s", (row["Player"],))
    player_id = cursor.fetchone()[0]

    # Insert club if not exists
    cursor.execute("INSERT IGNORE INTO clubs_trfrmrkt (name) VALUES (%s)", (row["Club"],))
    cursor.execute("SELECT id FROM clubs_trfrmrkt WHERE name=%s", (row["Club"],))
    club_id = cursor.fetchone()[0]

    # Insert competition if not exists
    cursor.execute("INSERT IGNORE INTO competitions_trfrmrkt (name) VALUES (%s)", (row["Competition"],))
    cursor.execute("SELECT id FROM competitions_trfrmrkt WHERE name=%s", (row["Competition"],))
    competition_id = cursor.fetchone()[0]

    # Insert market value
    cursor.execute("""
        INSERT INTO market_values_trfrmrkt (player_id, club_id, competition_id, market_value)
        VALUES (%s,%s,%s,%s)
    """, (
        player_id,
        club_id,
        competition_id,
        parse_market_value(row["Market Value"])
    ))

db.commit()
print("✅ Transfermarkt data imported into normalized schema")

cursor.close()
db.close()

