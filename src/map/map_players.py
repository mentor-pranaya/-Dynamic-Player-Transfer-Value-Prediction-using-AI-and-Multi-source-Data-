import mysql.connector
from rapidfuzz import process, fuzz

# Connect to DB
db = mysql.connector.connect(
    host="localhost", user="himanshu", password="yahoonet", database="AIProject"
)
cur = db.cursor()

# Fetch player names
cur.execute("SELECT player_id, player_name FROM players")
statsbomb_players = cur.fetchall()  # [(id, name), ...]

cur.execute("SELECT id, name FROM players_trfrmrkt")
trfrmrkt_players = cur.fetchall()

# Prepare lookup
trfrmrkt_names = {pid: name for pid, name in trfrmrkt_players}
trfrmrkt_name_list = list(trfrmrkt_names.values())

# Match
for sb_id, sb_name in statsbomb_players:
    best_match, score, idx = process.extractOne(
        sb_name, trfrmrkt_name_list, scorer=fuzz.token_sort_ratio
    )
    if score >= 90:  # High confidence exact/fuzzy match
        tr_id = [pid for pid, name in trfrmrkt_names.items() if name == best_match][0]
        cur.execute("""
            INSERT IGNORE INTO player_mapping (statsbomb_player_id, trfrmrkt_player_id, canonical_name, match_type, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """, (sb_id, tr_id, sb_name, "fuzzy" if score < 100 else "exact", score))

db.commit()
cur.close()
db.close()

