import mysql.connector
from rapidfuzz import process, fuzz

# -----------------------
# DB CONFIG
# -----------------------
DB_CFG = dict(
    host="localhost",
    user="root",
    password="yahoonet",
    database="AIProject"
)

# -----------------------
# HELPERS
# -----------------------
def get_db():
    return mysql.connector.connect(**DB_CFG)

def normalize(name):
    import unicodedata, re
    if not name:
        return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()

def save_mapping(canonical, trf_id, sb_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO player_mapping (canonical_name, transfermarkt_id, statsbomb_player_id)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
        transfermarkt_id=VALUES(transfermarkt_id),
        statsbomb_player_id=VALUES(statsbomb_player_id)
    """, (canonical, trf_id, sb_id))
    db.commit()
    cur.close()
    db.close()

# -----------------------
# MAIN
# -----------------------
def main():
    db = get_db()
    cur = db.cursor()

    # Load data
    cur.execute("SELECT transfermarkt_id, name FROM players_trfrmrkt")
    trf = [(r[0], r[1]) for r in cur.fetchall()]

    cur.execute("SELECT player_id, player_nickname FROM players")
    sb = [(r[0], r[1]) for r in cur.fetchall()]

    #cur.execute("SELECT id, name FROM player_injuries")
    #inj = [(r[0], r[1]) for r in cur.fetchall()]

    cur.close()
    db.close()

    sb_lookup = {normalize(name): (pid, name) for pid, name in sb}
    #inj_lookup = {normalize(name): (pid, name) for pid, name in inj}
    sb_names = [normalize(n) for _, n in sb]
    #inj_names = [normalize(n) for _, n in inj]

    auto_count = 0
    for trf_id, trf_name in trf:
        trf_norm = normalize(trf_name)

        # Find best match in StatsBomb
        sb_best = process.extractOne(trf_norm, sb_names, scorer=fuzz.token_sort_ratio)
        sb_id = None
        if sb_best and sb_best[1] >= 90:
            sb_match = sb_lookup[sb_best[0]]
            sb_id = sb_match[0]

        """
        # Find best match in Injuries
        inj_best = process.extractOne(trf_norm, inj_names, scorer=fuzz.token_sort_ratio)
        inj_id = None
        if inj_best and inj_best[1] >= 90:
            inj_match = inj_lookup[inj_best[0]]
            inj_id = inj_match[0]
        """
        # If we have at least one confident match
        if sb_id :
            save_mapping(trf_name, trf_id, sb_id)
            auto_count += 1

    print(f"✅ Auto-mapping complete: {auto_count} players inserted into player_mapping")

if __name__ == "__main__":
    main()

