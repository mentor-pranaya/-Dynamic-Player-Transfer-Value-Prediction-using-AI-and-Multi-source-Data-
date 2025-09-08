import streamlit as st
import mysql.connector
from rapidfuzz import fuzz, process

# -----------------------
# DB CONFIG
# -----------------------
DB_CFG = dict(
    host="localhost",
    user="himanshu",
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

def load_names():
    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT id, name FROM players_trfrmrkt where id not in (select trfrmrkt_player_id from player_mapping)")
    trf = [{"id": r[0], "name": r[1], "norm": normalize(r[1])} for r in cur.fetchall()]

    cur.execute("SELECT player_id, player_name FROM players")
    sb = [{"id": r[0], "name": r[1], "norm": normalize(r[1])} for r in cur.fetchall()]

    cur.execute("SELECT id, name FROM player_injuries")
    inj = [{"id": r[0], "name": r[1], "norm": normalize(r[1])} for r in cur.fetchall()]

    cur.close()
    db.close()
    return trf, sb, inj

def save_mapping(canonical, trf_id, sb_id, inj_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO player_mapping (canonical_name, trfrmrkt_player_id, statsbomb_player_id, injury_player_id)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        trfrmrkt_player_id=VALUES(trfrmrkt_player_id),
        statsbomb_player_id=VALUES(statsbomb_player_id),
        injury_player_id=VALUES(injury_player_id)
    """, (canonical, trf_id, sb_id, inj_id))
    db.commit()
    cur.close()
    db.close()

# -----------------------
# STREAMLIT APP
# -----------------------
st.set_page_config(page_title="Player Mapping Review", layout="wide")
st.title("⚽ Player Mapping Review Tool")

st.write("This tool helps you manually review and confirm player mappings between Transfermarkt, StatsBomb, and Injury data.")

# Load data
trf, sb, inj = load_names()

# Pick one Transfermarkt player at a time
trf_names = [f"{p['id']} - {p['name']}" for p in trf]
selected = st.selectbox("Select a Transfermarkt Player", trf_names)

if selected:
    trf_id = int(selected.split(" - ")[0])
    trf_player = next(p for p in trf if p["id"] == trf_id)

    st.subheader(f"🎯 Transfermarkt Player: {trf_player['name']}")

    # Fuzzy match suggestions
    sb_candidates = process.extract(trf_player["norm"], [p["norm"] for p in sb], scorer=fuzz.token_sort_ratio, limit=5)
    inj_candidates = process.extract(trf_player["norm"], [p["norm"] for p in inj], scorer=fuzz.token_sort_ratio, limit=5)

    # StatsBomb match
    st.write("### 🔵 StatsBomb Suggestions")
    sb_choice = st.radio(
        "Pick StatsBomb Player",
        ["None"] + [f"{score}% | {sb[i]['id']} - {sb[i]['name']}" for i, score, _ in [( [idx for idx, p in enumerate(sb) if p['norm']==cand][0], sc, cand ) for cand, sc, _ in sb_candidates]]
    )

    sb_id = None
    if sb_choice != "None":
        sb_id = int(sb_choice.split(" | ")[1].split(" - ")[0])

    # Injury match
    st.write("### ❤️ Injury Dataset Suggestions")
    inj_choice = st.radio(
        "Pick Injury Player",
        ["None"] + [f"{score}% | {inj[i]['id']} - {inj[i]['name']}" for i, score, _ in [( [idx for idx, p in enumerate(inj) if p['norm']==cand][0], sc, cand ) for cand, sc, _ in inj_candidates]]
    )

    inj_id = None
    if inj_choice != "None":
        inj_id = int(inj_choice.split(" | ")[1].split(" - ")[0])

    # Canonical name input
    canonical = st.text_input("Canonical Player Name", value=trf_player["name"])

    if st.button("💾 Save Mapping"):
        save_mapping(canonical, trf_id, sb_id, inj_id)
        st.success("✅ Mapping saved!")

