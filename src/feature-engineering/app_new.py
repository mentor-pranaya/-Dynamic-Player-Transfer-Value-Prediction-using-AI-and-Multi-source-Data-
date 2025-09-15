import streamlit as st
import mysql.connector
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# DB Connection
# ---------------------------
db_cfg = dict(
    host="localhost",
    user="himanshu",
    password="yahoonet",
    database="AIProject"
)

"""
    #9
    track how a player’s market value changes relative to injury events and sentiment shifts.
    click a player name and instantly see how **injuries impacted their value!
    Player Trendlines section.
    Line chart of market value over time**.
    Injury events shown as red shaded regions (date\_of\_injury → date\_of\_return).
    Injury names displayed as vertical annotations.

"""
@st.cache_data
def get_dashboard_data():
    query = """
    WITH latest_mv AS (
        SELECT
            mv.player_id,
            mv.market_value,
            mv.snapshot_date,
            ROW_NUMBER() OVER (PARTITION BY mv.player_id ORDER BY mv.snapshot_date DESC) AS rn
        FROM market_values_trfrmrkt mv
    )
    SELECT
        pt.id AS player_id,
        pt.name AS player_name,
        cl.name AS club_name,
        comp.name AS competition_name,
        pf.composite_score,
        pf.injury_score,
        pf.sentiment_score,
        pf.availability_score,
        pf.value_score,
        lm.market_value,
        RANK() OVER (ORDER BY pf.composite_score DESC) AS overall_rank
    FROM player_features pf
    JOIN player_mapping pm ON pf.mapping_id = pm.id
    LEFT JOIN players_trfrmrkt pt ON pm.trfrmrkt_id = pt.id
    LEFT JOIN market_values_trfrmrkt mv ON pm.trfrmrkt_id = mv.player_id
    LEFT JOIN latest_mv lm ON mv.player_id = lm.player_id AND lm.rn = 1
    LEFT JOIN clubs_trfrmrkt cl ON mv.club_id = cl.id
    LEFT JOIN competitions_trfrmrkt comp ON mv.competition_id = comp.id
    ORDER BY pf.composite_score DESC
    LIMIT 1000;
    """
    conn = mysql.connector.connect(**db_cfg)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


@st.cache_data
def get_market_value_trend(player_id: int):
    query = """
    SELECT snapshot_date, market_value
    FROM market_values_trfrmrkt
    WHERE player_id = %s
    ORDER BY snapshot_date
    """
    conn = mysql.connector.connect(**db_cfg)
    df = pd.read_sql(query, conn, params=(player_id,))
    conn.close()
    return df


@st.cache_data
def get_injury_events(player_id: int):
    query = """
    SELECT date_of_injury, date_of_return, injury
    FROM player_injuries
    WHERE name IN (SELECT name FROM players_trfrmrkt WHERE id = %s)
    """
    conn = mysql.connector.connect(**db_cfg)
    df = pd.read_sql(query, conn, params=(player_id,))
    conn.close()
    return df


# ---------------------------
# Streamlit Layout
# ---------------------------
st.set_page_config(page_title="⚽ Player Dashboard", layout="wide")

st.title("⚽ Player Analytics Dashboard")
st.markdown("Explore player rankings, competition insights, and market value trends.")

df = get_dashboard_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
competition_filter = st.sidebar.multiselect("Competition", sorted(df["competition_name"].dropna().unique()))
club_filter = st.sidebar.multiselect("Club", sorted(df["club_name"].dropna().unique()))
top_n = st.sidebar.slider("Number of Top Players", 10, 100, 20)

# Apply filters
filtered_df = df.copy()
if competition_filter:
    filtered_df = filtered_df[filtered_df["competition_name"].isin(competition_filter)]
if club_filter:
    filtered_df = filtered_df[filtered_df["club_name"].isin(club_filter)]

# Show top players table
st.subheader("🏆 Top Players")
st.dataframe(filtered_df.head(top_n))

# ---------------------------
# Charts
# ---------------------------
st.subheader("📊 Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Players by Composite Score")
    fig, ax = plt.subplots(figsize=(8,6))
    top_players = filtered_df.sort_values("overall_rank").head(top_n)
    sns.barplot(x="composite_score", y="player_name", data=top_players, palette="viridis", ax=ax)
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Player")
    st.pyplot(fig)

with col2:
    st.markdown("### Competition Comparison")
    fig, ax = plt.subplots(figsize=(8,6))
    comp_top10 = filtered_df.groupby("competition_name").apply(lambda x: x.sort_values("overall_rank").head(10)).reset_index(drop=True)
    if not comp_top10.empty:
        sns.boxplot(x="competition_name", y="composite_score", data=comp_top10, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("Not enough competition data for selected filters.")

st.markdown("### Injury Score vs Market Value")
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(
    x="injury_score", y="market_value", hue="competition_name",
    data=filtered_df, alpha=0.7, ax=ax
)
ax.set_xlabel("Injury Score (higher = more injuries)")
ax.set_ylabel("Market Value (€)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
st.pyplot(fig)

# ---------------------------
# Player Trendlines
# ---------------------------
st.subheader("📈 Player Market Value & Injuries Over Time")

player_choice = st.selectbox("Select a player to view trends", filtered_df["player_name"].unique())

if player_choice:
    player_row = filtered_df[filtered_df["player_name"] == player_choice].iloc[0]
    player_id = player_row["player_id"]

    mv_df = get_market_value_trend(player_id)
    inj_df = get_injury_events(player_id)

    if mv_df.empty:
        st.warning("No market value history available for this player.")
    else:
        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(x="snapshot_date", y="market_value", data=mv_df, marker="o", ax=ax, label="Market Value (€)")

        # overlay injuries
        for _, row in inj_df.iterrows():
            ax.axvspan(row["date_of_injury"], row["date_of_return"], color="red", alpha=0.3)
            ax.text(row["date_of_injury"], mv_df["market_value"].max()*0.9, row["injury"], rotation=90, fontsize=8, color="red")

        ax.set_title(f"Market Value vs Injuries: {player_choice}")
        ax.set_ylabel("Market Value (€)")
        st.pyplot(fig)

