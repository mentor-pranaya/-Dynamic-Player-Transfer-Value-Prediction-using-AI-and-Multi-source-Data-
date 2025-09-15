import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    1. Queries the top 500 players by composite score.
    2. Creates 3 charts:

   Bar chart → Top 20 players overall.
   Boxplot → Top 10 per competition (distribution view).
   Scatterplot → Injury Score vs Market Value (colored by competition).
"""

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
        pt.name AS player_name,
        cl.name AS club_name,
        comp.name AS competition_name,
        pf.composite_score,
        pf.injury_score,
        pf.sentiment_score,
        pf.availability_score,
        pf.value_score,
        lm.market_value,
        RANK() OVER (ORDER BY pf.composite_score DESC) AS overall_rank,
        RANK() OVER (PARTITION BY comp.name ORDER BY pf.composite_score DESC) AS competition_rank,
        RANK() OVER (PARTITION BY cl.name ORDER BY pf.composite_score DESC) AS club_rank
    FROM player_features pf
    JOIN player_mapping pm ON pf.mapping_id = pm.id
    LEFT JOIN players_trfrmrkt pt ON pm.trfrmrkt_id = pt.id
    LEFT JOIN market_values_trfrmrkt mv ON pm.trfrmrkt_id = mv.player_id
    LEFT JOIN latest_mv lm ON mv.player_id = lm.player_id AND lm.rn = 1
    LEFT JOIN clubs_trfrmrkt cl ON mv.club_id = cl.id
    LEFT JOIN competitions_trfrmrkt comp ON mv.competition_id = comp.id
    ORDER BY pf.composite_score DESC
    LIMIT 500;
    """
    conn = mysql.connector.connect(**db_cfg)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# ---------------------------
# Plots
# ---------------------------
def visualize(df):
    plt.figure(figsize=(12,6))
    top20 = df.sort_values("overall_rank").head(20)
    sns.barplot(x="composite_score", y="player_name", data=top20, palette="viridis")
    plt.title("🏆 Top 20 Players by Composite Score")
    plt.xlabel("Composite Score")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    comp_top10 = df.groupby("competition_name").apply(lambda x: x.sort_values("competition_rank").head(10)).reset_index(drop=True)
    sns.boxplot(x="competition_name", y="composite_score", data=comp_top10)
    plt.title("📊 Top 10 Players by Competition")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    sns.scatterplot(x="injury_score", y="market_value", hue="competition_name", data=df, alpha=0.7)
    plt.title("💡 Injury Score vs Market Value")
    plt.xlabel("Injury Score (higher = more injuries)")
    plt.ylabel("Market Value (€)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = get_dashboard_data()
    print(df.head(10))  # preview
    visualize(df)
