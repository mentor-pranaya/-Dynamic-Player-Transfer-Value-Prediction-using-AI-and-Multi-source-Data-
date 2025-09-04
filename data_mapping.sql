CREATE TABLE player_mapping (
    id INT AUTO_INCREMENT PRIMARY KEY,
    statsbomb_player_id INT,
    trfrmrkt_player_id INT,
    canonical_name VARCHAR(150),
    match_type ENUM('exact', 'fuzzy', 'manual') DEFAULT 'manual',
    confidence FLOAT DEFAULT NULL,
    UNIQUE(statsbomb_player_id, trfrmrkt_player_id),
    FOREIGN KEY (statsbomb_player_id) REFERENCES players(player_id) ON DELETE CASCADE,
    FOREIGN KEY (trfrmrkt_player_id) REFERENCES players_trfrmrkt(id) ON DELETE CASCADE
);


CREATE TABLE player_performance_summary AS
SELECT player, COUNT(*) AS total_events,
       SUM(CASE WHEN type='Goal' THEN 1 ELSE 0 END) AS goals,
       SUM(CASE WHEN type='Pass' THEN 1 ELSE 0 END) AS passes,
       SUM(CASE WHEN type='Yellow Card' THEN 1 ELSE 0 END) AS yellows
FROM eventsNew1
GROUP BY player;

CREATE TABLE player_sentiment_daily AS
SELECT player_name,
       DATE(created_at) as day,
       AVG(polarity) as avg_polarity,
       COUNT(*) as mentions
FROM (
    SELECT player_name, created_at, polarity FROM twitter_sentiments
    UNION ALL
    SELECT player_name, created_at, polarity FROM reddit_sentiments
    UNION ALL
    SELECT player_name, created_at, polarity FROM medium_sentiments
) all_sentiments
GROUP BY player_name, day;
