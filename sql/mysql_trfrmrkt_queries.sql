drop table if exists eventsNew;
drop table if exists transfermrkt;
drop table if exists player_injuries_impact;

select count(*) from player_injuries_impact;
select count(*) from transfermrkt;
select * from teams;
select * from lineup_positions;
select curdate(), CURRENT_DATE(),now();
select * from player_cards;
drop table if exists players;
drop table if exists teams;
drop table if exists lineup_positions;
drop table if exists player_cards;
CREATE TABLE IF NOT EXISTS market_values_trfrmrkt (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT,
    club_id INT,
    competition_id INT,
    market_value BIGINT,
    snapshot_date DATETIME DEFAULT current_timestamp, 
    FOREIGN KEY (player_id) REFERENCES players_trfrmrkt(id) ON DELETE CASCADE,
    FOREIGN KEY (club_id) REFERENCES clubs_trfrmrkt(id) ON DELETE CASCADE,
    FOREIGN KEY (competition_id) REFERENCES competitions_trfrmrkt(id) ON DELETE CASCADE);
-- Compare avg rating before vs after injury
SELECT p.name,
       AVG(CASE WHEN m.phase='before' THEN m.player_rating END) AS avg_before,
       AVG(CASE WHEN m.phase='after' THEN m.player_rating END)  AS avg_after
FROM player_injuries p
JOIN injury_matches m ON p.id = m.injury_id
GROUP BY p.name;
