-- verifying new fuzzy logic to add mappings including injury data (mapping via python)
use AIProject;
select count(*) from player_mapping_old where trfrmrkt_player_id is not null;
rename table player_mapping to player_mapping_old;  
CREATE TABLE IF NOT EXISTS player_mapping (
    id INT AUTO_INCREMENT PRIMARY KEY,
    canonical_name VARCHAR(100) UNIQUE,
    trfrmrkt_player_id INT,
    statsbomb_player_id INT,
    injury_player_id INT,
    INDEX idx_trf (trfrmrkt_player_id),
    INDEX idx_sb (statsbomb_player_id),
    INDEX idx_inj (injury_player_id)
);
select * from player_mapping where injury_player_id is not null;
select * from player_mapping where trfrmrkt_player_id = 1533;
select distinct(player) from transfermrkt order by player limit 1000000;
select distinct(player_name) from players order by player_name limit 1000000;
select distinct(player) from transfermrkt where player not in (
select distinct(player_name) from players order by player_name );
select * from player_mapping p, players_trfrmrkt t where t.id = p.trfrmrkt_player_id;
SELECT p.player_id, p.player_name FROM players p
LEFT JOIN player_mapping pm ON p.player_id = pm.statsbomb_player_id
WHERE pm.statsbomb_player_id IS NULL order by p.player_name limit 100000;
-- checking if new logic missed any previous values and / or added new values
select * from player_mapping_old where trfrmrkt_player_id not in (select trfrmrkt_player_id from player_mapping);
select * from player_mapping where trfrmrkt_player_id not in (select trfrmrkt_player_id from player_mapping_old);
select * from player_mapping_old where trfrmrkt_player_id not in (select trfrmrkt_player_id from player_mapping where statsbomb_player_id is not null);
select * from player_mapping where statsbomb_player_id is not null and trfrmrkt_player_id not in (select trfrmrkt_player_id from player_mapping_old);

SELECT t.id, t.name FROM players_trfrmrkt t 
left JOIN player_mapping pm ON t.id = pm.trfrmrkt_player_id
WHERE pm.trfrmrkt_player_id IS NULL order by t.name limit 100000;

select * from player_mapping_old p, players t where t.player_id = p.statsbomb_player_id;
select * from player_mapping p, players t where t.player_id = p.statsbomb_player_id;

select * from reddit_sentiments;
alter table reddit_sentiments add column player_id_trfrmrkt int;
update reddit_sentiments r, players_trfrmrkt t set r.player_id_trfrmrkt =t.id where r.player_name = t.name;
select * from player_injuries r, players_trfrmrkt t where r.name = t.name;
select count(*), r.name from player_injuries r, players_trfrmrkt t where r.name = t.name group by r.name order by count(*) desc;
select * from player_injuries r, players_trfrmrkt t where r.name = t.name and r.id not in (select injury_player_id from player_mapping);
select * from player_injuries;
select distinct(r.name) from player_injuries r, players t where r.name = t.player_name;
select distinct(r.name) from player_injuries r, players t,  players_trfrmrkt t1  where r.name = t1.name and r.name = t.player_name;
select * from player_injuries r, players_trfrmrkt t, players p where (r.name = p.player_name or r.name = t.name) and p.player_name <> t.name;
