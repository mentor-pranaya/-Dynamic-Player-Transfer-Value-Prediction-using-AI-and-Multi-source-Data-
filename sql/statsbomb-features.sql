ALTER TABLE player_features 
ADD COLUMN season_id INT,
ADD COLUMN minutes_played INT,
ADD COLUMN shots_per90 FLOAT,
ADD COLUMN pressures_per90 FLOAT;
-- match_id column, imported from statsbomb data, is null, however the file_name column contains <matchid>.json
UPDATE lineups SET match_id = REGEXP_SUBSTR(file_name, '[0-9]+') WHERE match_id IS NULL;
UPDATE eventsnew1 SET match_id = REGEXP_SUBSTR(file_name, '[0-9]+') WHERE match_id IS NULL;

update player_features set minutes_played = 0 where minutes_played is null;
update player_features set shots_per90 = 0 where shots_per90 is null;
update player_features set pressures_per90 = 0 where pressures_per90 is null;
