create database AIProject;
use AIProject;
show tables;
/*desc events;
select * from events;
select count(*) as NetCnt from events union
select count(*) as NoLocationCnt from events where location_x=0;
Created new table events1 with file_name column*/
desc events1;
select * from events1;
select count(*) as NetCnt from events1 union
select count(*) as NoLocationCnt from events1 where location_x=0;
select count(*) as FileCnt, 3464-count(*) as FileCntRem from (;
select count(*) as FileRecCnt from events1 group by file_name;
) as t;
select count(*) from (select distinct(file_name) from events1) as t;
SELECT TABLE_NAME AS `Table`, ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) AS `Size (MB)`
FROM information_schema.TABLES WHERE TABLE_SCHEMA = 'AIProject'; -- AND TABLE_NAME = 'your_table_name';
select * from events1 where player is not null;
drop table eventsNew;
drop table lineups;
select * from eventsNew;
select * from lineups;
