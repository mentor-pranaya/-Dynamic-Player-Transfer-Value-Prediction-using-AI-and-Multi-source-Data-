create database AIProject;
use AIProject;
show tables;
desc events;
select * from events;
select count(*) as NetCnt from events union
select count(*) as NoLocationCnt from events where location_x=0;
