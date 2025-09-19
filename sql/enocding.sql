create user 'himanshu@localhost' identified by 'yahoonet';
grant all privileges on *.* to 'himanshu@localhost';
flush privileges;

ALTER USER 'himanshu@localhost' IDENTIFIED WITH mysql_native_password BY 'yahoonet';
FLUSH PRIVILEGES;

update player_injuries_trfrmrkt set games_missed=0 where games_missed is null;
select * from player_injuries_trfrmrkt where end_date='1990-01-01'; -- placeholder for ongoing injuries
alter table player_transfer_history add column mv_raw varchar(50) after market_value;
alter table player_transfer_history add column fee_raw varchar(50) after fee;
create table player_transfer_history_18sep select * from player_transfer_history;
delete from player_transfer_history;
delete from clubs_trfrmrkt;
ALTER TABLE clubs_trfrmrkt AUTO_INCREMENT = 1;
insert into clubs_trfrmrkt (name) (select distinct club from transfermrkt_new18sep order by club);
alter table transfermrkt_new18sep add column clubid int, add column compid int;
update transfermrkt_new18sep p, clubs_trfrmrkt t set p.clubid=t.id where t.name = p.club;
update transfermrkt_new18sep p, competitions_trfrmrkt t set p.compid=t.id where t.name = p.competition;
select count(*) from transfermrkt_new18sep where clubid is null;
select count(*) from transfermrkt_new18sep where compid is null;

-- sentiment analysis for newly added players:
SELECT DISTINCT player FROM transfermrkt_new18sep where player not in (SELECT DISTINCT player FROM transfermrkt);
-- add new players to player list

SELECT null, t.transfermarkt_id, player FROM transfermrkt_new18sep t left outer join players_trfrmrkt p on t.transfermarkt_id=p.transfermarkt_id where p.transfermarkt_id is null;
SELECT t.transfermarkt_id FROM transfermrkt_new18sep t where transfermarkt_id not in (SELECT distinct p.transfermarkt_id FROM players_trfrmrkt p) limit 1000000;
select * from players_trfrmrkt where transfermarkt_id is null;
delete from players_trfrmrkt where transfermarkt_id is null; -- deletes "Maga" as no match found for this name to loopup for transfermarket id
select * from players_trfrmrkt p, transfermrkt_new18sep t where p.name = t.player and  p.transfermarkt_id <>t.transfermarkt_id ;
/*
Need to remove following duplicated name players as trafnsfermarkt and statsbomb data is being matched on the basis of names
 transfermarkt_id	 name
466805	 Nico González
607854	 Éderson
818495	 Otávio
964580	 Wesley
486031	 Nico González
238223	 Ederson
231289	 Otávio
1007378	 Wesley
*/
update players_trfrmrkt p, transfermrkt_new18sep t set p.transfermarkt_id =t.transfermarkt_id where p.name = t.player;
-- deleting same name but different id rows from players list
delete from players_trfrmrkt where transfermarkt_id in (486031, 238223, 231289, 1007378, 466805, 607854, 818495, 964580);
select count(*), player, transfermarkt_id from transfermrkt_new18sep group by player, transfermarkt_id having count(*)>1;
insert into players_trfrmrkt SELECT distinct null, t.transfermarkt_id, player FROM transfermrkt_new18sep t left outer join players_trfrmrkt p on t.transfermarkt_id=p.transfermarkt_id 
where p.transfermarkt_id is null and t.transfermarkt_id not in (486031, 238223, 231289, 1007378, 466805, 607854, 818495, 964580);
