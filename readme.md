**18 August 2025:**
----------------------------------------------------------------------------------------------------------

  Explored various data sorces

  Downloaded data from
  https://github.com/statsbomb/open-data

**19 August 2025:**
----------------------------------------------------------------------------------------------------------

  Working on python script to scrape data using BS Library.
  
**20 August 2025:**
----------------------------------------------------------------------------------------------------------

  Still working on python script to scrape data
  
**26 August 2025:**
----------------------------------------------------------------------------------------------------------  

  Completed testing first script transfermarkt_premier-league_initial.py for web scrping

  Created new script transfermarkt_loop.py to loop through pagination of table to add all the players from given set of Leagues.

  Completed scraping following URLs for data:
  https://www.transfermarkt.com/premier-league/marktwerte/wettbewerb/GB1
  https://www.transfermarkt.com/laliga/marktwerte/wettbewerb/ES1
  https://www.transfermarkt.com/bundesliga/marktwerte/wettbewerb/L1
  https://www.transfermarkt.com/serie-a/marktwerte/wettbewerb/IT1
  https://www.transfermarkt.com/ligue-1/marktwerte/wettbewerb/FR1
  https://www.transfermarkt.com/eredivisie/marktwerte/wettbewerb/NL1
  https://www.transfermarkt.com/super-lig/marktwerte/wettbewerb/TR1
  https://www.transfermarkt.com/saudi-professional-league/marktwerte/wettbewerb/SA1
  https://www.transfermarkt.com/uefa-champions-league/marktwerte/pokalwettbewerb/CL
  https://www.transfermarkt.com/europa-league/marktwerte/pokalwettbewerb/EL
  https://www.transfermarkt.com/uefa-europa-conference-league/marktwerte/pokalwettbewerb/UCOL

  ----------------------------------------------------------------------------------------------------------  
  Total Players Scraped Across All Competitions: 1550
  ----------------------------------------------------------------------------------------------------------

**27 Aug 2025:**
----------------------------------------------------------------------------------------------------------  

  Downloaded Player Injuries and Team Performance Dataset from Kaggle:

      This dataset investigates the impact of player injuries on team performance across seven Premier League clubs from 2019 to 2023, including Tottenham, Aston Villa, Brighton, Arsenal, Brentford, Everton, Burnley, and Manchester City. The dataset contains over 600 injury records, offering insights into how player absences influence match results and individual performance metrics.

**28 Aug 2025:**
----------------------------------------------------------------------------------------------------------  
   Working on MySQL DB to import various dataset records for easy quering, matching and cleaning of records.

   Created database "AIProject" on localhost
   importing statsbomb data to mysql database

**29 Aug 2025:**
----------------------------------------------------------------------------------------------------------  
    Working on python script to import data to mysql

    Stopped data import to add file_name column to data being imported from json to mysql for cross referencing.
**30th Aug 2025:**
----------------------------------------------------------------------------------------------------------  
  Completed importing 3464 files from events folder from StatsBomb data

  Total rows imported : 12188949

**31th Aug 2025:**
----------------------------------------------------------------------------------------------------------  
  Created new script to import statsbomb data, along with lineup data from events json files

  Created batch update file to create a new events table for every 100 files, that could be merged later into 1 table as inserting in single file was taking long time.

  Imported Transfermrkt and Injury data to MySQL

  Imported teams, players, lineup_positions, player_cards from "Lineups" data folder

  Imported Normalized data from Transfermrkt to players_trfrmrkt, clubs_trfrmrkt, competitions_trfrmrkt, market_values_trfrmrkt

**1st Sep 2025:**
----------------------------------------------------------------------------------------------------------  
  Sentiment analysis: Tried multithreading to capture data from multiple platforms. Failed repeatedly with multiple options to analyze Twitter data, along with Reddit and Medium
  
  Seperated the three social media data collection logic.
  
  Reddit data is being uploaded to MySQL 

**2nd Sept 2025:**
----------------------------------------------------------------------------------------------------------  
  Completed Reddit sentiment analysis 

  Merging all events data (created different tables earlier for each 100 files of 3464 files from StatsBomb data for faster processing) 

**4th Sept 2025:**
----------------------------------------------------------------------------------------------------------  
  Added the final schema (db_structure.sql) to git repository

**5th - 7th Sep 2025**
----------------------------------------------------------------------------------------------------------  
  Worked on new mapping scripts to improve mapping between statsbomb, transfermrkt and injury data

**8th Sep 2025**
----------------------------------------------------------------------------------------------------------  
  Improved GIT structure

  Player injury data file has too few matches
 
  Updated transfermarkt scraping script to fetch player id from site

  Scraping transfermarkt for injury data

  Also adding player transfer history data via scraping from player profile page on transfermarkt 

**9th to 12th September 2025:**
----------------------------------------------------------------------------------------------------------  
  Worked on data cleaning 
  
  while cleaning, transfermarkt injuries scraped data was found to be inconsistent.

  Rescraped injuries data, with pagination support and formating for multipage injury data from trasnfermarkt

**13th - 15th September 2025:**
----------------------------------------------------------------------------------------------------------  
  Created scripts for feature merging
  
  Working on visualizations for further feature engineering

**16th September 2025:**
----------------------------------------------------------------------------------------------------------  
  Created fresh new script for scraping player transfers from trasnfermarkt
  
  Using selemium to scrape data from 
  
    https://www.transfermarkt.com/spieler/transfers/spieler/{player_id}
  
  in loop, using new script scrape_trfr_record_new.py
  
**17th September 2025:**
----------------------------------------------------------------------------------------------------------  
  updated scraping script to scroll and retry on not finding the transfer grid, as the code sometimes fails before the grid loads

  Also found that the lazy load script would not let the grid load at times, till the page is scrolled till the grid, added scroll
  
  
