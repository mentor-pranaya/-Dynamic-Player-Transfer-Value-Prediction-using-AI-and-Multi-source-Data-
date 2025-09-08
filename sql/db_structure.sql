-- List of clubs extracted from transfermrkt.com scraping data (normalized data)
CREATE TABLE `clubs_trfrmrkt` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=1551 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- List of competitions extracted from statsbomb data
CREATE TABLE `competitions` (
  `file_name` varchar(100) DEFAULT NULL,
  `id` int NOT NULL,
  `country_name` varchar(100) DEFAULT NULL,
  `competition_name` varchar(200) DEFAULT NULL,
  `competition_gender` varchar(20) DEFAULT NULL,
  `competition_youth` tinyint(1) DEFAULT NULL,
  `competition_international` tinyint(1) DEFAULT NULL,
  `match_updated` datetime DEFAULT NULL,
  `match_updated_360` datetime DEFAULT NULL,
  `match_available_360` datetime DEFAULT NULL,
  `match_available` datetime DEFAULT NULL,
  `season_id` int DEFAULT NULL,
  `season_name` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- List of competitions extracted from transfermrkt data  (normalized data)
CREATE TABLE `competitions_trfrmrkt` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=1551 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- List of events (initial) extracted from statsbomb data
CREATE TABLE `events` (
  `id` int NOT NULL AUTO_INCREMENT,
  `match_id` int DEFAULT NULL,
  `index_no` int DEFAULT NULL,
  `period` int DEFAULT NULL,
  `timestamp` varchar(20) DEFAULT NULL,
  `type` varchar(100) DEFAULT NULL,
  `player` varchar(100) DEFAULT NULL,
  `team` varchar(100) DEFAULT NULL,
  `location_x` float DEFAULT NULL,
  `location_y` float DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9386940 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- List of events (final) extracted from statsbomb data
CREATE TABLE `eventsNew1` (
  `id` varchar(50) NOT NULL,
  `file_name` varchar(100) DEFAULT NULL,
  `match_id` varchar(100) DEFAULT NULL,
  `index_no` int DEFAULT NULL,
  `period` int DEFAULT NULL,
  `timestamp` varchar(20) DEFAULT NULL,
  `type` varchar(100) DEFAULT NULL,
  `player` varchar(100) DEFAULT NULL,
  `team` varchar(100) DEFAULT NULL,
  `location_x` float DEFAULT NULL,
  `location_y` float DEFAULT NULL,
  `related_events` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data of injuries extracted from injury data csv (nomralized)
CREATE TABLE `injury_matches` (
  `id` int NOT NULL AUTO_INCREMENT,
  `injury_id` int DEFAULT NULL,
  `phase` enum('before','missed','after') DEFAULT NULL,
  `match_number` int DEFAULT NULL,
  `result` varchar(20) DEFAULT NULL,
  `opposition` varchar(100) DEFAULT NULL,
  `gd` int DEFAULT NULL,
  `player_rating` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `injury_id` (`injury_id`),
  CONSTRAINT `injury_matches_ibfk_1` FOREIGN KEY (`injury_id`) REFERENCES `player_injuries` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3937 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Lineup positions extracted from statsbomb data (Lineups folder)
CREATE TABLE `lineup_positions` (
  `id` int NOT NULL AUTO_INCREMENT,
  `player_id` int DEFAULT NULL,
  `team_id` int DEFAULT NULL,
  `position_id` int DEFAULT NULL,
  `position_name` varchar(100) DEFAULT NULL,
  `from_time` varchar(10) DEFAULT NULL,
  `to_time` varchar(10) DEFAULT NULL,
  `from_period` int DEFAULT NULL,
  `to_period` int DEFAULT NULL,
  `start_reason` varchar(100) DEFAULT NULL,
  `end_reason` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `player_id` (`player_id`),
  KEY `team_id` (`team_id`),
  CONSTRAINT `lineup_positions_ibfk_1` FOREIGN KEY (`player_id`) REFERENCES `players` (`player_id`),
  CONSTRAINT `lineup_positions_ibfk_2` FOREIGN KEY (`team_id`) REFERENCES `teams` (`team_id`)
) ENGINE=InnoDB AUTO_INCREMENT=130890 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Lineup positions extracted from statsbomb data (events folder)
CREATE TABLE `lineups` (
  `file_name` varchar(100) DEFAULT NULL,
  `comp_id` varchar(50) DEFAULT NULL,
  `id` int NOT NULL AUTO_INCREMENT,
  `match_id` int DEFAULT NULL,
  `team_id` int DEFAULT NULL,
  `team_name` varchar(100) DEFAULT NULL,
  `player_id` int DEFAULT NULL,
  `player_name` varchar(100) DEFAULT NULL,
  `position_id` int DEFAULT NULL,
  `position_name` varchar(100) DEFAULT NULL,
  `jersey_number` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=76209 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Transfermrkt market values data (normalized data)
CREATE TABLE `market_values_trfrmrkt` (
  `id` int NOT NULL AUTO_INCREMENT,
  `player_id` int DEFAULT NULL,
  `club_id` int DEFAULT NULL,
  `competition_id` int DEFAULT NULL,
  `market_value` bigint DEFAULT NULL,
  `snapshot_date` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `player_id` (`player_id`),
  KEY `club_id` (`club_id`),
  KEY `competition_id` (`competition_id`),
  CONSTRAINT `market_values_trfrmrkt_ibfk_1` FOREIGN KEY (`player_id`) REFERENCES `players_trfrmrkt` (`id`) ON DELETE CASCADE,
  CONSTRAINT `market_values_trfrmrkt_ibfk_2` FOREIGN KEY (`club_id`) REFERENCES `clubs_trfrmrkt` (`id`) ON DELETE CASCADE,
  CONSTRAINT `market_values_trfrmrkt_ibfk_3` FOREIGN KEY (`competition_id`) REFERENCES `competitions_trfrmrkt` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=1551 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- list of matches extracted from Statsbomb data
CREATE TABLE `matches` (
  `file_name` varchar(100) DEFAULT NULL,
  `match_id` int NOT NULL,
  `competition_id` int DEFAULT NULL,
  `season_id` int DEFAULT NULL,
  `match_date` date DEFAULT NULL,
  `home_team` varchar(100) DEFAULT NULL,
  `away_team` varchar(100) DEFAULT NULL,
  `home_score` int DEFAULT NULL,
  `away_score` int DEFAULT NULL,
  `stadium` varchar(100) DEFAULT NULL,
  `referee` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`match_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Social media sentiment analysis (Medium - incomplete)
CREATE TABLE `medium_sentiments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `player_name` varchar(100) DEFAULT NULL,
  `url` text,
  `url_hash` varchar(191) DEFAULT NULL,
  `title` text,
  `created_at` datetime DEFAULT NULL,
  `sentiment` varchar(16) DEFAULT NULL,
  `polarity` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `url_hash` (`url_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- List of player cards extracted from statsbomb (lineups folder)
CREATE TABLE `player_cards` (
  `id` int NOT NULL AUTO_INCREMENT,
  `player_id` int DEFAULT NULL,
  `team_id` int DEFAULT NULL,
  `card_time` varchar(10) DEFAULT NULL,
  `card_type` varchar(50) DEFAULT NULL,
  `reason` varchar(200) DEFAULT NULL,
  `period` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `player_id` (`player_id`),
  KEY `team_id` (`team_id`),
  CONSTRAINT `player_cards_ibfk_1` FOREIGN KEY (`player_id`) REFERENCES `players` (`player_id`),
  CONSTRAINT `player_cards_ibfk_2` FOREIGN KEY (`team_id`) REFERENCES `teams` (`team_id`)
) ENGINE=InnoDB AUTO_INCREMENT=14112 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- player injuries - normalized
CREATE TABLE `player_injuries` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  `team_name` varchar(100) DEFAULT NULL,
  `position` varchar(50) DEFAULT NULL,
  `age` int DEFAULT NULL,
  `season` varchar(20) DEFAULT NULL,
  `fifa_rating` int DEFAULT NULL,
  `injury` varchar(100) DEFAULT NULL,
  `date_of_injury` date DEFAULT NULL,
  `date_of_return` date DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=657 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Main csv file player injuries 
CREATE TABLE `player_injuries_impact` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  `team_name` varchar(100) DEFAULT NULL,
  `position` varchar(50) DEFAULT NULL,
  `age` int DEFAULT NULL,
  `season` varchar(20) DEFAULT NULL,
  `fifa_rating` int DEFAULT NULL,
  `injury` varchar(100) DEFAULT NULL,
  `date_of_injury` date DEFAULT NULL,
  `date_of_return` date DEFAULT NULL,
  `match1_before_injury_result` varchar(20) DEFAULT NULL,
  `match1_before_injury_opposition` varchar(100) DEFAULT NULL,
  `match1_before_injury_gd` int DEFAULT NULL,
  `match1_before_injury_player_rating` varchar(10) DEFAULT NULL,
  `match2_before_injury_result` varchar(20) DEFAULT NULL,
  `match2_before_injury_opposition` varchar(100) DEFAULT NULL,
  `match2_before_injury_gd` int DEFAULT NULL,
  `match2_before_injury_player_rating` varchar(10) DEFAULT NULL,
  `match3_before_injury_result` varchar(20) DEFAULT NULL,
  `match3_before_injury_opposition` varchar(100) DEFAULT NULL,
  `match3_before_injury_gd` int DEFAULT NULL,
  `match3_before_injury_player_rating` varchar(10) DEFAULT NULL,
  `match1_missed_match_result` varchar(20) DEFAULT NULL,
  `match1_missed_match_opposition` varchar(100) DEFAULT NULL,
  `match1_missed_match_gd` int DEFAULT NULL,
  `match2_missed_match_result` varchar(20) DEFAULT NULL,
  `match2_missed_match_opposition` varchar(100) DEFAULT NULL,
  `match2_missed_match_gd` int DEFAULT NULL,
  `match3_missed_match_result` varchar(20) DEFAULT NULL,
  `match3_missed_match_opposition` varchar(100) DEFAULT NULL,
  `match3_missed_match_gd` int DEFAULT NULL,
  `match1_after_injury_result` varchar(20) DEFAULT NULL,
  `match1_after_injury_opposition` varchar(100) DEFAULT NULL,
  `match1_after_injury_gd` int DEFAULT NULL,
  `match1_after_injury_player_rating` varchar(10) DEFAULT NULL,
  `match2_after_injury_result` varchar(20) DEFAULT NULL,
  `match2_after_injury_opposition` varchar(100) DEFAULT NULL,
  `match2_after_injury_gd` int DEFAULT NULL,
  `match2_after_injury_player_rating` varchar(10) DEFAULT NULL,
  `match3_after_injury_result` varchar(20) DEFAULT NULL,
  `match3_after_injury_opposition` varchar(100) DEFAULT NULL,
  `match3_after_injury_gd` int DEFAULT NULL,
  `match3_after_injury_player_rating` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=657 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- List of players from statsbomb data (lineups folder)
CREATE TABLE `players` (
  `player_id` int NOT NULL,
  `player_name` varchar(100) DEFAULT NULL,
  `player_nickname` varchar(100) DEFAULT NULL,
  `jersey_number` int DEFAULT NULL,
  `country_id` int DEFAULT NULL,
  `country_name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`player_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- List of players from Transfermrkt data scraped
CREATE TABLE `players_trfrmrkt` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=1551 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Social media sentiment analysis (Reddit - complete)
CREATE TABLE `reddit_sentiments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `player_name` varchar(100) DEFAULT NULL,
  `post_id` varchar(64) DEFAULT NULL,
  `subreddit` varchar(100) DEFAULT NULL,
  `created_at` datetime DEFAULT NULL,
  `title` text,
  `selftext` mediumtext,
  `sentiment` varchar(16) DEFAULT NULL,
  `polarity` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `post_id` (`post_id`)
) ENGINE=InnoDB AUTO_INCREMENT=258272 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- logging
CREATE TABLE `sentiment_run_log` (
  `id` int NOT NULL AUTO_INCREMENT,
  `run_started` datetime DEFAULT NULL,
  `run_finished` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- list of teams from statsbomb lineups folder
CREATE TABLE `teams` (
  `team_id` int NOT NULL,
  `team_name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`team_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Transfermrkt initial scraped data
CREATE TABLE `transfermrkt` (
  `id` int NOT NULL AUTO_INCREMENT,
  `player` varchar(100) DEFAULT NULL,
  `club` varchar(100) DEFAULT NULL,
  `competition` varchar(100) DEFAULT NULL,
  `market_value` bigint DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1551 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Social media sentiment analysis (Twitter - incomplete)
CREATE TABLE `twitter_sentiments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `player_name` varchar(100) DEFAULT NULL,
  `tweet_id` varchar(64) DEFAULT NULL,
  `created_at` datetime DEFAULT NULL,
  `content` text,
  `sentiment` varchar(16) DEFAULT NULL,
  `polarity` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `tweet_id` (`tweet_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS player_injuries_trfrmrkt (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT NOT NULL,
    transfermarkt_id INT NOT NULL,
    injury TEXT,
    start_date DATE,
    end_date DATE,
    games_missed INT,
    days_out INT,
    competition VARCHAR(200),
    FOREIGN KEY (player_id) REFERENCES players_trfrmrkt(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS player_transfer_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT,
    transfermarkt_id INT,
    season VARCHAR(20),
    transfer_date DATE,
    club_left VARCHAR(200),
    club_joined VARCHAR(200),
    market_value VARCHAR(50),
    fee BIGINT DEFAULT 0,
    reason VARCHAR(200) DEFAULT NULL,
    FOREIGN KEY (player_id) REFERENCES players_trfrmrkt(id) ON DELETE CASCADE
);
