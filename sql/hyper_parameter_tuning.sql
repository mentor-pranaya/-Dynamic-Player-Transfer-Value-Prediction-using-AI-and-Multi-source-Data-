-- MySQL dump 10.13  Distrib 8.0.43, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: aiproject
-- ------------------------------------------------------
-- Server version	8.0.43

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `hyper_parameter_results`
--

DROP TABLE IF EXISTS `hyper_parameter_results`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `hyper_parameter_results` (
  `id` int NOT NULL AUTO_INCREMENT,
  `run_date` datetime DEFAULT (now()),
  `n_steps` int DEFAULT NULL,
  `n_future` int DEFAULT NULL,
  `LSTM_Iterations` int DEFAULT NULL,
  `LSTM_Epoch_Count` int DEFAULT NULL,
  `XGBoost_Iterations` int DEFAULT NULL,
  `Best_XGBoost_params` varchar(200) DEFAULT NULL,
  `XGBoost_RSME` float DEFAULT NULL,
  `Best_LSTM_params` varchar(200) DEFAULT NULL,
  `Val_Loss` float DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `hyper_parameter_results`
--

LOCK TABLES `hyper_parameter_results` WRITE;
/*!40000 ALTER TABLE `hyper_parameter_results` DISABLE KEYS */;
INSERT INTO `hyper_parameter_results` VALUES (1,'2025-10-06 10:49:13',3,3,20,11,20,'{\'n_estimators\': 300, \'max_depth\': 6, \'learning_rate\': 0.01, \'subsample\': 0.8, \'colsample_bytree\': 1.0}',0.2368,' {\'latent_dim\': 128, \'batch_size\': 16, \'learning_rate\': 0.001}',0.0689),(3,'2025-10-06 11:08:02',3,3,21,11,21,'{\'n_estimators\': 100, \'max_depth\': 4, \'learning_rate\': 0.05, \'subsample\': 0.8, \'colsample_bytree\': 0.7}',0.234336,'{\'latent_dim\': 128, \'batch_size\': 16, \'learning_rate\': 0.01}',0.0688472),(4,'2025-10-06 11:14:53',3,3,21,50,21,'{\'n_estimators\': 500, \'max_depth\': 4, \'learning_rate\': 0.01, \'subsample\': 0.7, \'colsample_bytree\': 0.8}',0.235332,'{\'latent_dim\': 64, \'batch_size\': 32, \'learning_rate\': 0.005}',0.0662888),(5,'2025-10-06 11:24:46',3,3,21,50,21,'{\'n_estimators\': 300, \'max_depth\': 4, \'learning_rate\': 0.01, \'subsample\': 1.0, \'colsample_bytree\': 0.8}',0.234058,'{\'latent_dim\': 128, \'batch_size\': 16, \'learning_rate\': 0.005}',0.0663276),(6,'2025-10-06 17:47:10',3,3,10,10,10,'{\'n_estimators\': 300, \'max_depth\': 4, \'learning_rate\': 0.01, \'subsample\': 1.0, \'colsample_bytree\': 0.7}',0.233589,'{\'latent_dim\': 32, \'batch_size\': 16, \'learning_rate\': 0.01}',0.0703875),(7,'2025-10-06 22:01:24',3,3,14,10,21,'{\'n_estimators\': 300, \'max_depth\': 4, \'learning_rate\': 0.01, \'subsample\': 1.0, \'colsample_bytree\': 1.0}',0.233005,'{\'latent_dim\': 128, \'batch_size\': 16, \'learning_rate\': 0.001}',0.0693997);
/*!40000 ALTER TABLE `hyper_parameter_results` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-10-11 18:39:35
