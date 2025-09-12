
# WEEK 1


# Day 1 
Date: 18/08/2025 (Monday)

1. Perfect Data Illusion
- Discussed the concept of Perfect Data Illusion.
- Real-world data usually has missing values, noise, and outliers.
- Perfectly clean data can be a bad sign (over-processing, fabrication, or lack of variability).

2. GitHub Setup
- Joined the GitHub repository using mentor’s invitation.
- Created my personal branch `Abhay-raj`.
- Added a text file for daily updates.

3. Data Collection Exploration
- Explored StatsBomb Open Data for player performance datasets.
- Looked into Transfermarkt for player market value data.
- Searched initial sources for injury-related data.

4. Tomorrow’s Question
- What is "huge data" in machine learning?
- Is it absolute (millions of rows, GBs) or relative to the model/problem?
- Could small datasets also be “huge” depending on context?

--------------------------------------------------

# Day 2 
Date: 19/08/2025 (Tuesday)

1. Meeting Minutes (Recap from Mentor’s Mail)
- Agenda: Data Collection
- Discussed weekly goals.
- Each intern must update a text/docs file daily in their GitHub branch.
- Question for discussion given.

2. Daily Work
- Updated my branch with Day 1 report.
- Continued exploring StatsBomb Open Data.
- Reviewed Transfermarkt for player market values.

3. Tomorrow’s Question
- Suppose data comes from a single source (one device/location/group).  
- Is it reliable for a general model?  
- What risks arise if trained only on this source?  
- How might it behave on unseen data?  
- What should be done differently during data collection?  

--------------------------------------------------

# Day 3 
Date: 20/08/2025 (Wednesday)

1. Meeting Minutes (Recap from Mentor’s Mail)
- Agenda: Data Collection
- Mentor clarified doubts.
- Discussed yesterday’s question.
- New discussion question assigned.

2. Daily Work
- Updated my branch with Day 2 progress.
- Continued exploring StatsBomb datasets.
- Analyzed Transfermarkt structure for market values.

3. Tomorrow’s Question
- Which is more valuable for training ML models?  
  a) Large but narrow dataset (similar examples).  
  b) Smaller but diverse dataset (edge cases, variations).  
- Can a model trained on huge homogeneous data perform worse than one trained on fewer but varied samples?

--------------------------------------------------

# Day 4 
Date: 21/08/2025 (Thursday)

1. Meeting Minutes (Recap from Mentor’s Mail)
- Agenda: Data Collection.
- Discussed yesterday’s question (large vs. diverse data).
- New discussion question assigned.

2. Daily Work
- Updated my branch with Day 3 progress.
- Studied the importance of **data diversity vs. sheer size**.
- Key takeaway: Diversity is more crucial than just volume.
  - Large but homogeneous data → Overfits to repeated patterns.
  - Small but diverse data → Better generalization in real-world use cases.
- Explored **injury-related data sources** for player histories.

3. Tomorrow’s Question
- What strategies should be used while **collecting data** (not just preprocessing) to ensure representativeness while avoiding misleading biases?

--------------------------------------------------

# Day 5 
Date: 22/08/2025 (Friday)

1. Daily Work
- Explored **social sentiment data sources** (Twitter API, alternatives).
- Collected datasets:
  - **StatsBomb** (events, lineups, matches, three-sixty, competitions).
  - **Transfermarkt** (players, market valuations, transfers, clubs, appearances).
  - **Injury dataset** (historical records).
- Mentor assigned task to maintain **`dataset_info.md`** file documenting dataset sources and formats.

2. Mentor’s Questions
- What strategies would you use during data collection to ensure representativeness without making data misleading?
- Can you think of scenarios where artificially balancing the dataset may hurt model performance?
---------------

# WEEK 2


# Day 6 
Date: 25/08/2025 (Monday)

1. Daily Work
- Started **processing the collected data** for preparation.
- Mentor advised to push all code and related scripts into GitHub repository.

2. Tomorrow’s Question
- What are the different techniques used to handle **missing data**?
- How do we choose which technique is appropriate in different cases?

--------------------------------------------------

# Day 7 
Date: 26/08/2025 (Tuesday)

1. Daily Work
- Attempted to fetch data from **Twitter API**, but faced restrictions (rate limits, access issues).
- Switched to **Reddit API** as an alternative sentiment/engagement source.
- Worked on creating the **final_data.csv** file that will be the main input for ML model training.

2. Daily Question
- Continued discussion on techniques for handling **missing data** and criteria for choosing suitable methods.

--------------------------------------------------

# Day 8 
Date: 27/08/2025 (Wednesday)

- Marked as **Holiday (Ganesh Chaturthi)**.

--------------------------------------------------

# Day 9 
Date: 28/08/2025 (Thursday)

1. Daily Work
- Resumed after holiday.
- Continued work on **data preparation pipeline** for the ML model.
- Cleaned and standardized collected datasets (StatsBomb, Transfermarkt, injury records).
- Started aligning player IDs across different sources to ensure consistency.
- Reviewed sentiment data collection approach (Twitter + Reddit) for feasibility.

--------------------------------------------------

# Day 10 
Date: 29/08/2025 (Friday)

1. Meeting Minutes (MoM Recap)
- Agenda: Data Cleaning and Feature Engineering.
- Discussed updating GitHub with all Python scripts, daily progress, and dataset details.
- Emphasis on completing data collection, cleaning, and feature engineering by end of day.

2. Daily Work
- Updated GitHub repository with all Python files and dataset documentation.
- Continued work on **data collection module**.
- Started **data cleaning** and **feature engineering** tasks.
- Began exploring handling of **outliers**:
  - Noted that some outliers align with real-world scenarios.
  - Considered whether to remove them or retain for model realism.
- Reviewed strategy for handling **columns with both missing values and outliers**.

3. Tomorrow’s Question
- If your dataset contains outliers that reflect real-world behavior, should you remove them or keep them? Why?
- In columns with both missing values and outliers, which should be addressed first? Why?
------------

# WEEK 3


# Day 11
Date: 01/09/2025 (Monday)

1. Meeting Minutes (MoM Recap)
- Agenda: Advanced Feature Engineering and Sentiment Analysis.
- Discussed updating GitHub with all scripts and dataset progress.
- Focus for the week: **advanced feature engineering** and **sentiment analysis**.

2. Daily Work
- Continued **advanced feature engineering** on the collected datasets.
- Started integrating **sentiment analysis** pipeline for social media data (Twitter + Reddit).
- Reviewed **data cleaning, preprocessing, EDA, and feature engineering**:
  - Clarified differences between these terms:
    - **Cleaning:** Handling missing values, duplicates, and errors.
    - **Preprocessing:** Transforming raw data into model-ready format.
    - **EDA (Exploratory Data Analysis):** Understanding patterns, distributions, and correlations.
    - **Feature Engineering:** Creating meaningful features from raw data to improve model performance.

3. Tomorrow’s Question
- How do cleaning, preprocessing, EDA, and feature engineering differ, and why is each step important in the ML workflow?
- When integrating sentiment analysis features, how can you ensure they align correctly with player data from other sources?

--------------------------------------------------

# Day 12 
**Date:** 02/09/2025  

### Daily Work
- Worked on structuring the feature engineering module in Python.  
- Explored feature scaling techniques (Standardization vs Normalization).  
- Updated GitHub with cleaned dataset and preprocessing scripts.  

---

# Day 13 Progress Report
**Date:** 03/09/2025  

### Meeting Minutes (Recap from Mentor’s Mail)
- **Agenda:** Feature Engineering and Sentiment Analysis.  
- Discussed about outliers and when to treat missing values vs. outliers.  
- Mentor asked to complete feature engineering and sentiment analysis part.  

### Daily Work
- Implemented handling of outliers using IQR and Z-score methods.  
- Started correlation analysis between features.  
- Committed progress and updated GitHub repository.  

### Tomorrow’s Question
- What is multicollinearity? Is it always required to remove one of the columns or can we keep it?  
- What is correlation?

  ---

# Day 16 Progress Report
**Date:** 04/09/2025  

### Daily Work
- Continued correlation analysis and visualized feature relationships using heatmaps.  
- Implemented encoding for categorical variables (One-Hot Encoding and Label Encoding).  
- Pushed updated preprocessing scripts to GitHub.  

---

# Day 17 Progress Report
**Date:** 05/09/2025  

### Daily Work
- Completed feature engineering (handled skewness using log transformation).  
- Performed feature selection using correlation thresholding.  
- Updated `dataset_info.md` with final feature descriptions.  
- Began initial **Exploratory Data Analysis (EDA)** visualizations (distributions, boxplots, pairplots).  

---

# Day 18 Progress Report
**Date:** 08/09/2025  

### Daily Work
- Finalized **EDA** report (insights on player market values, injury impact, sentiment trends).  
- Started designing univariate LSTM model for time-series prediction.  
- Set up train-test split for temporal data.  

---

# Day 19 Progress Report
**Date:** 09/09/2025  

### Meeting Minutes (Recap from Mentor’s Mail)
- **Agenda:** Feature Engineering and Sentiment Analysis  
- Update the GitHub with all the Python files and progress.  
- Complete the work on data cleaning and feature engineering.  
- If preprocessing is complete, start with the modelling.  

### Daily Work
- Completed preprocessing and feature engineering module.  
- Implemented and trained first **Univariate LSTM model** for player market value prediction.  
- Evaluated performance using RMSE and MAE metrics.  

### Tomorrow’s Question
- What is PCA and LDA?  

---

# Day 20 Progress Report
**Date:** 10/09/2025  

### Meeting Minutes (Recap from Mentor’s Mail)
- **Agenda:** Feature Engineering and Sentiment Analysis  
- Discussed PCA basics and clarified:
  - Why PCA is used.  
  - Whether PCA can be applied before statistical analysis.  
  - Variance concepts (applied on features, not datapoints).  

### Daily Work
- Developed **Multivariate LSTM model** including multiple features (player stats, injuries, sentiment score).  
- Compared model performance against univariate version.  
- Documented results and updated `modelling_progress.md`.  

### Tomorrow’s Question
- Deep dive into PCA and LDA.  
