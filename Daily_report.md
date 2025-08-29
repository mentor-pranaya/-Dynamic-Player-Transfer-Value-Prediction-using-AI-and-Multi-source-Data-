-------------------------------------------

# Day 1 Progress Report
Date: 18/08/2025

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

# Day 2 Progress Report
Date: 19/08/2025

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

# Day 3 Progress Report
Date: 20/08/2025

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

# Day 4 Progress Report
Date: 21/08/2025

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

# Day 5 Progress Report
Date: 22/08/2025

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

--------------------------------------------------

# Day 6 Progress Report
Date: 25/08/2025

1. Daily Work
- Started **processing the collected data** for preparation.
- Mentor advised to push all code and related scripts into GitHub repository.

2. Tomorrow’s Question
- What are the different techniques used to handle **missing data**?
- How do we choose which technique is appropriate in different cases?

--------------------------------------------------

# Day 7 Progress Report
Date: 26/08/2025

1. Daily Work
- Attempted to fetch data from **Twitter API**, but faced restrictions (rate limits, access issues).
- Switched to **Reddit API** as an alternative sentiment/engagement source.
- Worked on creating the **final_data.csv** file that will be the main input for ML model training.

2. Daily Question
- Continued discussion on techniques for handling **missing data** and criteria for choosing suitable methods.

--------------------------------------------------

# Day 8 Progress Report
Date: 27/08/2025

- Marked as **Holiday (Ganesh Chaturthi)**.

--------------------------------------------------
