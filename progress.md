# Progress Report

## Day 1 (WEEK-1)
### Perfect Data Illusion
- Discussed the concept of **Perfect Data Illusion**.
- Real-world data usually has missing values, noise, and outliers.
- Perfectly clean data can be a **bad sign**:
  - Over-processing (loss of important signals).
  - Possible fabrication or manipulation.
  - Lack of real-world variability, leading to poor generalization.

### Creating a New Branch
- Created a new branch `Aman_Singh` for personal work without affecting main.

### Data Exploration: StatsBomb Football Data
- StatsBomb provides one of the largest open datasets for football analytics.
- Includes **event data** (passes, shots, tackles, dribbles) and **match data** (teams, players, results).
- Covers multiple leagues and competitions.
- Observed that the dataset is **huge**, causing performance issues on limited hardware.
- Handling requires:
  - Efficient file formats.
  - Loading data in chunks.
  - Cloud/Colab environments if local resources are insufficient.

### Tomorrow’s Question
- **What is "huge data"?**
  - Should it be defined as an absolute value (e.g., >1GB)?
  - Or does it depend on project requirements and resources?

---

## Day 2
### Discussion: Huge Data in Context of Project & Model
- "Huge" data is **relative**, not absolute.
- A dataset may be huge for a laptop but trivial for a server.
- In this project:
  - Huge = Data that **exceeds local compute/memory limits**.
  - Focus is on whether the data is **manageable for preprocessing and model training**.

### Data Exploration
- Explored **Market Value Data**: Transfermarkt data (scraping/other sources).

### Tomorrow’s Question
- **Is one data source enough to make a model generalize, or are multiple data sources better?**
- How does a model trained on single-source data behave in real-world scenarios?

---

## Day 3
### Discussion: One vs. Multiple Data Sources
- One data source:
  - Easier to manage and consistent.
  - Risk of **bias** and poor generalization in real-world cases.
- Multiple data sources:
  - Provide diversity and robustness.
  - Help capture varied scenarios → better real-world performance.
- Example: A player’s value estimated only from match stats may fail in real world if **market value sentiment, injuries, or contracts** are ignored.

### Data Exploration
- Explored **Social Media Sentiment**: Using Twitter API for sentiment analysis of player mentions.

### Tomorrow’s Question
- **Which generalizes a model better:**
  - Huge data with less diversity?
  - Small data with high diversity?

---

## Day 4
### Discussion: Huge vs. Diverse Data
- Diversity matters more than just volume.
- Huge data with low diversity → Overfits to repeated patterns, poor generalization.
- Small but diverse data → Captures variability, helps models adapt.
- Example: Face recognition trained on fewer diverse faces generalizes better than massive data of similar-looking faces.

### Data Exploration
- Explored **Injury Data**: Historical player injury records.


# Day 5 Progress

## Tasks Completed
1. **Exploring Sentiment Data Sources**
   - Looked into **Twitter API** and other possible APIs for capturing player sentiment from social media.  
   - Plan: integrate later once performance, injury, and market data pipeline is stable.  

2. **Data Collection Completed**
   - **StatsBomb data** (events, lineups, matches, three-sixty, competitions).  
   - **Transfermarkt data** (players, player valuations, transfers, appearances, clubs).  
   - **Injury dataset** (historical player injury records).  

3. **Mentor Task**
   - Mentor asked to maintain a new file: **`dataset_info.md`** documenting sources and formats.  

---

## Mentor’s Question  

### Q1: What strategies would you use while collecting data, not just later in preprocessing, to ensure your dataset is representative yet not misleading?

---

### Q2: Can you think of a scenario where artificially balancing the dataset might actually hurt the model’s real-world performance?

---

## Day-1 (WEEK-2)
### Work Done:
1. Started working on processing the collected data for preparation.
2. Mentor advised to put all code and related scripts on GitHub.

### Daily Question:
**Q:** What are the different techniques used to treat the missing data? How to choose which techniques to use?  

---

## Day-2
### Work Done:
1. Attempted to fetch data using Twitter API but faced limitations in access and rate restrictions.  
2. Switched to Reddit API as an alternative source for sentiment/engagement analysis.  
3. Worked on finalizing the **final_data.csv** which will serve as the main feed to the ML model.

### Daily Question:
**Q:** What are the different techniques used to treat the missing data? How to choose which techniques to use?

---

## Day-3 (Holiday)
Marked as a holiday due to **Ganesh Chaturthi**.

---

## Day-4
- Completed fetching **Reddit sentiments** using VADER.  
- Merged sentiment scores with `final_data.csv` → produced **`final_data_with_sentiment.csv`**.  
- Outcome: enriched base dataset with fan/community sentiment signals.

---

## Day-5
- Loaded **`final_data_with_sentiment.csv`** for feature engineering.  
- Performed preprocessing and feature engineering steps:
  - Numeric scaling, categorical encoding, sentiment integration.  
  - Created **rolling seasonal trend features** (1 & 3 seasons).  
  - Built **injury decay scores** (recent injuries weigh more).  
  - Added **contract buckets** based on `days_to_contract_end`.  
  - Constructed **sentiment × performance interaction terms**.  
  - Removed **76 high-correlation features** to avoid redundancy.  
- Saved **`features_augmented.parquet`** as the final feature set.  
- Outcome: obtained a clean, enriched dataset (1272 rows × 226 features) ready to feed into modeling.

---

**Artifacts produced so far**
- `final_data_with_sentiment.csv`
- `features_augmented.parquet`
- `high_corr_drop_candidates.csv` (audit list of dropped features)

---

## Day-1(WEEK-3)
**Tasks:**
1. Performed EDA on the featured data.  

**Questions:**
- You detect outliers in your dataset, but they seem to align with real-world scenarios. Do you remove them? Why or why not?  
  - Outliers that represent genuine variations in real-world behavior should not be removed, as they may contain valuable information. They are only removed if they are errors, noise, or data entry mistakes.  

- If your dataset has both missing values and outliers in the same column, what would you handle first and why?  
  - Missing values are handled first because imputation methods might depend on the data distribution. Removing/adjusting outliers before fixing missing values can bias the imputation process.  

---

## Day-2
**Tasks:**
1. Found more valuable features in the raw dataset and re-created the featured dataset with additional columns.  

**Questions:**
- We use the terms cleaning, pre-processing, EDA, and feature engineering — do they mean the same or different? Why or why not?  
  - They are related but not the same:  
    - **Cleaning**: Removing inconsistencies, duplicates, or incorrect values.  
    - **Pre-processing**: Transforming raw data into a usable form (scaling, encoding, imputing).  
    - **EDA (Exploratory Data Analysis)**: Understanding patterns, trends, and relationships in data.  
    - **Feature Engineering**: Creating new features or modifying existing ones to improve model performance.  

---

## Day-3
**Tasks:**
1. Performed advanced EDA on the newly featured dataset to prepare data for LSTM models.  

**Questions:**
- What is multicollinearity? Is it always required to remove one of the columns or keeping the column is good?  
  - **Multicollinearity** occurs when two or more features are highly correlated, leading to redundancy and unstable model coefficients.  
  - It is not always required to remove columns; tree-based models can handle it, but linear models may suffer. Decision depends on the model type and performance impact.  

- What is correlation?  
  - **Correlation** measures the linear relationship between two variables (ranges from -1 to +1). Positive correlation means they increase together, negative means one decreases as the other increases, and zero means no linear relation.  

---

## Day-4
**Tasks:**
1. Prepared univariate and multivariate models for experimentation.  

---

## Day-5
**Tasks:**
1. Implemented encoder-decoder LSTM and saved all deliverables.  

**Questions:**
- How to handle imbalanced data?   

- What is PCA and LDA?   