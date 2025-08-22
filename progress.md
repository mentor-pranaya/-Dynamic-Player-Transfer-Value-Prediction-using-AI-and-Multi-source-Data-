# Progress Report

## Day 1
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
