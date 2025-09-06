# 📊 EDA Summary – Player Value Prediction Project

**Project Context**  
- Goal: Predict football players’ market value (`target_log1p`) per season.  
- Data sources: Transfermarkt (valuations, transfers), StatsBomb (performance stats), Reddit (sentiment), injury records, player metadata.  
- Current dataset: `final_training_master_fe_safe.csv` (87,223 rows × ~140 safe features).  

---

## ✅ Key Findings from EDA

### 1. Target distribution
- `target_log1p` is continuous, well-defined, no missing values.  
- Distribution is slightly skewed, but log1p transform stabilizes it.  

### 2. Leakage detection
- Initial feature importance was dominated by valuation-derived features (`last_value_log1p`, `value_median_log1p`, etc.).  
- These were highly correlated with the target → **leakage risk**.  
- **Action taken:** Dropped all valuation-derived features, keeping only safe lagged versions.  

### 3. Safe feature set
- Final safe dataset created: `final_training_master_fe_safe.csv`.  
- Safe feature list saved separately: `safe_features_list.csv`.  
- ~140 features remain, covering:
  - Demographics: age, height, foot, position.  
  - Performance: appearances, minutes, goals, assists, per90 stats.  
  - Transfers: counts, fees, days since last transfer.  
  - Injuries: count, avg days out.  
  - Sentiment: Reddit features.  
  - Engineered: interactions (e.g., `age_x_mins`, `goals_per_min`), lag features (`goals_lag1`, `assists_lag1`, etc.).  

### 4. Baseline modeling insights
- **Leaky run RMSE:** ~0.01 (unrealistically low).  
- **Safe run RMSE:** ~0.79 (realistic baseline).  
- Top safe predictors included:
  - `appearances_count`, `total_minutes`, `goals`, `assists`.  
  - `transfers_sum_fee`, `transfers_max_fee`, `days_since_last_transfer`.  
  - `age_at_season_start`.  
- Artifacts like `snapshots_count` and raw IDs were removed to avoid hidden leakage.  

---

## 📂 Outputs Saved
- `sandbox/final_training_master_fe.csv` → full FE dataset (185 cols).  
- `sandbox/final_training_master_fe_safe.csv` → safe FE dataset (~140 cols).  
- `sandbox/safe_features_list.csv` → list of modeling features.  
- `sandbox/diagnostics/` → missingness CSV, correlations, target stats, feature importance CSVs, and plots.  

---

## 🎯 Next Steps
- **Milestone 4 – Modeling**
  - Define proper **time-aware train/validation splits** (train ≤ season X, test > season X).  
  - Train baseline models (Linear, RandomForest, XGBoost, LightGBM).  
  - Evaluate performance using RMSE and feature importance.  
  - Tune hyperparameters for best generalization.  
- **Feature refinement loop**
  - Consider log-transforming heavy-skewed features (fees, minutes).  
  - Handle categorical encodings more robustly (e.g., target encoding for competitions/clubs).  
  - Evaluate Reddit sentiment contribution.  

---

✅ With this, **EDA milestone is completed**.  
