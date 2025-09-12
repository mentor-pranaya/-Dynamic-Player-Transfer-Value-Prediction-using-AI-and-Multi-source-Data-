# 🗣️ Sentiment Insights – Player Value Prediction Project

## Context
We integrated sentiment features from Reddit (VADER/TextBlob), including:
- `reddit_num_posts`, `reddit_num_comments_used`
- `reddit_pos_ratio`, `reddit_neg_ratio`, `reddit_neu_ratio`
- `reddit_mean_compound` (compound sentiment score)
- `reddit_subreddits_count` (breadth of subreddit coverage)
- Missingness flags and scaled versions

## Key Findings
1. **Weak linear correlations with target**
   - Correlations with `target_log1p` were small:  
     - `reddit_num_posts`, `reddit_neu_ratio` ≈ +0.04  
     - `reddit_num_comments_used` ≈ +0.039  
     - `reddit_mean_compound` ≈ +0.018  
     - Negativity ratio and other features ≈ near zero
   - Indicates sentiment has **minor direct impact** on market value.

2. **Scatter plot patterns**
   - Sentiment scatter plots vs. `target_log1p` showed wide dispersion with no clear linear trend.  
   - Suggests sentiment influences are noisy and not strongly predictive on their own.

3. **Model importance**
   - In safe LightGBM runs, sentiment features ranked low compared to performance (goals, minutes), transfers, and demographics.  
   - They contribute marginally when combined with other features.

## Interpretation
- Public sentiment correlates only weakly with player valuations.  
- Sentiment likely reflects **popularity/visibility** rather than direct value drivers.  
- While not strong standalone predictors, sentiment features enrich the dataset by capturing **off-pitch factors** that traditional stats miss.

## Conclusion
Sentiment analysis milestone is complete:
- Features engineered and included in FE dataset.  
- Correlation/importance analysis performed.  
- Sentiment adds **weak but diverse signal**, and will remain in the safe feature set for modeling.  
