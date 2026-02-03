"""
Advanced Feature Engineering for Player Valuation
================================================

This module implements advanced feature engineering techniques including:
- Performance trends over time (rolling averages, exponential moving averages, form scores)
- Injury impact features (games missed, injury counts, injury-adjusted metrics)
- Time-based aggregations and trend analysis

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering class for player performance and injury data.
    """
    
    def __init__(self, data_path: str = "../processed/dataset_processed.csv"):
        """
        Initialize the feature engineer with processed data.
        
        Args:
            data_path: Path to the processed dataset
        """
        self.data_path = data_path
        self.df = None
        self.features = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the processed dataset."""
        print("Loading processed dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def create_performance_trends(self, df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """
        Create performance trend features using rolling averages and exponential moving averages.
        
        Args:
            window_sizes: List of window sizes for rolling averages
            
        Returns:
            DataFrame with performance trend features
        """
        print("Creating performance trend features...")
        
        # Sort by player and year for proper time series
        df_sorted = self.df.sort_values(['p_id2', 'start_year']).copy()
        
        # Performance metrics to analyze
        performance_cols = [
            'season_minutes_played', 'season_games_played', 'season_days_injured',
            'pace', 'physic', 'fifa_rating'
        ]
        
        # Initialize result dataframe
        result_df = df_sorted.copy()
        
        for player in df_sorted['p_id2'].unique():
            player_mask = df_sorted['p_id2'] == player
            player_data = df_sorted[player_mask].copy()
            
            if len(player_data) < 2:  # Need at least 2 records for trends
                continue
                
            # Ensure we have numeric data for trend calculations
            for col in performance_cols:
                if col in player_data.columns:
                    player_data[col] = pd.to_numeric(player_data[col], errors='coerce')
                
            for col in performance_cols:
                if col not in player_data.columns:
                    continue
                    
                # Rolling averages
                for window in window_sizes:
                    # Ensure window is an integer
                    window = int(window)
                    if len(player_data) >= window:
                        try:
                            rolling_mean = player_data[col].rolling(window=window, min_periods=1).mean()
                            rolling_std = player_data[col].rolling(window=window, min_periods=1).std()
                            
                            result_df.loc[player_mask, f'{col}_roll{window}_mean'] = rolling_mean
                            result_df.loc[player_mask, f'{col}_roll{window}_std'] = rolling_std
                        except Exception as e:
                            print(f"Warning: Could not create rolling features for {col} with window {window}: {e}")
                            continue
                        
                # Rolling trend (slope of linear regression over window)
                if len(player_data) >= window:
                    trends = []
                    for i in range(len(player_data)):
                        if i < window - 1:
                            trends.append(np.nan)
                        else:
                            y = player_data[col].iloc[i-window+1:i+1].values
                            x = np.arange(len(y))
                            if len(y) > 1 and not np.isnan(y).all():
                                try:
                                    slope = np.polyfit(x, y, 1)[0]
                                    trends.append(slope)
                                except:
                                    trends.append(np.nan)
                            else:
                                trends.append(np.nan)
                    result_df.loc[player_mask, f'{col}_roll{window}_trend'] = trends
                
                # Exponential moving averages (recent games weighted more heavily)
                for alpha in [0.3, 0.5, 0.7]:  # Different smoothing factors
                    try:
                        ema = player_data[col].ewm(alpha=alpha, adjust=False).mean()
                        result_df.loc[player_mask, f'{col}_ema_{alpha}'] = ema
                    except Exception as e:
                        print(f"Warning: Could not create EMA for {col} with alpha {alpha}: {e}")
                        continue
                
                # Form score (z-score comparing last 5 matches vs season average)
                if len(player_data) >= 5:
                    season_avg = player_data[col].mean()
                    season_std = player_data[col].std()
                    
                    if season_std > 0:
                        # Last 5 games average
                        last_5_avg = player_data[col].rolling(window=5, min_periods=1).mean()
                        form_score = (last_5_avg - season_avg) / season_std
                        result_df.loc[player_mask, f'{col}_form_score'] = form_score
                    else:
                        result_df.loc[player_mask, f'{col}_form_score'] = 0
                else:
                    result_df.loc[player_mask, f'{col}_form_score'] = 0
        
        print("Performance trend features created successfully")
        return result_df
    
    def create_injury_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create injury impact features.
        
        Args:
            df: DataFrame with performance data
            
        Returns:
            DataFrame with injury impact features
        """
        print("Creating injury impact features...")
        
        result_df = df.copy()
        
        # Binary features for recent injury
        result_df['recent_injury_30_days'] = (result_df['season_days_injured'] > 0).astype(int)
        result_df['recent_injury_90_days'] = (result_df['season_days_injured'] > 30).astype(int)
        result_df['recent_injury_180_days'] = (result_df['season_days_injured'] > 90).astype(int)
        
        # Injury frequency features
        for player in df['p_id2'].unique():
            player_mask = df['p_id2'] == player
            player_data = df[player_mask].copy()
            
            if len(player_data) < 2:
                continue
                
            # Count of injury seasons
            injury_seasons = (player_data['season_days_injured'] > 0).sum()
            result_df.loc[player_mask, 'injury_seasons_count'] = injury_seasons
            
            # Average days injured per season
            avg_days_injured = player_data['season_days_injured'].mean()
            result_df.loc[player_mask, 'avg_days_injured_per_season'] = avg_days_injured
            
            # Injury severity (days injured as percentage of season)
            # Assuming 365 days in a season
            injury_severity = (player_data['season_days_injured'] / 365) * 100
            result_df.loc[player_mask, 'injury_severity_pct'] = injury_severity
            
            # Injury-adjusted performance metrics
            for col in ['season_minutes_played', 'season_games_played']:
                if col in player_data.columns:
                    # Performance per available day (excluding injured days)
                    available_days = 365 - player_data['season_days_injured']
                    available_days = available_days.replace(0, 1)  # Avoid division by zero
                    
                    injury_adjusted = player_data[col] / available_days
                    result_df.loc[player_mask, f'{col}_injury_adjusted'] = injury_adjusted
        
        # Injury risk score (combination of historical injury patterns)
        injury_risk_factors = [
            'injury_seasons_count', 'avg_days_injured_per_season', 
            'injury_severity_pct', 'season_days_injured'
        ]
        
        # Normalize each factor to 0-1 scale
        for factor in injury_risk_factors:
            if factor in result_df.columns:
                max_val = result_df[factor].max()
                if max_val > 0:
                    result_df[f'{factor}_normalized'] = result_df[factor] / max_val
                else:
                    result_df[f'{factor}_normalized'] = 0
        
        # Calculate composite injury risk score
        normalized_factors = [f'{f}_normalized' for f in injury_risk_factors if f'{f}_normalized' in result_df.columns]
        if normalized_factors:
            result_df['injury_risk_score'] = result_df[normalized_factors].mean(axis=1)
        else:
            result_df['injury_risk_score'] = 0
        
        print("Injury impact features created successfully")
        return result_df
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for trend analysis.
        
        Args:
            df: DataFrame with performance data
            
        Returns:
            DataFrame with time-based features
        """
        print("Creating time-based features...")
        
        result_df = df.copy()
        
        # Career stage features
        result_df['career_year'] = result_df['start_year'] - result_df.groupby('p_id2')['start_year'].transform('min')
        result_df['is_rookie'] = (result_df['career_year'] <= 2).astype(int)
        result_df['is_veteran'] = (result_df['career_year'] >= 8).astype(int)
        result_df['is_peak_age'] = ((result_df['age'] >= 25) & (result_df['age'] <= 29)).astype(int)
        
        # Year-over-year changes
        for col in ['season_minutes_played', 'season_games_played', 'fifa_rating', 'pace', 'physic']:
            if col in result_df.columns:
                result_df[f'{col}_yoy_change'] = result_df.groupby('p_id2')[col].pct_change()
                result_df[f'{col}_yoy_abs_change'] = result_df.groupby('p_id2')[col].diff()
        
        # Performance consistency (coefficient of variation)
        for player in df['p_id2'].unique():
            player_mask = df['p_id2'] == player
            player_data = df[player_mask].copy()
            
            if len(player_data) < 2:
                continue
                
            for col in ['season_minutes_played', 'season_games_played', 'fifa_rating']:
                if col in player_data.columns:
                    mean_val = player_data[col].mean()
                    std_val = player_data[col].std()
                    
                    if mean_val > 0:
                        cv = std_val / mean_val
                        result_df.loc[player_mask, f'{col}_consistency'] = cv
                    else:
                        result_df.loc[player_mask, f'{col}_consistency'] = 0
        
        print("Time-based features created successfully")
        return result_df
    
    def create_market_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market value related features.
        
        Args:
            df: DataFrame with performance data
            
        Returns:
            DataFrame with market value features
        """
        print("Creating market value features...")
        
        result_df = df.copy()
        
        # Load player valuations data
        try:
            valuations_df = pd.read_csv("../player_valuations.csv")
            valuations_df['date'] = pd.to_datetime(valuations_df['date'])
            valuations_df['year'] = valuations_df['date'].dt.year
            
            # Ensure data types match for merging
            valuations_df['player_id'] = valuations_df['player_id'].astype(str)
            result_df['p_id2'] = result_df['p_id2'].astype(str)
            valuations_df['year'] = valuations_df['year'].astype(int)
            result_df['start_year'] = result_df['start_year'].astype(int)
            
            # Merge with player data
            result_df = result_df.merge(
                valuations_df[['player_id', 'year', 'market_value_in_eur']], 
                left_on=['p_id2', 'start_year'], 
                right_on=['player_id', 'year'], 
                how='left'
            )
            
            # Market value features
            if 'market_value_in_eur' in result_df.columns:
                # Market value growth
                result_df['market_value_yoy_growth'] = result_df.groupby('p_id2')['market_value_in_eur'].pct_change()
                
                # Market value percentiles by position and age
                if 'position' in result_df.columns:
                    for position in result_df['position'].unique():
                        if pd.notna(position):
                            pos_mask = result_df['position'] == position
                            result_df.loc[pos_mask, 'market_value_percentile_by_position'] = result_df.loc[pos_mask, 'market_value_in_eur'].rank(pct=True)
                
                # Market value efficiency (performance per market value)
                for col in ['season_minutes_played', 'season_games_played', 'fifa_rating']:
                    if col in result_df.columns and 'market_value_in_eur' in result_df.columns:
                        result_df[f'{col}_per_market_value'] = result_df[col] / (result_df['market_value_in_eur'] + 1)
            
        except FileNotFoundError:
            print("Player valuations file not found, skipping market value features")
        
        print("Market value features created successfully")
        return result_df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Run all feature engineering steps.
        
        Returns:
            DataFrame with all engineered features
        """
        print("Starting advanced feature engineering...")
        
        # Load data
        df = self.load_data()
        
        # Create performance trends
        df = self.create_performance_trends(df, window_sizes=[3, 5, 10])
        
        # Create injury impact features
        df = self.create_injury_impact_features(df)
        
        # Create time-based features
        df = self.create_time_based_features(df)
        
        # Create market value features
        df = self.create_market_value_features(df)
        
        # Save intermediate results
        output_path = "../processed/advanced_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Advanced features saved to {output_path}")
        
        self.features = df
        return df
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of engineered features.
        
        Returns:
            Dictionary with feature summary
        """
        if self.features is None:
            print("No features available. Run engineer_all_features() first.")
            return {}
        
        feature_types = {
            'Performance Trends': [col for col in self.features.columns if any(x in col for x in ['roll', 'ema', 'form_score', 'trend'])],
            'Injury Impact': [col for col in self.features.columns if any(x in col for x in ['injury', 'recent_injury', 'injury_risk'])],
            'Time-based': [col for col in self.features.columns if any(x in col for x in ['yoy', 'consistency', 'career', 'rookie', 'veteran', 'peak'])],
            'Market Value': [col for col in self.features.columns if any(x in col for x in ['market_value', 'per_market_value'])]
        }
        
        summary = {
            'total_features': len(self.features.columns),
            'total_records': len(self.features),
            'feature_types': {k: len(v) for k, v in feature_types.items()},
            'feature_list': feature_types
        }
        
        return summary

def main():
    """Main function to run feature engineering."""
    print("=" * 60)
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 60)
    
    # Initialize feature engineer
    fe = AdvancedFeatureEngineer()
    
    # Engineer all features
    features_df = fe.engineer_all_features()
    
    # Get feature summary
    summary = fe.get_feature_summary()
    
    print("\nFeature Engineering Summary:")
    print("-" * 40)
    print(f"Total features: {summary['total_features']}")
    print(f"Total records: {summary['total_records']}")
    print("\nFeature breakdown:")
    for feature_type, count in summary['feature_types'].items():
        print(f"  {feature_type}: {count} features")
    
    print("\nAdvanced feature engineering completed successfully!")
    return features_df

if __name__ == "__main__":
    main()
