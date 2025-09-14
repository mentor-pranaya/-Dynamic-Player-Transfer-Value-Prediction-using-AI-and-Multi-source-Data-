"""
Generate Final Features for Player Valuation
===========================================

This module combines all engineered features (performance, injury, sentiment) 
into a final feature set for player valuation modeling.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FinalFeatureGenerator:
    """
    Class to combine all features into a final feature set.
    """
    
    def __init__(self):
        """Initialize the feature generator."""
        self.base_features = None
        self.advanced_features = None
        self.sentiment_features = None
        self.final_features = None
        
    def load_base_features(self, path: str = "../processed/dataset_processed.csv") -> pd.DataFrame:
        """
        Load base processed features.
        
        Args:
            path: Path to base features file
            
        Returns:
            DataFrame with base features
        """
        print("Loading base features...")
        try:
            self.base_features = pd.read_csv(path)
            print(f"Loaded {len(self.base_features)} records with {len(self.base_features.columns)} features")
            return self.base_features
        except FileNotFoundError:
            print(f"Base features file not found: {path}")
            return pd.DataFrame()
    
    def load_advanced_features(self, path: str = "../processed/advanced_features.csv") -> pd.DataFrame:
        """
        Load advanced engineered features.
        
        Args:
            path: Path to advanced features file
            
        Returns:
            DataFrame with advanced features
        """
        print("Loading advanced features...")
        try:
            self.advanced_features = pd.read_csv(path)
            print(f"Loaded {len(self.advanced_features)} records with {len(self.advanced_features.columns)} features")
            return self.advanced_features
        except FileNotFoundError:
            print(f"Advanced features file not found: {path}")
            return pd.DataFrame()
    
    def load_sentiment_features(self, path: str = "../processed/player_sentiment_features.csv") -> pd.DataFrame:
        """
        Load sentiment features.
        
        Args:
            path: Path to sentiment features file
            
        Returns:
            DataFrame with sentiment features
        """
        print("Loading sentiment features...")
        try:
            self.sentiment_features = pd.read_csv(path)
            print(f"Loaded {len(self.sentiment_features)} records with {len(self.sentiment_features.columns)} features")
            return self.sentiment_features
        except FileNotFoundError:
            print(f"Sentiment features file not found: {path}")
            return pd.DataFrame()
    
    def align_features_by_player(self) -> pd.DataFrame:
        """
        Align all features by player and time period.
        
        Returns:
            DataFrame with aligned features
        """
        print("Aligning features by player...")
        
        if self.base_features is None or self.base_features.empty:
            print("No base features available")
            return pd.DataFrame()
        
        # Start with base features
        result_df = self.base_features.copy()
        
        # Merge advanced features
        if self.advanced_features is not None and not self.advanced_features.empty:
            # Find common columns to merge on
            merge_cols = ['p_id2', 'start_year']
            common_cols = [col for col in merge_cols if col in self.advanced_features.columns]
            
            if common_cols:
                # Get columns that are not in base features
                advanced_cols_to_add = [col for col in self.advanced_features.columns 
                                      if col not in result_df.columns]
                
                if advanced_cols_to_add:
                    advanced_subset = self.advanced_features[common_cols + advanced_cols_to_add]
                    result_df = result_df.merge(advanced_subset, on=common_cols, how='left')
                    print(f"Added {len(advanced_cols_to_add)} advanced features")
        
        # Merge sentiment features
        if self.sentiment_features is not None and not self.sentiment_features.empty:
            # Map player names to IDs (this is a simplified approach)
            # In practice, you'd need a proper player name to ID mapping
            if 'player_name' in self.sentiment_features.columns:
                # For now, we'll create a simple mapping based on available data
                # This is a placeholder - in practice, you'd need proper name matching
                sentiment_mapping = {}
                for idx, row in self.sentiment_features.iterrows():
                    player_name = row['player_name']
                    # Simple mapping - in practice, use fuzzy matching or lookup table
                    if player_name in result_df['p_id2'].values:
                        sentiment_mapping[player_name] = player_name
                
                # Add sentiment features for matched players
                sentiment_cols = [col for col in self.sentiment_features.columns 
                                if col != 'player_name']
                
                for player_name, player_id in sentiment_mapping.items():
                    player_sentiment = self.sentiment_features[
                        self.sentiment_features['player_name'] == player_name
                    ][sentiment_cols].iloc[0]
                    
                    # Add to all records for this player
                    player_mask = result_df['p_id2'] == player_id
                    for col in sentiment_cols:
                        result_df.loc[player_mask, f'sentiment_{col}'] = player_sentiment[col]
                
                print(f"Added sentiment features for {len(sentiment_mapping)} players")
        
        print(f"Final dataset has {len(result_df)} records with {len(result_df.columns)} features")
        return result_df
    
    def create_feature_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features by type.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature categories
        """
        feature_categories = {
            'Basic Demographics': [],
            'Performance Metrics': [],
            'Injury Features': [],
            'Performance Trends': [],
            'Injury Impact': [],
            'Time-based Features': [],
            'Market Value Features': [],
            'Sentiment Features': [],
            'Other': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(x in col_lower for x in ['age', 'height', 'weight', 'nationality', 'position', 'dob']):
                feature_categories['Basic Demographics'].append(col)
            elif any(x in col_lower for x in ['minutes', 'games', 'rating', 'pace', 'physic', 'fifa']):
                feature_categories['Performance Metrics'].append(col)
            elif any(x in col_lower for x in ['injury', 'injured', 'days']):
                feature_categories['Injury Features'].append(col)
            elif any(x in col_lower for x in ['roll', 'ema', 'form_score', 'trend']):
                feature_categories['Performance Trends'].append(col)
            elif any(x in col_lower for x in ['injury_risk', 'recent_injury', 'injury_adjusted']):
                feature_categories['Injury Impact'].append(col)
            elif any(x in col_lower for x in ['yoy', 'consistency', 'career', 'rookie', 'veteran', 'peak']):
                feature_categories['Time-based Features'].append(col)
            elif any(x in col_lower for x in ['market_value', 'per_market_value']):
                feature_categories['Market Value Features'].append(col)
            elif any(x in col_lower for x in ['sentiment', 'tweet', 'compound', 'polarity']):
                feature_categories['Sentiment Features'].append(col)
            else:
                feature_categories['Other'].append(col)
        
        return feature_categories
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the final feature set.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with missing values handled
        """
        print("Handling missing values...")
        
        result_df = df.copy()
        
        # Count missing values before handling
        missing_before = result_df.isnull().sum().sum()
        print(f"Missing values before handling: {missing_before}")
        
        # Strategy 1: Fill numerical columns with median
        numerical_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if result_df[col].isnull().any():
                median_val = result_df[col].median()
                result_df[col].fillna(median_val, inplace=True)
        
        # Strategy 2: Fill categorical columns with mode
        categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if result_df[col].isnull().any():
                mode_val = result_df[col].mode()
                if len(mode_val) > 0:
                    result_df[col].fillna(mode_val[0], inplace=True)
                else:
                    result_df[col].fillna('unknown', inplace=True)
        
        # Strategy 3: Fill remaining missing values with 0
        result_df.fillna(0, inplace=True)
        
        missing_after = result_df.isnull().sum().sum()
        print(f"Missing values after handling: {missing_after}")
        
        return result_df
    
    def create_final_feature_set(self) -> pd.DataFrame:
        """
        Create the final feature set by combining all available features.
        
        Returns:
            DataFrame with final features
        """
        print("=" * 60)
        print("GENERATING FINAL FEATURE SET")
        print("=" * 60)
        
        # Load all feature sets
        self.load_base_features()
        self.load_advanced_features()
        self.load_sentiment_features()
        
        # Align features by player
        self.final_features = self.align_features_by_player()
        
        if self.final_features.empty:
            print("No features available to combine")
            return pd.DataFrame()
        
        # Handle missing values
        self.final_features = self.handle_missing_values(self.final_features)
        
        # Create feature categories
        feature_categories = self.create_feature_categories(self.final_features)
        
        # Print feature summary
        print("\nFeature Summary:")
        print("-" * 40)
        for category, features in feature_categories.items():
            if features:
                print(f"{category}: {len(features)} features")
        
        # Save final features
        output_path = "../processed/features_final.csv"
        self.final_features.to_csv(output_path, index=False)
        print(f"\nFinal features saved to {output_path}")
        
        # Save feature categories for reference
        categories_path = "../processed/feature_categories.json"
        import json
        with open(categories_path, 'w') as f:
            json.dump(feature_categories, f, indent=2)
        print(f"Feature categories saved to {categories_path}")
        
        return self.final_features
    
    def get_feature_importance_preview(self) -> pd.DataFrame:
        """
        Get a preview of feature importance based on correlation with target.
        
        Returns:
            DataFrame with feature importance preview
        """
        if self.final_features is None or self.final_features.empty:
            print("No final features available")
            return pd.DataFrame()
        
        # Look for target variable (market value or similar)
        target_candidates = ['market_value_in_eur', 'fifa_rating', 'season_minutes_played']
        target_col = None
        
        for candidate in target_candidates:
            if candidate in self.final_features.columns:
                target_col = candidate
                break
        
        if target_col is None:
            print("No suitable target variable found for importance analysis")
            return pd.DataFrame()
        
        # Calculate correlations with target
        numerical_cols = self.final_features.select_dtypes(include=[np.number]).columns
        correlations = []
        
        for col in numerical_cols:
            if col != target_col and not col.startswith('dob_') and not col.startswith('nationality_'):
                try:
                    corr = self.final_features[col].corr(self.final_features[target_col])
                    if not pd.isna(corr):
                        correlations.append({
                            'feature': col,
                            'correlation': abs(corr),
                            'correlation_raw': corr,
                            'target': target_col
                        })
                except:
                    continue
        
        # Sort by absolute correlation
        importance_df = pd.DataFrame(correlations)
        if not importance_df.empty:
            importance_df = importance_df.sort_values('correlation', ascending=False)
        
        if not importance_df.empty:
            print(f"\nTop 20 features by correlation with {target_col}:")
            print("-" * 50)
            print(importance_df.head(20)[['feature', 'correlation_raw']].to_string(index=False))
        else:
            print(f"\nNo correlations found with {target_col}")
        
        return importance_df

def main():
    """Main function to generate final features."""
    # Initialize feature generator
    generator = FinalFeatureGenerator()
    
    # Create final feature set
    final_features = generator.create_final_feature_set()
    
    if not final_features.empty:
        # Get feature importance preview
        generator.get_feature_importance_preview()
        
        print(f"\nFinal feature set created successfully!")
        print(f"Total features: {len(final_features.columns)}")
        print(f"Total records: {len(final_features)}")
        
        return final_features
    else:
        print("Failed to create final feature set")
        return None

if __name__ == "__main__":
    main()
