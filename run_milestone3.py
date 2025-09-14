"""
Milestone 3: Weeks 3-4 - Advanced Feature Engineering and Sentiment Analysis
==========================================================================

This script runs all components of Milestone 3:
1. Advanced feature engineering
2. Sentiment analysis
3. Final feature generation
4. Report generation

Author: AI Assistant
Date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_advanced_feature_engineering():
    """Run advanced feature engineering."""
    print("=" * 60)
    print("RUNNING ADVANCED FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        from feature_engineering import AdvancedFeatureEngineer
        
        fe = AdvancedFeatureEngineer()
        features_df = fe.engineer_all_features()
        
        if not features_df.empty:
            print("✓ Advanced feature engineering completed successfully")
            return True
        else:
            print("✗ Advanced feature engineering failed")
            return False
            
    except Exception as e:
        print(f"✗ Error in advanced feature engineering: {e}")
        return False

def run_sentiment_analysis():
    """Run sentiment analysis."""
    print("\n" + "=" * 60)
    print("RUNNING SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        from sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # Try to process Twitter data
        twitter_files = ["../twitter_training.csv", "../twitter_validation.csv"]
        sentiment_data = None
        
        for twitter_file in twitter_files:
            try:
                print(f"Trying to process {twitter_file}...")
                sentiment_data = analyzer.process_twitter_data(twitter_file)
                if not sentiment_data.empty:
                    print(f"✓ Successfully processed {twitter_file}")
                    break
            except FileNotFoundError:
                print(f"File {twitter_file} not found")
                continue
        
        # If no Twitter data, create synthetic data
        if sentiment_data is None or sentiment_data.empty:
            print("Creating synthetic sentiment data...")
            try:
                players_df = pd.read_csv("../processed/dataset_processed.csv")
                sentiment_data = analyzer.create_synthetic_player_sentiment(players_df)
                print("✓ Synthetic sentiment data created")
            except Exception as e:
                print(f"✗ Error creating synthetic data: {e}")
                return False
        
        # Create player-level features
        player_sentiment = analyzer.create_player_sentiment_features(sentiment_data)
        
        if not player_sentiment.empty:
            # Save results
            output_path = "../processed/player_sentiment_features.csv"
            player_sentiment.to_csv(output_path, index=False)
            print(f"✓ Player sentiment features saved to {output_path}")
            return True
        else:
            print("✗ Sentiment analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ Error in sentiment analysis: {e}")
        return False

def run_final_feature_generation():
    """Run final feature generation."""
    print("\n" + "=" * 60)
    print("RUNNING FINAL FEATURE GENERATION")
    print("=" * 60)
    
    try:
        from generate_final_features import FinalFeatureGenerator
        
        generator = FinalFeatureGenerator()
        final_features = generator.create_final_feature_set()
        
        if not final_features.empty:
            print("✓ Final feature generation completed successfully")
            
            # Get feature importance preview
            generator.get_feature_importance_preview()
            
            return True
        else:
            print("✗ Final feature generation failed")
            return False
            
    except Exception as e:
        print(f"✗ Error in final feature generation: {e}")
        return False

def generate_summary_report():
    """Generate a summary report of all activities."""
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    try:
        # Check if final features exist
        final_features_path = "../processed/features_final.csv"
        if os.path.exists(final_features_path):
            final_df = pd.read_csv(final_features_path)
            
            # Generate summary
            summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_records': int(len(final_df)),
                'total_features': int(len(final_df.columns)),
                'missing_values': int(final_df.isnull().sum().sum()),
                'feature_types': {
                    'Performance Trends': int(len([col for col in final_df.columns if any(x in col for x in ['roll', 'ema', 'form_score', 'trend'])])),
                    'Injury Impact': int(len([col for col in final_df.columns if any(x in col for x in ['injury', 'recent_injury', 'injury_risk'])])),
                    'Time-based': int(len([col for col in final_df.columns if any(x in col for x in ['yoy', 'consistency', 'career', 'rookie', 'veteran', 'peak'])])),
                    'Market Value': int(len([col for col in final_df.columns if any(x in col for x in ['market_value', 'per_market_value'])])),
                    'Sentiment': int(len([col for col in final_df.columns if any(x in col for x in ['sentiment', 'tweet', 'compound', 'polarity'])]))
                }
            }
            
            # Save summary
            summary_path = "../processed/milestone3_summary.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print("✓ Summary report generated")
            print(f"Total records: {summary['total_records']}")
            print(f"Total features: {summary['total_features']}")
            print(f"Missing values: {summary['missing_values']}")
            print("\nFeature breakdown:")
            for category, count in summary['feature_types'].items():
                print(f"  {category}: {count} features")
            
            return True
        else:
            print("✗ Final features file not found")
            return False
            
    except Exception as e:
        print(f"✗ Error generating summary report: {e}")
        return False

def main():
    """Main function to run all Milestone 3 components."""
    print("=" * 80)
    print("MILESTONE 3: WEEKS 3-4 - ADVANCED FEATURE ENGINEERING & SENTIMENT ANALYSIS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track success of each component
    results = {}
    
    # Run advanced feature engineering
    results['feature_engineering'] = run_advanced_feature_engineering()
    
    # Run sentiment analysis
    results['sentiment_analysis'] = run_sentiment_analysis()
    
    # Run final feature generation
    results['final_features'] = run_final_feature_generation()
    
    # Generate summary report
    results['summary_report'] = generate_summary_report()
    
    # Print final results
    print("\n" + "=" * 80)
    print("MILESTONE 3 COMPLETION SUMMARY")
    print("=" * 80)
    
    success_count = sum(results.values())
    total_components = len(results)
    
    print(f"Components completed successfully: {success_count}/{total_components}")
    print("\nDetailed results:")
    for component, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {component.replace('_', ' ').title()}: {status}")
    
    if success_count == total_components:
        print("\n🎉 MILESTONE 3 COMPLETED SUCCESSFULLY! 🎉")
        print("\nDeliverables created:")
        print("  - ../processed/advanced_features.csv")
        print("  - ../processed/player_sentiment_features.csv")
        print("  - ../processed/features_final.csv")
        print("  - ../processed/feature_categories.json")
        print("  - ../processed/milestone3_summary.json")
        print("  - week3_4/sentiment_analysis_report.md")
    else:
        print(f"\n⚠️  MILESTONE 3 COMPLETED WITH {total_components - success_count} FAILURES")
        print("Please check the error messages above and retry failed components.")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success_count == total_components

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
