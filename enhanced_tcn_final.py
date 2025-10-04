#!/usr/bin/env python3
"""
Enhanced TCN Model - Quick Fix Version

This version applies the performance enhancements while ensuring robust data handling.
It includes all the optimizations but with better missing data handling.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os

# Import the base model
from ensemble_air_quality_model import TCNAirQualityModel

class EnhancedTCNModel(TCNAirQualityModel):
    """
    Enhanced TCN model with improved data handling and all optimizations
    """
    
    def prepare_sequences_enhanced(self, df, target_pollutants=['NO2', 'O3']):
        """
        Enhanced sequence preparation with better missing data handling
        """
        print(f"Enhanced sequence preparation for better data handling...")
        print(f"Target pollutants: {target_pollutants}")
        
        # Print available columns for debugging
        print(f"Available columns: {len(df.columns)} features")
        
        # Group by monitoring site
        site_groups = df.groupby('location_id')
        print(f"Processing {len(site_groups)} monitoring sites...")
        
        all_X = []
        all_y = []
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['location_id', 'datetime', 'Date Local', 'Time Local', 
                       'State Name', 'County Name', 'Latitude', 'Longitude']
        feature_cols = [col for col in df.columns if col not in exclude_cols and not pd.api.types.is_string_dtype(df[col])]
        
        print(f"  Using {len(feature_cols)} features")
        
        valid_sites = 0
        for site_id, site_data in site_groups:
            if len(site_data) < self.sequence_length + 1:
                continue
                
            # Sort by datetime if available
            if 'datetime' in site_data.columns:
                site_data = site_data.sort_values('datetime').copy()
            
            # Enhanced missing value handling
            # 1. Check if we have enough data for target pollutants
            target_data_available = True
            for pollutant in target_pollutants:
                if pollutant not in site_data.columns:
                    target_data_available = False
                    break
                # Check if we have sufficient non-null values
                non_null_ratio = site_data[pollutant].notna().sum() / len(site_data)
                if non_null_ratio < 0.3:  # Need at least 30% valid data
                    target_data_available = False
                    break
            
            if not target_data_available:
                continue
                
            # 2. Fill missing values more robustly
            for col in feature_cols:
                if col in site_data.columns:
                    # Forward fill, then backward fill, then fill with median
                    site_data[col] = site_data[col].fillna(method='ffill')
                    site_data[col] = site_data[col].fillna(method='bfill')
                    if site_data[col].isna().any():
                        median_val = site_data[col].median()
                        if pd.notna(median_val):
                            site_data[col] = site_data[col].fillna(median_val)
                        else:
                            site_data[col] = site_data[col].fillna(0)
            
            # 3. Create sequences only if we have valid data
            sequences_added = 0
            for i in range(len(site_data) - self.sequence_length):
                # Input sequence
                X_seq = site_data.iloc[i:i + self.sequence_length][feature_cols].values
                
                # Target
                target_idx = i + self.sequence_length
                if target_idx < len(site_data):
                    y_seq = []
                    valid_target = True
                    
                    for pollutant in target_pollutants:
                        if pollutant in site_data.columns:
                            target_val = site_data.iloc[target_idx][pollutant]
                            if pd.notna(target_val):
                                y_seq.append(target_val)
                            else:
                                valid_target = False
                                break
                        else:
                            valid_target = False
                            break
                    
                    # Only add if we have valid data for both input and target
                    if valid_target and not np.isnan(X_seq).any() and len(y_seq) == len(target_pollutants):
                        all_X.append(X_seq)
                        all_y.append(y_seq)
                        sequences_added += 1
            
            if sequences_added > 0:
                valid_sites += 1
        
        if not all_X:
            print("❌ No valid sequences created!")
            return np.array([]), np.array([]), feature_cols, target_pollutants
            
        X = np.array(all_X)
        y = np.array(all_y)
        
        print(f"✅ Created {len(X)} sequences from {valid_sites} valid sites")
        print(f"  Input shape: {X.shape} (samples, time_steps, features)")
        print(f"  Target shape: {y.shape} (samples, targets)")
        
        return X, y, feature_cols, target_pollutants

def main():
    """Enhanced main function"""
    
    print("🚀 ENHANCED TCN AIR QUALITY MODEL")
    print("="*60)
    print("Optimizations enabled:")
    print("✅ RobustScaler for outlier handling")
    print("✅ Enhanced feature engineering")
    print("✅ Weighted multi-task loss")
    print("✅ Optimized training configuration")
    print("✅ Improved data handling")
    print("="*60)
    
    # Change to NASA Space Apps directory
    repo_dir = "/Users/a91788/Desktop/NASA/Nasa-Space-Apps"
    os.chdir(repo_dir)
    
    # Initialize enhanced model
    model = EnhancedTCNModel(
        sequence_length=24,
        n_filters=40,  # Slightly increased capacity
        kernel_size=3,
        n_layers=4,
        dropout_rate=0.28,  # Slightly reduced dropout
        l2_reg=0.008,  # Slightly reduced L2
        dilation_rates=[1, 2, 4, 8]
    )
    
    # Check if processed data exists
    processed_file = 'tcn_models/real_air_quality_time_series.csv'
    if not os.path.exists(processed_file):
        print("📊 Loading and processing real data...")
        df = model.load_real_data()
        if df is None or df.empty:
            print("❌ Error: Could not load real data!")
            return
    else:
        print("📊 Loading processed time series data...")
        df = pd.read_csv(processed_file)
        print(f"  Loaded {len(df)} records")
    
    # Enhanced sequence preparation
    print("\n🔄 Preparing sequences with enhanced data handling...")
    target_pollutants = ['NO2', 'O3']
    X, y, feature_names, target_names = model.prepare_sequences_enhanced(df, target_pollutants)
    
    if len(X) == 0:
        print("❌ Error: No valid sequences created!")
        return
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )
    
    print(f"  Training: {len(X_train)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Train enhanced model
    print("\n🎯 Training enhanced TCN model...")
    tcn_model = model.train_tcn_model(X_train, X_val, y_train, y_val, feature_names, target_names)
    
    # Evaluate model
    print("\n📊 Evaluating enhanced model...")
    metrics, y_true, y_pred = model.evaluate_tcn_model(X_test, y_test, target_names)
    
    # Results comparison
    print(f"\n{'='*60}")
    print("🎯 ENHANCED MODEL RESULTS")
    print(f"{'='*60}")
    
    baseline_no2 = 42.3
    baseline_o3 = 76.8
    
    for target_name, target_metrics in metrics['per_target'].items():
        current_acc = target_metrics['Accuracy_10%'] * 100
        
        if target_name == 'NO2':
            improvement = current_acc - baseline_no2
            print(f"\n{target_name} Performance:")
            print(f"  Baseline: {baseline_no2:.1f}% (±10%)")
            print(f"  Enhanced: {current_acc:.1f}% (±10%)")
            print(f"  Improvement: {improvement:+.1f} percentage points")
            
            if current_acc >= 46:
                print(f"  ✅ TARGET ACHIEVED! ({current_acc:.1f}% ≥ 46%)")
            else:
                print(f"  📈 Progress toward 46% target")
                
        elif target_name == 'O3':
            change = current_acc - baseline_o3
            print(f"\n{target_name} Performance:")
            print(f"  Baseline: {baseline_o3:.1f}% (±10%)")
            print(f"  Enhanced: {current_acc:.1f}% (±10%)")
            print(f"  Change: {change:+.1f} percentage points")
        
        print(f"\n  Detailed Metrics:")
        print(f"    RMSE: {target_metrics['RMSE']:.4f}")
        print(f"    MAE:  {target_metrics['MAE']:.4f}")
        print(f"    R²:   {target_metrics['R²']:.4f}")
        print(f"    Accuracy (±20%): {target_metrics['Accuracy_20%']*100:.1f}%")
        print(f"    MAPE: {target_metrics['MAPE']:.2f}%")
    
    print(f"\nOverall Performance:")
    print(f"  RMSE: {metrics['overall']['RMSE']:.4f}")
    print(f"  MAE:  {metrics['overall']['MAE']:.4f}")
    
    # Save enhanced model
    try:
        model.tcn_model.save('enhanced_tcn_air_quality_model.keras')
        print(f"\n💾 Enhanced model saved as 'enhanced_tcn_air_quality_model.keras'")
    except Exception as e:
        print(f"\n⚠️  Could not save model: {e}")
    
    print(f"\n🎉 Enhanced training complete with all optimizations!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
