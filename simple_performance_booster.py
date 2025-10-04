#!/usr/bin/env python3
"""
Simple Performance Booster

Creates a working baseline model and demonstrates performance improvement.
Focus on achieving measurable gains in NO2 prediction accuracy.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os

class SimplePerformanceBooster:
    """
    Simple approach to boost model performance
    """
    
    def __init__(self):
        print("🚀 Simple Performance Booster initialized")
        print("  Target: Demonstrate NO2 accuracy improvement")
    
    def load_and_prepare_data(self):
        """Load and prepare data with working approach"""
        print("📊 Loading and preparing data...")
        
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        if not os.path.exists(data_file):
            print("❌ Data file not found!")
            return None
        
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} records from {df['location_id'].nunique()} locations")
        
        # Use very permissive filtering to get working sequences
        print("  • Using permissive location filtering...")
        
        location_quality = []
        for location_id, group in df.groupby('location_id'):
            no2_count = group['NO2'].count()
            o3_count = group['O3'].count()
            total_count = len(group)
            
            if (total_count >= 20 and no2_count >= 5 and o3_count >= 5):
                location_quality.append({
                    'location_id': location_id,
                    'total': total_count,
                    'no2_pct': no2_count / total_count,
                    'o3_pct': o3_count / total_count
                })
        
        # Sort by data quality and take best locations
        location_quality.sort(key=lambda x: x['no2_pct'] + x['o3_pct'], reverse=True)
        good_locations = [loc['location_id'] for loc in location_quality[:200]]  # Top 200 locations
        
        df_filtered = df[df['location_id'].isin(good_locations)].copy()
        
        print(f"  • Selected {len(good_locations)} high-quality locations")
        print(f"  • Filtered dataset: {len(df_filtered)} records")
        
        return df_filtered
    
    def create_working_sequences(self, df):
        """Create sequences that actually work"""
        print("🔄 Creating working sequences...")
        
        # Simple feature selection
        exclude_cols = ['location_id', 'datetime', 'Latitude', 'Longitude', 
                       'State Name', 'County Name']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       col not in ['NO2', 'O3'] and
                       not pd.api.types.is_string_dtype(df[col])]
        
        print(f"  Using {len(feature_cols)} features: {feature_cols}")
        
        sequences = []
        targets = []
        sequence_length = 24
        
        # Create sequences with aggressive data cleaning
        for location_id, site_data in df.groupby('location_id'):
            if len(site_data) < sequence_length + 10:
                continue
            
            # Simple sorting
            site_data = site_data.reset_index(drop=True)
            
            # Aggressive cleaning: remove rows with any NaN in key columns
            key_cols = feature_cols + ['NO2', 'O3']
            site_data_clean = site_data.dropna(subset=key_cols).copy()
            
            if len(site_data_clean) < sequence_length + 5:
                continue
            
            # Fill remaining NaNs with median
            for col in key_cols:
                if col in site_data_clean.columns:
                    if site_data_clean[col].isnull().any():
                        median_val = site_data_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        site_data_clean[col].fillna(median_val, inplace=True)
            
            # Create sequences from clean data
            for i in range(len(site_data_clean) - sequence_length):
                X_seq = site_data_clean.iloc[i:i + sequence_length][feature_cols].values
                
                # Target at next timestep
                target_idx = i + sequence_length
                if target_idx < len(site_data_clean):
                    no2_target = site_data_clean.iloc[target_idx]['NO2']
                    o3_target = site_data_clean.iloc[target_idx]['O3']
                    
                    # Final quality check
                    if (not np.any(np.isnan(X_seq)) and 
                        not np.any(np.isinf(X_seq)) and
                        not pd.isna(no2_target) and not pd.isna(o3_target) and
                        not np.isinf(no2_target) and not np.isinf(o3_target) and
                        np.all(np.isfinite(X_seq))):
                        sequences.append(X_seq)
                        targets.append([no2_target, o3_target])
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"  Created {len(X)} clean sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols
    
    def create_baseline_model(self, n_features):
        """Create baseline TCN model"""
        print("🏗️ Building baseline TCN model...")
        
        inputs = keras.layers.Input(shape=(24, n_features), name='input')
        
        # Simple TCN architecture
        n_filters = 32
        dilation_rates = [1, 2, 4, 8]
        
        x = inputs
        
        # TCN blocks
        for i, dilation_rate in enumerate(dilation_rates):
            # Dilated convolution
            conv = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                name=f'conv_{i}'
            )(x)
            
            conv = keras.layers.Dropout(0.3, name=f'drop_{i}')(conv)
            
            # Residual connection
            if x.shape[-1] != n_filters:
                residual = keras.layers.Conv1D(filters=n_filters, kernel_size=1, name=f'residual_{i}')(x)
            else:
                residual = x
            
            x = keras.layers.Add(name=f'add_{i}')([conv, residual])
        
        # Global pooling
        x = keras.layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        x = keras.layers.Dense(64, activation='relu', name='dense1')(x)
        x = keras.layers.Dropout(0.3, name='dense1_dropout')(x)
        x = keras.layers.Dense(32, activation='relu', name='dense2')(x)
        x = keras.layers.Dropout(0.2, name='dense2_dropout')(x)
        
        # Output layer (2 pollutants)
        outputs = keras.layers.Dense(2, activation='linear', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='BaselineTCN')
        
        # Simple loss function
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"  Baseline model: {model.count_params():,} parameters")
        return model
    
    def create_enhanced_model(self, n_features):
        """Create enhanced TCN model"""
        print("🏗️ Building enhanced TCN model...")
        
        inputs = keras.layers.Input(shape=(24, n_features), name='input')
        
        # Enhanced TCN architecture
        n_filters = 40  # Increased capacity
        dilation_rates = [1, 2, 4, 8, 16]  # Added layer
        
        x = inputs
        
        # Enhanced TCN blocks
        for i, dilation_rate in enumerate(dilation_rates):
            # First conv
            conv1 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name=f'conv1_{i}'
            )(x)
            
            conv1 = keras.layers.BatchNormalization(name=f'bn1_{i}')(conv1)
            conv1 = keras.layers.Dropout(0.2, name=f'drop1_{i}')(conv1)
            
            # Second conv
            conv2 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name=f'conv2_{i}'
            )(conv1)
            
            conv2 = keras.layers.BatchNormalization(name=f'bn2_{i}')(conv2)
            conv2 = keras.layers.Dropout(0.2, name=f'drop2_{i}')(conv2)
            
            # Residual connection
            if x.shape[-1] != n_filters:
                residual = keras.layers.Conv1D(filters=n_filters, kernel_size=1, name=f'residual_{i}')(x)
            else:
                residual = x
            
            x = keras.layers.Add(name=f'add_{i}')([conv2, residual])
        
        # Enhanced feature extraction
        global_avg = keras.layers.GlobalAveragePooling1D(name='global_avg')(x)
        global_max = keras.layers.GlobalMaxPooling1D(name='global_max')(x)
        combined = keras.layers.Concatenate(name='combined')([global_avg, global_max])
        
        # Enhanced dense layers
        x = keras.layers.Dense(96, activation='relu', name='dense1')(combined)
        x = keras.layers.BatchNormalization(name='dense1_bn')(x)
        x = keras.layers.Dropout(0.3, name='dense1_dropout')(x)
        
        x = keras.layers.Dense(48, activation='relu', name='dense2')(x)
        x = keras.layers.BatchNormalization(name='dense2_bn')(x)
        x = keras.layers.Dropout(0.25, name='dense2_dropout')(x)
        
        # Multi-task outputs
        # NO2 head
        no2_head = keras.layers.Dense(24, activation='relu', name='no2_head')(x)
        no2_head = keras.layers.Dropout(0.2, name='no2_dropout')(no2_head)
        no2_output = keras.layers.Dense(1, activation='linear', name='no2_output')(no2_head)
        
        # O3 head  
        o3_head = keras.layers.Dense(16, activation='relu', name='o3_head')(x)
        o3_head = keras.layers.Dropout(0.15, name='o3_dropout')(o3_head)
        o3_output = keras.layers.Dense(1, activation='linear', name='o3_output')(o3_head)
        
        # Combine outputs
        outputs = keras.layers.Concatenate(name='outputs')([no2_output, o3_output])
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='EnhancedTCN')
        
        # Enhanced loss function (weight NO2 more)
        def weighted_loss(y_true, y_pred):
            no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
            no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
            
            no2_loss = keras.losses.MeanSquaredError()(no2_true, no2_pred)
            o3_loss = keras.losses.MeanSquaredError()(o3_true, o3_pred)
            
            return 2.0 * no2_loss + 1.0 * o3_loss  # Weight NO2 more
        
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.0008,
                weight_decay=0.01
            ),
            loss=weighted_loss,
            metrics=['mae']
        )
        
        print(f"  Enhanced model: {model.count_params():,} parameters")
        return model
    
    def train_and_compare_models(self, X, y):
        """Train both models and compare performance"""
        print("🎯 Training and comparing models...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"  Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
        
        # Scale data
        scaler = StandardScaler()
        
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(-1, n_timesteps, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(-1, n_timesteps, n_features)
        
        results = {}
        
        # Train baseline model
        print("\n  📊 Training baseline model...")
        baseline_model = self.create_baseline_model(n_features)
        
        history_baseline = baseline_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=30,
            batch_size=64,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        
        # Evaluate baseline
        y_pred_baseline = baseline_model.predict(X_test_scaled, verbose=0)
        results['baseline'] = self.calculate_metrics(y_test, y_pred_baseline)
        
        # Train enhanced model  
        print("\n  🚀 Training enhanced model...")
        enhanced_model = self.create_enhanced_model(n_features)
        
        history_enhanced = enhanced_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=64,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=8, min_lr=1e-6)
            ]
        )
        
        # Evaluate enhanced
        y_pred_enhanced = enhanced_model.predict(X_test_scaled, verbose=0)
        results['enhanced'] = self.calculate_metrics(y_test, y_pred_enhanced)
        
        return results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        
        metrics = {}
        
        for i, pollutant in enumerate(['NO2', 'O3']):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            
            # Core metrics
            mse = np.mean((y_true_i - y_pred_i) ** 2)
            mae = np.mean(np.abs(y_true_i - y_pred_i))
            rmse = np.sqrt(mse)
            
            # R²
            ss_res = np.sum((y_true_i - y_pred_i) ** 2)
            ss_tot = np.sum((y_true_i - np.mean(y_true_i)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Accuracy metrics
            abs_errors = np.abs(y_true_i - y_pred_i)
            abs_true = np.abs(y_true_i)
            threshold_base = np.maximum(abs_true, np.percentile(abs_true, 5))
            
            acc_10 = np.mean(abs_errors <= 0.10 * threshold_base)
            acc_20 = np.mean(abs_errors <= 0.20 * threshold_base)
            
            metrics[pollutant] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_10%': acc_10,
                'Accuracy_20%': acc_20
            }
        
        return metrics

def main():
    """Main execution function"""
    
    print("🚀 SIMPLE PERFORMANCE BOOSTER")
    print("="*60)
    print("Strategy: Demonstrate NO2 accuracy improvement")
    print("="*60)
    
    # Initialize booster
    booster = SimplePerformanceBooster()
    
    # Load and prepare data
    df = booster.load_and_prepare_data()
    if df is None:
        print("❌ Could not load data!")
        return
    
    # Create working sequences
    X, y, feature_names = booster.create_working_sequences(df)
    
    if len(X) == 0:
        print("❌ No sequences created!")
        return
    
    # Train and compare models
    results = booster.train_and_compare_models(X, y)
    
    # Display results
    print(f"\n{'='*60}")
    print("🏆 PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    print("\n📊 Baseline Model:")
    for pollutant, metrics in results['baseline'].items():
        print(f"  {pollutant}:")
        for metric, value in metrics.items():
            if 'Accuracy' in metric:
                print(f"    {metric}: {value*100:.1f}%")
            else:
                print(f"    {metric}: {value:.4f}")
    
    print("\n🚀 Enhanced Model:")
    for pollutant, metrics in results['enhanced'].items():
        print(f"  {pollutant}:")
        for metric, value in metrics.items():
            if 'Accuracy' in metric:
                print(f"    {metric}: {value*100:.1f}%")
            else:
                print(f"    {metric}: {value:.4f}")
    
    # Calculate improvements
    print(f"\n🎯 IMPROVEMENT SUMMARY:")
    
    for pollutant in ['NO2', 'O3']:
        baseline_acc = results['baseline'][pollutant]['Accuracy_10%'] * 100
        enhanced_acc = results['enhanced'][pollutant]['Accuracy_10%'] * 100
        improvement = enhanced_acc - baseline_acc
        
        print(f"  {pollutant} (10%± accuracy):")
        print(f"    Baseline: {baseline_acc:.1f}%")
        print(f"    Enhanced: {enhanced_acc:.1f}%")
        if improvement > 0:
            print(f"    ✅ Improvement: +{improvement:.1f} percentage points")
        elif improvement == 0:
            print(f"    ➡️  No change")
        else:
            print(f"    📉 Change: {improvement:.1f} percentage points")
    
    # Success assessment
    no2_baseline = results['baseline']['NO2']['Accuracy_10%'] * 100
    no2_enhanced = results['enhanced']['NO2']['Accuracy_10%'] * 100
    
    print(f"\n🎯 SUCCESS ASSESSMENT:")
    if no2_enhanced > no2_baseline:
        print(f"  ✅ SUCCESS: Enhanced model shows improvement for NO2!")
        print(f"  Achieved {no2_enhanced - no2_baseline:.1f}pp gain in NO2 accuracy")
    else:
        print(f"  📊 Results: Enhanced model performance compared to baseline")
    
    print(f"\n{'='*60}")
    print("✨ Performance boost demonstration complete!")

if __name__ == "__main__":
    main()
