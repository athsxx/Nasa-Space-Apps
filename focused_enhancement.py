#!/usr/bin/env python3
"""
Focused Performance Enhancement for Existing TCN Model

Applies targeted improvements to the working ensemble model:
1. Enhanced preprocessing with RobustScaler
2. Improved loss function with multi-task weighting
3. Better regularization and optimization
4. Advanced callbacks and learning rate scheduling

Target: Improve NO2 accuracy from 42.3% to 45%+ (±10%)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os

# Import the existing model
from ensemble_air_quality_model import TCNAirQualityModel

class FocusedPerformanceEnhancer:
    """
    Focused enhancement of existing working TCN model
    """
    
    def __init__(self):
        print("🎯 Focused Performance Enhancer initialized")
        print("  Strategy: Enhance existing working model with proven optimizations")
        print("  Target: NO2 accuracy 42.3% → 45%+ (±10%)")
    
    def load_existing_data(self):
        """Load the existing processed data"""
        print("📊 Loading existing processed data...")
        
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        if not os.path.exists(data_file):
            print("❌ Data file not found!")
            return None
        
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} records from {df['location_id'].nunique()} locations")
        
        return df
    
    def apply_focused_enhancements(self, df):
        """Apply focused enhancements to the existing model pipeline"""
        print("🔧 Applying focused enhancements...")
        
        # Filter for best locations (same approach as working model but optimized)
        print("  • Enhanced location filtering...")
        
        location_stats = []
        for location_id, group in df.groupby('location_id'):
            no2_count = group['NO2'].count()
            o3_count = group['O3'].count()
            total_count = len(group)
            
            # Calculate completeness
            no2_completeness = no2_count / total_count if total_count > 0 else 0
            o3_completeness = o3_count / total_count if total_count > 0 else 0
            
            location_stats.append({
                'location_id': location_id,
                'total_records': total_count,
                'no2_completeness': no2_completeness,
                'o3_completeness': o3_completeness,
                'combined_completeness': (no2_completeness + o3_completeness) / 2
            })
        
        stats_df = pd.DataFrame(location_stats)
        
        # Enhanced filtering criteria
        good_locations = stats_df[
            (stats_df['no2_completeness'] >= 0.25) &
            (stats_df['o3_completeness'] >= 0.25) &
            (stats_df['total_records'] >= 30) &
            (stats_df['combined_completeness'] >= 0.3)
        ]['location_id'].values
        
        df_filtered = df[df['location_id'].isin(good_locations)].copy()
        
        print(f"  • Selected {len(good_locations)} high-quality locations")
        print(f"  • Filtered to {len(df_filtered)} records")
        
        return df_filtered
    
    def create_enhanced_sequences(self, df):
        """Create sequences with enhanced preprocessing"""
        print("🔄 Creating enhanced sequences...")
        
        # Use existing feature approach but with enhancements
        exclude_cols = ['location_id', 'datetime', 'Latitude', 'Longitude', 
                       'State Name', 'County Name']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       col not in ['NO2', 'O3'] and
                       not pd.api.types.is_string_dtype(df[col])]
        
        print(f"  Using {len(feature_cols)} features")
        
        sequences = []
        targets = []
        sequence_length = 24
        
        # Enhanced sequence creation with better imputation
        for location_id, site_data in df.groupby('location_id'):
            if len(site_data) < sequence_length + 10:
                continue
            
            # Sort by datetime
            if 'datetime' in site_data.columns:
                try:
                    site_data = site_data.sort_values('datetime').copy()
                except:
                    pass
            
            site_data = site_data.reset_index(drop=True)
            
            # Enhanced imputation strategy
            site_data_clean = site_data.copy()
            
            # Forward fill then backward fill for all columns
            for col in feature_cols + ['NO2', 'O3']:
                if col in site_data_clean.columns:
                    site_data_clean[col] = site_data_clean[col].fillna(method='ffill')
                    site_data_clean[col] = site_data_clean[col].fillna(method='bfill')
                    # Use median for remaining nulls
                    if site_data_clean[col].isnull().any():
                        median_val = site_data_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        site_data_clean[col].fillna(median_val, inplace=True)
            
            # Create sequences
            for i in range(len(site_data_clean) - sequence_length):
                X_seq = site_data_clean.iloc[i:i + sequence_length][feature_cols].values
                
                # Target at next timestep
                target_idx = i + sequence_length
                if target_idx < len(site_data_clean):
                    no2_target = site_data_clean.iloc[target_idx]['NO2']
                    o3_target = site_data_clean.iloc[target_idx]['O3']
                    
                    # Quality checks
                    if (not np.isnan(X_seq).any() and 
                        not np.isnan(no2_target) and 
                        not np.isnan(o3_target) and
                        not np.isinf(X_seq).any()):
                        sequences.append(X_seq)
                        targets.append([no2_target, o3_target])
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"  Created {len(X)} enhanced sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols
    
    def build_enhanced_tcn(self, n_features, sequence_length=24):
        """Build enhanced TCN with focused improvements"""
        print("🏗️ Building enhanced TCN model...")
        
        # Input layer
        inputs = keras.layers.Input(shape=(sequence_length, n_features), name='input')
        
        # Enhanced TCN architecture (based on working model but improved)
        n_filters = 32  # Proven to work
        dilation_rates = [1, 2, 4, 8]  # Good balance
        
        # Initial convolution
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='initial_conv'
        )(inputs)
        
        # Enhanced TCN blocks with proven architecture
        for i, dilation_rate in enumerate(dilation_rates):
            # First dilated convolution  
            conv1 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.008),  # Slightly reduced
                name=f'conv1_{i}'
            )(x)
            
            conv1 = keras.layers.BatchNormalization(name=f'bn1_{i}')(conv1)
            conv1 = keras.layers.Dropout(0.2, name=f'drop1_{i}')(conv1)  # Reduced dropout
            
            # Second dilated convolution
            conv2 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.008),
                name=f'conv2_{i}'
            )(conv1)
            
            conv2 = keras.layers.BatchNormalization(name=f'bn2_{i}')(conv2)
            conv2 = keras.layers.Dropout(0.2, name=f'drop2_{i}')(conv2)
            
            # Residual connection
            if x.shape[-1] != n_filters:
                residual = keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=1,
                    kernel_regularizer=keras.regularizers.l2(0.008),
                    name=f'residual_{i}'
                )(x)
            else:
                residual = x
            
            x = keras.layers.Add(name=f'add_{i}')([conv2, residual])
            x = keras.layers.Activation('relu', name=f'relu_{i}')(x)
        
        # Enhanced feature extraction (use proven approach)
        x_global = keras.layers.GlobalAveragePooling1D(name='global_avg')(x)
        x_last = keras.layers.Lambda(lambda t: t[:, -1, :], name='last_timestep')(x)
        combined = keras.layers.Concatenate(name='combined')([x_global, x_last])
        
        # Dense layers with enhanced architecture
        dense1 = keras.layers.Dense(
            64,  # Proven size
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.008),
            name='dense1'
        )(combined)
        dense1 = keras.layers.BatchNormalization(name='dense1_bn')(dense1)
        dense1 = keras.layers.Dropout(0.3, name='dense1_dropout')(dense1)
        
        dense2 = keras.layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.008),
            name='dense2'
        )(dense1)
        dense2 = keras.layers.BatchNormalization(name='dense2_bn')(dense2)
        dense2 = keras.layers.Dropout(0.25, name='dense2_dropout')(dense2)
        
        # Enhanced multi-task output
        # NO2 head (more challenging, needs more capacity)
        no2_head = keras.layers.Dense(24, activation='relu', name='no2_head')(dense2)
        no2_head = keras.layers.Dropout(0.2, name='no2_head_dropout')(no2_head)
        no2_output = keras.layers.Dense(1, activation='linear', name='no2_output')(no2_head)
        
        # O3 head (easier, smaller capacity)
        o3_head = keras.layers.Dense(16, activation='relu', name='o3_head')(dense2)
        o3_head = keras.layers.Dropout(0.15, name='o3_head_dropout')(o3_head)
        o3_output = keras.layers.Dense(1, activation='linear', name='o3_output')(o3_head)
        
        # Combine outputs
        outputs = keras.layers.Concatenate(name='outputs')([no2_output, o3_output])
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='EnhancedTCN')
        
        # Enhanced loss function with weighted multi-task learning
        def enhanced_weighted_loss(y_true, y_pred):
            no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
            no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
            
            # Use Huber loss for robustness (proven to work)
            huber = keras.losses.Huber(delta=1.0)
            
            no2_loss = huber(no2_true, no2_pred)
            o3_loss = huber(o3_true, o3_pred)
            
            # Enhanced weighting: Emphasize NO2 improvement while keeping O3 stable
            return 2.5 * no2_loss + 1.0 * o3_loss
        
        # Enhanced optimizer configuration
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.0008,  # Slightly lower for stability
                weight_decay=0.01,
                clipnorm=0.8
            ),
            loss=enhanced_weighted_loss,
            metrics=['mae', 'mse']
        )
        
        print(f"  Enhanced model created with {model.count_params():,} parameters")
        return model
    
    def train_enhanced_model(self, X_train, X_val, y_train, y_val):
        """Train with enhanced configuration"""
        print("🎯 Training enhanced model...")
        
        # Use RobustScaler for better outlier handling (key enhancement)
        print("  • Using RobustScaler for robust preprocessing")
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        # Scale features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
        X_val_scaled = feature_scaler.transform(X_val_reshaped).reshape(-1, n_timesteps, n_features)
        
        # Scale targets
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_val_scaled = target_scaler.transform(y_val)
        
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        
        # Build enhanced model
        model = self.build_enhanced_tcn(n_features)
        
        # Enhanced callbacks (proven to work)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                min_delta=0.0005
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='enhanced_tcn_focused.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Enhanced training configuration
        print("  🚀 Starting focused enhancement training...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100,  # More epochs for better convergence
            batch_size=64,  # Proven batch size
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        return model
    
    def evaluate_enhanced_model(self, X_test, y_test):
        """Evaluate enhanced model"""
        print("📊 Evaluating enhanced model...")
        
        # Scale test data
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
        
        # Predict
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate enhanced metrics
        results = {}
        
        for i, pollutant in enumerate(['NO2', 'O3']):
            y_true_i = y_test[:, i]
            y_pred_i = y_pred[:, i]
            
            # Core metrics
            mse = np.mean((y_true_i - y_pred_i) ** 2)
            mae = np.mean(np.abs(y_true_i - y_pred_i))
            rmse = np.sqrt(mse)
            
            # R²
            ss_res = np.sum((y_true_i - y_pred_i) ** 2)
            ss_tot = np.sum((y_true_i - np.mean(y_true_i)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Enhanced accuracy calculation (same as baseline for comparison)
            abs_errors = np.abs(y_true_i - y_pred_i)
            abs_true = np.abs(y_true_i)
            threshold_base = np.maximum(abs_true, np.percentile(abs_true, 5))
            
            acc_10 = np.mean(abs_errors <= 0.10 * threshold_base)
            
            results[pollutant] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_10%': acc_10
            }
        
        return results

def main():
    """Main execution function"""
    
    print("🎯 FOCUSED PERFORMANCE ENHANCEMENT")
    print("="*60)
    print("Strategy: Enhance working model with proven optimizations")
    print("Target: NO2 accuracy 42.3% → 45%+ (±10%)")
    print("="*60)
    
    # Initialize enhancer
    enhancer = FocusedPerformanceEnhancer()
    
    # Load existing data
    df = enhancer.load_existing_data()
    if df is None:
        return
    
    # Apply focused enhancements
    df_enhanced = enhancer.apply_focused_enhancements(df)
    
    # Create enhanced sequences
    X, y, feature_names = enhancer.create_enhanced_sequences(df_enhanced)
    
    if len(X) == 0:
        print("❌ No sequences created!")
        return
    
    # Split data
    print("\n🔄 Splitting data for training...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train enhanced model
    model = enhancer.train_enhanced_model(X_train, X_val, y_train, y_val)
    
    # Evaluate
    print("\n📊 Final evaluation...")
    results = enhancer.evaluate_enhanced_model(X_test, y_test)
    
    # Display results
    print(f"\n{'='*60}")
    print("🏆 FOCUSED ENHANCEMENT RESULTS")
    print(f"{'='*60}")
    
    for pollutant, metrics in results.items():
        print(f"\n{pollutant} Performance:")
        for metric, value in metrics.items():
            if 'Accuracy' in metric:
                print(f"  {metric}: {value*100:.1f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Performance comparison
    baseline_no2 = 42.3
    enhanced_no2 = results['NO2']['Accuracy_10%'] * 100
    target_no2 = 45.0
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"  Baseline NO2 (10%±): {baseline_no2:.1f}%")
    print(f"  Enhanced NO2 (10%±): {enhanced_no2:.1f}%")
    print(f"  Target NO2 (10%±): {target_no2:.1f}%")
    
    improvement = enhanced_no2 - baseline_no2
    if enhanced_no2 >= target_no2:
        print(f"  ✅ TARGET ACHIEVED! (+{improvement:.1f}pp)")
    elif improvement > 0:
        print(f"  📈 Improvement: +{improvement:.1f}pp")
        remaining = target_no2 - enhanced_no2
        print(f"  Remaining to target: {remaining:.1f}pp")
    else:
        print(f"  📉 Change: {improvement:.1f}pp")
    
    enhanced_o3 = results['O3']['Accuracy_10%'] * 100
    print(f"  Enhanced O3 (10%±): {enhanced_o3:.1f}%")
    
    print(f"\n{'='*60}")
    print("✨ Focused enhancement complete!")

if __name__ == "__main__":
    main()
