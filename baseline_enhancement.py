#!/usr/bin/env python3
"""
Baseline Model Performance Verification and Enhancement

Loads the existing trained model and verifies the baseline performance,
then applies targeted improvements to increase NO2 accuracy.
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

class BaselineModelEnhancer:
    """
    Verify baseline and enhance existing trained model
    """
    
    def __init__(self):
        print("🎯 Baseline Model Enhancer initialized")
        print("  Strategy: Verify existing baseline and enhance performance")
        
    def load_existing_model_and_data(self):
        """Load the existing model and data"""
        print("📊 Loading existing model and data...")
        
        # Load model
        model_path = 'tcn_models/best_tcn_model.keras'
        if not os.path.exists(model_path):
            print("❌ Baseline model not found!")
            return None, None
        
        try:
            model = keras.models.load_model(model_path)
            print(f"  ✅ Loaded model: {model.count_params():,} parameters")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            return None, None
        
        # Load data
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        if not os.path.exists(data_file):
            print("❌ Data file not found!")
            return None, None
        
        df = pd.read_csv(data_file)
        print(f"  ✅ Loaded data: {len(df)} records from {df['location_id'].nunique()} locations")
        
        return model, df
    
    def recreate_original_sequences(self, df):
        """Recreate sequences using the original approach that worked"""
        print("🔄 Recreating original sequences...")
        
        # Use the same approach as the original successful model
        # Based on the analysis, we need to be more permissive with data quality
        
        # Filter locations more permissively
        print("  • Filtering locations with permissive criteria...")
        
        good_locations = []
        for location_id, group in df.groupby('location_id'):
            no2_count = group['NO2'].count()
            o3_count = group['O3'].count()
            total_count = len(group)
            
            # Very permissive criteria (like original model likely used)
            if (no2_count >= 10 and o3_count >= 10 and total_count >= 25):
                good_locations.append(location_id)
        
        df_filtered = df[df['location_id'].isin(good_locations)].copy()
        
        print(f"  • Selected {len(good_locations)} locations")
        print(f"  • Filtered dataset: {len(df_filtered)} records")
        
        # Feature selection (based on what worked originally)
        exclude_cols = ['location_id', 'datetime', 'Latitude', 'Longitude', 
                       'State Name', 'County Name']
        
        feature_cols = [col for col in df_filtered.columns 
                       if col not in exclude_cols and 
                       col not in ['NO2', 'O3'] and
                       not pd.api.types.is_string_dtype(df_filtered[col])]
        
        print(f"  Using {len(feature_cols)} features")
        
        sequences = []
        targets = []
        sequence_length = 24
        
        # Create sequences with the original approach
        for location_id, site_data in df_filtered.groupby('location_id'):
            if len(site_data) < sequence_length + 5:
                continue
            
            # Sort by datetime
            if 'datetime' in site_data.columns:
                try:
                    site_data = site_data.sort_values('datetime').copy()
                except:
                    pass
            
            site_data = site_data.reset_index(drop=True)
            
            # Simple imputation (like original)
            site_data_clean = site_data.copy()
            
            # Fill missing values with simple approach
            for col in feature_cols + ['NO2', 'O3']:
                if col in site_data_clean.columns:
                    # Use forward fill then median (original approach)
                    site_data_clean[col] = site_data_clean[col].fillna(method='ffill')
                    if site_data_clean[col].isnull().any():
                        median_val = site_data_clean[col].median()
                        if pd.isna(median_val):
                            median_val = site_data_clean[col].mean()
                            if pd.isna(median_val):
                                median_val = 0.0
                        site_data_clean[col].fillna(median_val, inplace=True)
            
            # Create sequences (original approach - more permissive)
            for i in range(len(site_data_clean) - sequence_length):
                X_seq = site_data_clean.iloc[i:i + sequence_length][feature_cols].values
                
                # Target at next timestep
                target_idx = i + sequence_length
                if target_idx < len(site_data_clean):
                    no2_target = site_data_clean.iloc[target_idx]['NO2'] 
                    o3_target = site_data_clean.iloc[target_idx]['O3']
                    
                    # More permissive quality checks (like original)
                    if (not np.any(np.isnan(X_seq)) and 
                        not np.any(np.isinf(X_seq)) and
                        not pd.isna(no2_target) and 
                        not pd.isna(o3_target) and
                        not np.isinf(no2_target) and
                        not np.isinf(o3_target)):
                        sequences.append(X_seq)
                        targets.append([no2_target, o3_target])
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"  Created {len(X)} sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols
    
    def verify_baseline_performance(self, model, X, y):
        """Verify the baseline performance using the original model"""
        print("📊 Verifying baseline performance...")
        
        # Split data the same way
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"  Test set: {len(X_test)} samples")
        
        # Use StandardScaler (original approach)
        scaler = StandardScaler()
        
        # Scale features
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
        
        # Make predictions
        try:
            y_pred = model.predict(X_test_scaled, verbose=0)
        except Exception as e:
            print(f"  ❌ Error making predictions: {e}")
            # Try with different input shape
            print("  Attempting with original training features...")
            return None
        
        # Calculate metrics (same as baseline)
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
            
            # Accuracy calculation (same as baseline)
            abs_errors = np.abs(y_true_i - y_pred_i)
            abs_true = np.abs(y_true_i)
            threshold_base = np.maximum(abs_true, np.percentile(abs_true, 5))
            
            acc_10 = np.mean(abs_errors <= 0.10 * threshold_base)
            acc_20 = np.mean(abs_errors <= 0.20 * threshold_base)
            
            results[pollutant] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_10%': acc_10,
                'Accuracy_20%': acc_20
            }
        
        return results, X_test, y_test
    
    def create_enhanced_model_from_baseline(self, original_model, n_features):
        """Create enhanced version based on the baseline architecture"""
        print("🏗️ Creating enhanced model from baseline...")
        
        # Get baseline architecture details
        print(f"  Original model layers: {len(original_model.layers)}")
        
        # Build enhanced version with similar architecture but improvements
        inputs = keras.layers.Input(shape=(24, n_features), name='input')
        
        # Enhanced TCN with proven architecture (based on original)
        n_filters = 32  # Keep same as original
        dilation_rates = [1, 2, 4, 8]  # Keep same as original
        
        # Initial convolution
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='initial_conv'
        )(inputs)
        
        # Enhanced TCN blocks (improved version of original)
        for i, dilation_rate in enumerate(dilation_rates):
            # First dilated convolution
            conv1 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.005),  # Reduced regularization
                name=f'conv1_{i}'
            )(x)
            
            conv1 = keras.layers.BatchNormalization(name=f'bn1_{i}')(conv1)
            conv1 = keras.layers.Dropout(0.15, name=f'drop1_{i}')(conv1)  # Reduced dropout
            
            # Second dilated convolution
            conv2 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.005),
                name=f'conv2_{i}'
            )(conv1)
            
            conv2 = keras.layers.BatchNormalization(name=f'bn2_{i}')(conv2)
            conv2 = keras.layers.Dropout(0.15, name=f'drop2_{i}')(conv2)
            
            # Residual connection
            if x.shape[-1] != n_filters:
                residual = keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=1,
                    kernel_regularizer=keras.regularizers.l2(0.005),
                    name=f'residual_{i}'
                )(x)
            else:
                residual = x
            
            x = keras.layers.Add(name=f'add_{i}')([conv2, residual])
            x = keras.layers.Activation('relu', name=f'relu_{i}')(x)
        
        # Enhanced feature extraction
        x_global = keras.layers.GlobalAveragePooling1D(name='global_avg')(x)
        x_last = keras.layers.Lambda(lambda t: t[:, -1, :], name='last_timestep')(x)
        
        # Add simple attention mechanism
        attention_weights = keras.layers.Dense(n_filters, activation='softmax', name='attention_weights')(x_last)
        attended = keras.layers.Multiply(name='attention_applied')([x_global, attention_weights])
        
        combined = keras.layers.Concatenate(name='combined')([x_global, x_last, attended])
        
        # Enhanced dense layers
        dense1 = keras.layers.Dense(
            80,  # Increased capacity
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.005),
            name='dense1'
        )(combined)
        dense1 = keras.layers.BatchNormalization(name='dense1_bn')(dense1)
        dense1 = keras.layers.Dropout(0.25, name='dense1_dropout')(dense1)
        
        dense2 = keras.layers.Dense(
            40,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.005),
            name='dense2'
        )(dense1)
        dense2 = keras.layers.BatchNormalization(name='dense2_bn')(dense2)
        dense2 = keras.layers.Dropout(0.2, name='dense2_dropout')(dense2)
        
        # Multi-task outputs (enhanced for NO2)
        # NO2 head (enhanced)
        no2_head = keras.layers.Dense(32, activation='relu', name='no2_head')(dense2)
        no2_head = keras.layers.BatchNormalization(name='no2_head_bn')(no2_head)
        no2_head = keras.layers.Dropout(0.15, name='no2_head_dropout')(no2_head)
        no2_output = keras.layers.Dense(1, activation='linear', name='no2_output')(no2_head)
        
        # O3 head
        o3_head = keras.layers.Dense(16, activation='relu', name='o3_head')(dense2)
        o3_head = keras.layers.Dropout(0.1, name='o3_head_dropout')(o3_head)
        o3_output = keras.layers.Dense(1, activation='linear', name='o3_output')(o3_head)
        
        # Combine outputs
        outputs = keras.layers.Concatenate(name='outputs')([no2_output, o3_output])
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='EnhancedBaseline')
        
        # Enhanced loss function (focus on NO2 improvement)
        def enhanced_loss(y_true, y_pred):
            no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
            no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
            
            # Use MAE for better stability
            no2_loss = keras.losses.MeanAbsoluteError()(no2_true, no2_pred)
            o3_loss = keras.losses.MeanAbsoluteError()(o3_true, o3_pred)
            
            # Weight NO2 more heavily for improvement
            return 2.0 * no2_loss + 1.0 * o3_loss
        
        # Compile with careful optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0005,  # Lower learning rate for stability
                clipnorm=1.0
            ),
            loss=enhanced_loss,
            metrics=['mae']
        )
        
        print(f"  Enhanced model created with {model.count_params():,} parameters")
        return model

def main():
    """Main execution function"""
    
    print("🎯 BASELINE MODEL ENHANCEMENT")
    print("="*60)
    print("Strategy: Verify baseline and enhance existing trained model")
    print("="*60)
    
    # Initialize enhancer
    enhancer = BaselineModelEnhancer()
    
    # Load existing model and data
    original_model, df = enhancer.load_existing_model_and_data()
    if original_model is None or df is None:
        print("❌ Could not load baseline model or data!")
        return
    
    # Recreate original sequences
    X, y, feature_names = enhancer.recreate_original_sequences(df)
    
    if len(X) == 0:
        print("❌ No sequences created!")
        return
    
    # Verify baseline performance
    baseline_results, X_test, y_test = enhancer.verify_baseline_performance(original_model, X, y)
    
    if baseline_results is None:
        print("❌ Could not verify baseline!")
        return
    
    # Display baseline results
    print(f"\n{'='*60}")
    print("📊 BASELINE VERIFICATION RESULTS")
    print(f"{'='*60}")
    
    for pollutant, metrics in baseline_results.items():
        print(f"\n{pollutant} Baseline:")
        for metric, value in metrics.items():
            if 'Accuracy' in metric:
                print(f"  {metric}: {value*100:.1f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Check if we can create and train enhanced model
    if len(X) > 1000:  # Need sufficient data
        print(f"\n🏗️ Training enhanced model...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test_new, y_val, y_test_new = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test_new)} samples")
        
        # Create enhanced model
        enhanced_model = enhancer.create_enhanced_model_from_baseline(
            original_model, len(feature_names)
        )
        
        # Use RobustScaler for enhanced preprocessing
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        # Scale features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test_new.reshape(-1, n_features)
        
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
        X_val_scaled = feature_scaler.transform(X_val_reshaped).reshape(-1, n_timesteps, n_features)
        X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(-1, n_timesteps, n_features)
        
        # Scale targets
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_val_scaled = target_scaler.transform(y_val)
        
        # Enhanced training with careful callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.8,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='enhanced_baseline_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train enhanced model
        print("  🚀 Training enhanced baseline...")
        history = enhanced_model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=60,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate enhanced model
        y_pred_scaled = enhanced_model.predict(X_test_scaled, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate enhanced results
        enhanced_results = {}
        
        for i, pollutant in enumerate(['NO2', 'O3']):
            y_true_i = y_test_new[:, i]
            y_pred_i = y_pred[:, i]
            
            # Core metrics
            mse = np.mean((y_true_i - y_pred_i) ** 2)
            mae = np.mean(np.abs(y_true_i - y_pred_i))
            rmse = np.sqrt(mse)
            
            # R²
            ss_res = np.sum((y_true_i - y_pred_i) ** 2)
            ss_tot = np.sum((y_true_i - np.mean(y_true_i)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Accuracy calculation
            abs_errors = np.abs(y_true_i - y_pred_i)
            abs_true = np.abs(y_true_i)
            threshold_base = np.maximum(abs_true, np.percentile(abs_true, 5))
            
            acc_10 = np.mean(abs_errors <= 0.10 * threshold_base)
            acc_20 = np.mean(abs_errors <= 0.20 * threshold_base)
            
            enhanced_results[pollutant] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_10%': acc_10,
                'Accuracy_20%': acc_20
            }
        
        # Display enhanced results and comparison
        print(f"\n{'='*60}")
        print("🏆 ENHANCED MODEL RESULTS")
        print(f"{'='*60}")
        
        for pollutant in ['NO2', 'O3']:
            baseline_acc = baseline_results[pollutant]['Accuracy_10%'] * 100
            enhanced_acc = enhanced_results[pollutant]['Accuracy_10%'] * 100
            improvement = enhanced_acc - baseline_acc
            
            print(f"\n{pollutant} Comparison:")
            print(f"  Baseline (10%±): {baseline_acc:.1f}%")
            print(f"  Enhanced (10%±): {enhanced_acc:.1f}%")
            if improvement > 0:
                print(f"  ✅ Improvement: +{improvement:.1f} percentage points")
            else:
                print(f"  📉 Change: {improvement:.1f} percentage points")
        
        # Overall assessment
        no2_baseline = baseline_results['NO2']['Accuracy_10%'] * 100
        no2_enhanced = enhanced_results['NO2']['Accuracy_10%'] * 100
        target = 45.0
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"  NO2 Target: {target:.1f}%")
        print(f"  NO2 Baseline: {no2_baseline:.1f}%")
        print(f"  NO2 Enhanced: {no2_enhanced:.1f}%")
        
        if no2_enhanced >= target:
            print(f"  ✅ TARGET ACHIEVED!")
        elif no2_enhanced > no2_baseline:
            print(f"  📈 Progress made toward target")
        else:
            print(f"  📊 Further optimization needed")
    
    print(f"\n{'='*60}")
    print("✨ Baseline enhancement complete!")

if __name__ == "__main__":
    main()
