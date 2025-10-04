#!/usr/bin/env python3
"""
Simplified Performance Enhancement for TCN Model

Implements key optimizations that work with existing data:
1. RobustScaler for better outlier handling
2. Enhanced feature engineering  
3. Multi-task weighted loss
4. Improved training configuration

Target: Improve NO2 accuracy from 42.3% to 48%+ (±10%)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
import os

class SimplifiedPerformanceBoost:
    """
    Simplified performance boost for existing TCN model
    """
    
    def __init__(self):
        print("🚀 Simplified Performance Boost initialized")
        print("  Targeting NO2 accuracy improvement: 42.3% → 48%+ (±10%)")
    
    def enhance_existing_data(self):
        """Enhance existing data with better preprocessing"""
        print("📊 Loading and enhancing existing data...")
        
        # Load existing processed data
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        if not os.path.exists(data_file):
            print("❌ Data file not found!")
            return None
        
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} records")
        
        # Filter for locations with sufficient NO2 and O3 data
        print("  • Filtering for quality data...")
        
        # Group by location and check data availability
        good_locations = []
        for location_id, group in df.groupby('location_id'):
            no2_count = group['NO2'].count()
            o3_count = group['O3'].count()
            total_count = len(group)
            
            # Require at least 50% data completeness and 50+ records
            if (no2_count / total_count >= 0.5 and 
                o3_count / total_count >= 0.5 and 
                total_count >= 50):
                good_locations.append(location_id)
        
        # Filter to good locations
        df_filtered = df[df['location_id'].isin(good_locations)].copy()
        
        print(f"  • Kept {len(good_locations)} locations with quality data")
        print(f"  • Filtered dataset: {len(df_filtered)} records")
        
        # Enhanced feature engineering (simple version)
        print("  • Adding enhanced features...")
        
        # Safe feature creation with null handling
        df_filtered['NO2_O3_ratio'] = df_filtered['NO2'] / (df_filtered['O3'].abs() + 0.001)
        df_filtered['CO_NO2_ratio'] = df_filtered['CO'] / (df_filtered['NO2'].abs() + 0.001)
        df_filtered['total_pollution'] = (df_filtered['NO2'].fillna(0) + 
                                         df_filtered['O3'].fillna(0) + 
                                         df_filtered['CO'].fillna(0) + 
                                         df_filtered['SO2'].fillna(0))
        
        # Better temporal features
        df_filtered['hour_sin'] = np.sin(2 * np.pi * df_filtered['hour'] / 24)
        df_filtered['hour_cos'] = np.cos(2 * np.pi * df_filtered['hour'] / 24)
        df_filtered['day_sin'] = np.sin(2 * np.pi * df_filtered['day_of_week'] / 7)
        df_filtered['day_cos'] = np.cos(2 * np.pi * df_filtered['day_of_week'] / 7)
        df_filtered['month_sin'] = np.sin(2 * np.pi * df_filtered['month'] / 12)
        df_filtered['month_cos'] = np.cos(2 * np.pi * df_filtered['month'] / 12)
        
        # Behavioral features
        df_filtered['is_peak_hour'] = df_filtered['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        df_filtered['is_night'] = df_filtered['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        df_filtered['is_workday'] = ((df_filtered['day_of_week'] < 5) & 
                                    (df_filtered['is_weekend'] == 0)).astype(int)
        
        print(f"  • Added 10 enhanced features")
        
        return df_filtered
    
    def create_robust_sequences(self, df, sequence_length=24, target_pollutants=['NO2', 'O3']):
        """Create sequences with robust handling of missing data"""
        print("🔄 Creating robust sequences...")
        
        # Feature columns
        exclude_cols = ['location_id', 'datetime', 'Latitude', 'Longitude', 
                       'State Name', 'County Name']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and not pd.api.types.is_string_dtype(df[col])]
        
        print(f"  Using {len(feature_cols)} features")
        
        all_X = []
        all_y = []
        
        # Process each location
        for location_id, site_data in df.groupby('location_id'):
            if len(site_data) < sequence_length + 5:
                continue
            
            # Sort by datetime
            if 'datetime' in site_data.columns:
                site_data = site_data.sort_values('datetime').copy()
            
            # Forward fill then backward fill for features
            for col in feature_cols:
                if col in site_data.columns:
                    site_data[col] = site_data[col].fillna(method='ffill').fillna(method='bfill')
                    # Fill remaining with column mean
                    if site_data[col].isnull().any():
                        site_data[col].fillna(site_data[col].mean(), inplace=True)
            
            # Create sequences
            for i in range(len(site_data) - sequence_length):
                # Input sequence
                X_seq = site_data.iloc[i:i + sequence_length][feature_cols].values
                
                # Target values
                target_idx = i + sequence_length
                if target_idx < len(site_data):
                    y_seq = []
                    valid_target = True
                    
                    for pollutant in target_pollutants:
                        val = site_data.iloc[target_idx][pollutant]
                        if pd.isna(val):
                            valid_target = False
                            break
                        y_seq.append(val)
                    
                    # Quality checks
                    if (valid_target and 
                        not np.isnan(X_seq).any() and 
                        not np.isinf(X_seq).any() and
                        len(y_seq) == len(target_pollutants)):
                        all_X.append(X_seq)
                        all_y.append(y_seq)
        
        X = np.array(all_X)
        y = np.array(all_y)
        
        print(f"  Created {len(X)} valid sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols, target_pollutants
    
    def enhanced_preprocessing(self, X, y):
        """Enhanced preprocessing with outlier handling"""
        print("🔧 Enhanced preprocessing...")
        
        # Conservative outlier removal
        print("  • Outlier detection...")
        outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        
        X_flat = X.reshape(len(X), -1)
        outlier_mask = outlier_detector.fit_predict(X_flat) == 1
        
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        removed = np.sum(~outlier_mask)
        print(f"  • Removed {removed} outliers ({100*removed/len(outlier_mask):.1f}%)")
        
        return X_clean, y_clean
    
    def create_enhanced_model(self, n_features, n_targets, sequence_length=24):
        """Create enhanced TCN model"""
        print("🏗️ Building enhanced TCN model...")
        
        # Input
        inputs = keras.layers.Input(shape=(sequence_length, n_features), name='input')
        
        # Enhanced TCN architecture
        n_filters = 48  # Increased from 32
        dilation_rates = [1, 2, 4, 8]
        
        # Initial convolution
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='initial_conv'
        )(inputs)
        
        # Enhanced TCN blocks
        for i, dilation_rate in enumerate(dilation_rates):
            # First conv
            conv1 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.008),
                name=f'conv1_{i}'
            )(x)
            
            conv1 = keras.layers.BatchNormalization(name=f'bn1_{i}')(conv1)
            conv1 = keras.layers.Dropout(0.25, name=f'drop1_{i}')(conv1)
            
            # Second conv
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
            conv2 = keras.layers.Dropout(0.25, name=f'drop2_{i}')(conv2)
            
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
        
        # Enhanced feature extraction
        # Use last timestep (most relevant for next prediction)
        last_step = keras.layers.Lambda(lambda t: t[:, -1, :], name='last_step')(x)
        
        # Dense layers with enhanced architecture
        dense1 = keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.008),
            name='dense1'
        )(last_step)
        dense1 = keras.layers.BatchNormalization(name='dense1_bn')(dense1)
        dense1 = keras.layers.Dropout(0.4, name='dense1_dropout')(dense1)
        
        dense2 = keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.008),
            name='dense2'
        )(dense1)
        dense2 = keras.layers.BatchNormalization(name='dense2_bn')(dense2)
        dense2 = keras.layers.Dropout(0.3, name='dense2_dropout')(dense2)
        
        # Multi-task heads for better specialization
        if n_targets == 2:
            # NO2 head (more capacity for harder task)
            no2_head = keras.layers.Dense(48, activation='relu', name='no2_dense')(dense2)
            no2_head = keras.layers.BatchNormalization(name='no2_bn')(no2_head)
            no2_head = keras.layers.Dropout(0.25, name='no2_dropout')(no2_head)
            no2_output = keras.layers.Dense(1, activation='linear', name='no2_output')(no2_head)
            
            # O3 head (simpler for easier task)
            o3_head = keras.layers.Dense(24, activation='relu', name='o3_dense')(dense2)
            o3_head = keras.layers.BatchNormalization(name='o3_bn')(o3_head)
            o3_head = keras.layers.Dropout(0.2, name='o3_dropout')(o3_head)
            o3_output = keras.layers.Dense(1, activation='linear', name='o3_output')(o3_head)
            
            outputs = keras.layers.Concatenate(name='outputs')([no2_output, o3_output])
        else:
            outputs = keras.layers.Dense(n_targets, activation='linear', name='outputs')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='EnhancedTCN')
        
        # Enhanced multi-task loss
        def enhanced_weighted_loss(y_true, y_pred):
            """Enhanced weighted loss focusing on NO2"""
            if n_targets == 2:
                no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
                no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
                
                # Use Huber loss for robustness
                huber = keras.losses.Huber(delta=1.0)
                
                no2_loss = huber(no2_true, no2_pred)
                o3_loss = huber(o3_true, o3_pred)
                
                # Higher weight for NO2 (challenging pollutant)
                return 2.8 * no2_loss + 1.0 * o3_loss
            else:
                return keras.losses.Huber(delta=1.0)(y_true, y_pred)
        
        # Compile with enhanced optimizer
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.0008,
                weight_decay=0.01,
                clipnorm=0.8
            ),
            loss=enhanced_weighted_loss,
            metrics=['mae', 'mse']
        )
        
        print(f"  Model created with {model.count_params():,} parameters")
        return model
    
    def train_enhanced_model(self, X_train, X_val, y_train, y_val, target_names):
        """Train with enhanced configuration"""
        print("🎯 Training enhanced model...")
        
        # Use RobustScaler for better outlier handling
        print("  • Using RobustScaler for preprocessing")
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        # Fit and transform features
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
        
        # Build model
        model = self.create_enhanced_model(n_features, len(target_names))
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                min_delta=0.0001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=12,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='enhanced_tcn_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with enhanced configuration
        print("  🚀 Starting enhanced training...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=120,  # Increased epochs
            batch_size=48,  # Optimal batch size
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        return model
    
    def evaluate_enhanced_model(self, X_test, y_test, target_names):
        """Evaluate enhanced model"""
        print("📊 Evaluating enhanced model...")
        
        # Scale test data
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
        
        # Predict
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        metrics = {}
        
        for i, target_name in enumerate(target_names):
            # Basic metrics
            mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
            mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
            rmse = np.sqrt(mse)
            
            # R²
            ss_res = np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Enhanced accuracy metrics
            abs_true = np.abs(y_test[:, i])
            denom = np.maximum(abs_true, np.percentile(abs_true, 5))
            
            acc_5 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.05 * denom)
            acc_10 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.10 * denom)
            acc_15 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.15 * denom)
            acc_20 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.20 * denom)
            
            # MAPE
            mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / 
                                 np.maximum(np.abs(y_test[:, i]), 0.01))) * 100
            
            metrics[target_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_5%': acc_5,
                'Accuracy_10%': acc_10,
                'Accuracy_15%': acc_15,
                'Accuracy_20%': acc_20,
                'MAPE': mape
            }
        
        # Overall metrics
        overall_mae = np.mean(np.abs(y_test - y_pred))
        overall_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        metrics['overall'] = {
            'MAE': overall_mae,
            'RMSE': overall_rmse
        }
        
        return metrics

def main():
    """Main execution"""
    
    print("🚀 SIMPLIFIED PERFORMANCE ENHANCEMENT")
    print("="*60)
    print("Target: Improve NO2 accuracy 42.3% → 48%+ (±10%)")
    print("="*60)
    
    # Initialize
    booster = SimplifiedPerformanceBoost()
    
    # Load and enhance data
    df = booster.enhance_existing_data()
    if df is None:
        return
    
    # Create sequences
    X, y, feature_names, target_names = booster.create_robust_sequences(df)
    
    if len(X) == 0:
        print("❌ No sequences created!")
        return
    
    # Preprocessing
    X_processed, y_processed = booster.enhanced_preprocessing(X, y)
    
    # Split data
    print("\n🔄 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y_processed, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train
    print("\n🎯 Training enhanced model...")
    model = booster.train_enhanced_model(X_train, X_val, y_train, y_val, target_names)
    
    # Evaluate
    print("\n📊 Final evaluation...")
    metrics = booster.evaluate_enhanced_model(X_test, y_test, target_names)
    
    # Results
    print(f"\n{'='*60}")
    print("🏆 ENHANCED RESULTS")
    print(f"{'='*60}")
    
    for target_name, target_metrics in metrics.items():
        if target_name != 'overall':
            print(f"\n{target_name}:")
            for metric, value in target_metrics.items():
                if 'Accuracy' in metric:
                    print(f"  {metric}: {value*100:.1f}%")
                else:
                    print(f"  {metric}: {value:.4f}")
    
    print(f"\nOverall:")
    print(f"  MAE: {metrics['overall']['MAE']:.4f}")
    print(f"  RMSE: {metrics['overall']['RMSE']:.4f}")
    
    # Performance check
    if 'NO2' in metrics:
        no2_acc = metrics['NO2']['Accuracy_10%'] * 100
        baseline = 42.3
        target = 48.0
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print(f"  Baseline: {baseline:.1f}%")
        print(f"  Current:  {no2_acc:.1f}%")
        print(f"  Target:   {target:.1f}%")
        
        improvement = no2_acc - baseline
        if no2_acc >= target:
            print(f"  ✅ TARGET ACHIEVED! (+{improvement:.1f}pp)")
        else:
            print(f"  📈 Progress: +{improvement:.1f}pp")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
