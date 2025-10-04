#!/usr/bin/env python3
"""
Realistic Performance Enhancement for TCN Model

Works with the actual available columns in the dataset.
Uses only available features while implementing key optimizations.

Target: Improve NO2 accuracy from 42.3% to 46%+ (±10%)
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

class RealisticPerformanceBoost:
    """
    Realistic performance boost using only available columns
    """
    
    def __init__(self):
        print("🚀 Realistic Performance Boost initialized")
        print("  Targeting NO2 accuracy improvement: 42.3% → 46%+ (±10%)")
    
    def load_and_enhance_data(self):
        """Load data and create enhanced features from available columns"""
        print("📊 Loading and enhancing data...")
        
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        if not os.path.exists(data_file):
            print("❌ Data file not found!")
            return None
        
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} records from {df['location_id'].nunique()} locations")
        
        # Select locations with decent data quality
        print("  • Filtering for locations with sufficient data...")
        
        good_locations = []
        for location_id, group in df.groupby('location_id'):
            no2_count = group['NO2'].count()
            o3_count = group['O3'].count()
            total_count = len(group)
            
            # Relaxed criteria: 30%+ data availability and 30+ records
            if (no2_count / total_count >= 0.3 and 
                o3_count / total_count >= 0.3 and 
                total_count >= 30):
                good_locations.append(location_id)
        
        df_filtered = df[df['location_id'].isin(good_locations)].copy()
        
        print(f"  • Selected {len(good_locations)} locations")
        print(f"  • Filtered dataset: {len(df_filtered)} records")
        
        # Create enhanced features from available columns
        print("  • Creating enhanced features from available data...")
        
        # Enhanced pollutant interactions
        df_filtered['NO2_CO_interaction'] = df_filtered['NO2'] * df_filtered['CO']
        df_filtered['O3_NO2_interaction'] = df_filtered['O3'] * df_filtered['NO2']
        df_filtered['total_pollutants'] = (df_filtered['NO2'].fillna(0) + 
                                          df_filtered['O3'].fillna(0) + 
                                          df_filtered['CO'].fillna(0) + 
                                          df_filtered['SO2'].fillna(0))
        
        # Enhanced temporal patterns
        df_filtered['day_sin'] = np.sin(2 * np.pi * df_filtered['day_of_week'] / 7)
        df_filtered['day_cos'] = np.cos(2 * np.pi * df_filtered['day_of_week'] / 7)
        df_filtered['year_progress'] = df_filtered['day_of_year'] / 365.25
        df_filtered['year_sin'] = np.sin(2 * np.pi * df_filtered['year_progress'])
        df_filtered['year_cos'] = np.cos(2 * np.pi * df_filtered['year_progress'])
        
        # Enhanced behavioral patterns  
        df_filtered['is_workday'] = ((df_filtered['day_of_week'] < 5) & 
                                    (df_filtered['is_weekend'] == 0)).astype(float)
        df_filtered['is_morning_rush'] = df_filtered['hour'].isin([7, 8, 9]).astype(float)
        df_filtered['is_evening_rush'] = df_filtered['hour'].isin([16, 17, 18, 19]).astype(float)
        df_filtered['is_midday'] = df_filtered['hour'].isin([10, 11, 12, 13, 14, 15]).astype(float)
        df_filtered['is_early_morning'] = df_filtered['hour'].isin([4, 5, 6]).astype(float)
        
        # Location-based features
        df_filtered['lat_normalized'] = (df_filtered['Latitude'] - df_filtered['Latitude'].mean()) / df_filtered['Latitude'].std()
        df_filtered['lon_normalized'] = (df_filtered['Longitude'] - df_filtered['Longitude'].mean()) / df_filtered['Longitude'].std()
        df_filtered['location_cluster'] = (df_filtered['lat_normalized'] * df_filtered['lon_normalized']).fillna(0)
        
        # Advanced pollutant ratios (with safe division)
        epsilon = 1e-3
        df_filtered['CO_SO2_ratio'] = df_filtered['CO'] / (df_filtered['SO2'].abs() + epsilon)
        df_filtered['NO2_SO2_ratio'] = df_filtered['NO2'] / (df_filtered['SO2'].abs() + epsilon)
        df_filtered['O3_SO2_ratio'] = df_filtered['O3'] / (df_filtered['SO2'].abs() + epsilon)
        
        # Pollution severity indicators
        df_filtered['high_NO2'] = (df_filtered['NO2'] > df_filtered['NO2'].quantile(0.75)).astype(float)
        df_filtered['high_O3'] = (df_filtered['O3'] > df_filtered['O3'].quantile(0.75)).astype(float)
        df_filtered['high_pollution_day'] = ((df_filtered['high_NO2'] == 1) | 
                                           (df_filtered['high_O3'] == 1)).astype(float)
        
        print(f"  • Added 20 enhanced features")
        
        return df_filtered
    
    def create_sequences_with_smart_imputation(self, df, sequence_length=24, target_pollutants=['NO2', 'O3']):
        """Create sequences with smart imputation strategies"""
        print("🔄 Creating sequences with smart imputation...")
        
        # Feature selection from available columns
        exclude_cols = ['location_id', 'datetime', 'State Name', 'County Name']
        
        # Select all numeric columns except targets and excluded ones
        feature_cols = []
        for col in df.columns:
            if (col not in exclude_cols and 
                col not in target_pollutants and
                not pd.api.types.is_string_dtype(df[col])):
                feature_cols.append(col)
        
        print(f"  Using {len(feature_cols)} features")
        print(f"  Feature columns: {feature_cols[:5]}... (showing first 5)")
        
        all_X = []
        all_y = []
        
        # Process each location
        for location_id, site_data in df.groupby('location_id'):
            if len(site_data) < sequence_length + 10:
                continue
            
            # Sort by datetime if possible
            if 'datetime' in site_data.columns:
                try:
                    site_data = site_data.sort_values('datetime').copy()
                except:
                    pass
            
            site_data = site_data.reset_index(drop=True)
            
            # Smart imputation strategy
            site_data_imputed = site_data.copy()
            
            # Impute features
            for col in feature_cols:
                if col in site_data_imputed.columns:
                    # Forward fill first
                    site_data_imputed[col] = site_data_imputed[col].fillna(method='ffill')
                    # Backward fill
                    site_data_imputed[col] = site_data_imputed[col].fillna(method='bfill')
                    # Use median for remaining nulls
                    if site_data_imputed[col].isnull().any():
                        median_val = site_data_imputed[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        site_data_imputed[col].fillna(median_val, inplace=True)
            
            # Impute targets with location-specific strategy
            for pollutant in target_pollutants:
                if pollutant in site_data_imputed.columns:
                    # Use forward fill for temporal continuity
                    site_data_imputed[pollutant] = site_data_imputed[pollutant].fillna(method='ffill')
                    # Backward fill for gaps
                    site_data_imputed[pollutant] = site_data_imputed[pollutant].fillna(method='bfill')
                    # Location median for remaining
                    if site_data_imputed[pollutant].isnull().any():
                        location_median = site_data_imputed[pollutant].median()
                        if pd.isna(location_median):
                            # Use global median
                            global_median = df[pollutant].median()
                            location_median = global_median if not pd.isna(global_median) else 20.0
                        site_data_imputed[pollutant].fillna(location_median, inplace=True)
            
            # Create sequences
            for i in range(len(site_data_imputed) - sequence_length):
                # Input sequence
                X_seq = site_data_imputed.iloc[i:i + sequence_length][feature_cols].values
                
                # Target values
                target_idx = i + sequence_length
                if target_idx < len(site_data_imputed):
                    y_seq = []
                    for pollutant in target_pollutants:
                        val = site_data_imputed.iloc[target_idx][pollutant]
                        y_seq.append(val)
                    
                    # Quality checks
                    if (not np.isnan(X_seq).any() and 
                        not np.isinf(X_seq).any() and
                        not any(np.isnan(y) or np.isinf(y) for y in y_seq)):
                        all_X.append(X_seq)
                        all_y.append(y_seq)
        
        X = np.array(all_X)
        y = np.array(all_y)
        
        print(f"  Created {len(X)} valid sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols, target_pollutants
    
    def create_realistic_model(self, n_features, n_targets, sequence_length=24):
        """Create realistic enhanced model"""
        print("🏗️ Building realistic enhanced model...")
        
        # Input layer
        inputs = keras.layers.Input(shape=(sequence_length, n_features), name='input')
        
        # Enhanced TCN architecture
        n_filters = 32
        dilation_rates = [1, 2, 4, 8]
        
        # Initial convolution with batch normalization
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='initial_conv'
        )(inputs)
        x = keras.layers.BatchNormalization(name='initial_bn')(x)
        
        # TCN blocks with residual connections
        for i, dilation_rate in enumerate(dilation_rates):
            # First dilated convolution
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
            conv1 = keras.layers.Dropout(0.25, name=f'drop1_{i}')(conv1)
            
            # Second dilated convolution
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
            conv2 = keras.layers.Dropout(0.25, name=f'drop2_{i}')(conv2)
            
            # Residual connection
            if x.shape[-1] != n_filters:
                residual = keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=1,
                    kernel_regularizer=keras.regularizers.l2(0.01),
                    name=f'residual_{i}'
                )(x)
            else:
                residual = x
            
            x = keras.layers.Add(name=f'add_{i}')([conv2, residual])
            x = keras.layers.Activation('relu', name=f'relu_{i}')(x)
        
        # Feature extraction with multiple approaches
        # Global average for overall pattern
        global_avg = keras.layers.GlobalAveragePooling1D(name='global_avg')(x)
        # Last timestep for recency bias
        last_step = keras.layers.Lambda(lambda t: t[:, -1, :], name='last_step')(x)
        # Combine for richer representation
        combined = keras.layers.Concatenate(name='combined')([global_avg, last_step])
        
        # Enhanced dense layers
        dense1 = keras.layers.Dense(
            80,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='dense1'
        )(combined)
        dense1 = keras.layers.BatchNormalization(name='dense1_bn')(dense1)
        dense1 = keras.layers.Dropout(0.4, name='dense1_dropout')(dense1)
        
        dense2 = keras.layers.Dense(
            40,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='dense2'
        )(dense1)
        dense2 = keras.layers.BatchNormalization(name='dense2_bn')(dense2)
        dense2 = keras.layers.Dropout(0.3, name='dense2_dropout')(dense2)
        
        # Multi-task outputs if needed
        if n_targets == 2:
            # NO2 prediction head (needs more capacity)
            no2_head = keras.layers.Dense(24, activation='relu', name='no2_dense')(dense2)
            no2_head = keras.layers.Dropout(0.2, name='no2_dropout')(no2_head)
            no2_output = keras.layers.Dense(1, activation='linear', name='no2_output')(no2_head)
            
            # O3 prediction head
            o3_head = keras.layers.Dense(16, activation='relu', name='o3_dense')(dense2)
            o3_head = keras.layers.Dropout(0.2, name='o3_dropout')(o3_head)
            o3_output = keras.layers.Dense(1, activation='linear', name='o3_output')(o3_head)
            
            outputs = keras.layers.Concatenate(name='outputs')([no2_output, o3_output])
        else:
            outputs = keras.layers.Dense(n_targets, activation='linear', name='outputs')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='RealisticTCN')
        
        # Enhanced loss function
        def realistic_multi_task_loss(y_true, y_pred):
            if n_targets == 2:
                no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
                no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
                
                # Use Huber loss for robustness
                huber = keras.losses.Huber(delta=2.0)
                
                no2_loss = huber(no2_true, no2_pred)
                o3_loss = huber(o3_true, o3_pred)
                
                # Weighted combination (emphasize NO2 improvement)
                return 2.2 * no2_loss + 1.0 * o3_loss
            else:
                return keras.losses.Huber(delta=2.0)(y_true, y_pred)
        
        # Compile with enhanced optimizer
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.001,
                weight_decay=0.01,
                clipnorm=1.0
            ),
            loss=realistic_multi_task_loss,
            metrics=['mae', 'mse']
        )
        
        print(f"  Model created with {model.count_params():,} parameters")
        return model
    
    def train_realistic_model(self, X_train, X_val, y_train, y_val, target_names):
        """Train with realistic configuration"""
        print("🎯 Training realistic enhanced model...")
        
        # Use RobustScaler for better outlier handling
        print("  • Using RobustScaler for preprocessing")
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
        
        # Build model
        model = self.create_realistic_model(n_features, len(target_names))
        
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='realistic_tcn_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        print("  🚀 Starting realistic training...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=60,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        return model
    
    def evaluate_realistic_model(self, X_test, y_test, target_names):
        """Evaluate realistic model performance"""
        print("📊 Evaluating realistic enhanced model...")
        
        # Scale test data
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate comprehensive metrics
        metrics = {}
        
        for i, target_name in enumerate(target_names):
            y_true_i = y_test[:, i]
            y_pred_i = y_pred[:, i]
            
            # Basic metrics
            mse = np.mean((y_true_i - y_pred_i) ** 2)
            mae = np.mean(np.abs(y_true_i - y_pred_i))
            rmse = np.sqrt(mse)
            
            # R-squared
            ss_res = np.sum((y_true_i - y_pred_i) ** 2)
            ss_tot = np.sum((y_true_i - np.mean(y_true_i)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Accuracy metrics (percentage within threshold)
            abs_true = np.abs(y_true_i)
            threshold_base = np.maximum(abs_true, np.percentile(abs_true, 10))
            
            acc_5 = np.mean(np.abs(y_true_i - y_pred_i) <= 0.05 * threshold_base)
            acc_10 = np.mean(np.abs(y_true_i - y_pred_i) <= 0.10 * threshold_base)
            acc_15 = np.mean(np.abs(y_true_i - y_pred_i) <= 0.15 * threshold_base)
            acc_20 = np.mean(np.abs(y_true_i - y_pred_i) <= 0.20 * threshold_base)
            
            # MAPE with protection against division by zero
            mape = np.mean(np.abs((y_true_i - y_pred_i) / 
                                 np.maximum(np.abs(y_true_i), 0.1))) * 100
            
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
        
        return metrics

def main():
    """Main execution function"""
    
    print("🚀 REALISTIC PERFORMANCE ENHANCEMENT")
    print("="*60)
    print("Target: Improve NO2 accuracy 42.3% → 46%+ (±10%)")
    print("="*60)
    
    # Initialize enhancement system
    booster = RealisticPerformanceBoost()
    
    # Load and enhance data
    df = booster.load_and_enhance_data()
    if df is None or len(df) == 0:
        print("❌ Could not load data!")
        return
    
    # Create sequences with imputation
    X, y, feature_names, target_names = booster.create_sequences_with_smart_imputation(df)
    
    if len(X) == 0:
        print("❌ No sequences created!")
        return
    
    # Split data for training
    print("\n🔄 Splitting data for training...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")  
    print(f"  Test: {len(X_test)} samples")
    
    # Train enhanced model
    model = booster.train_realistic_model(X_train, X_val, y_train, y_val, target_names)
    
    # Evaluate performance
    print("\n📊 Final evaluation...")
    metrics = booster.evaluate_realistic_model(X_test, y_test, target_names)
    
    # Display results
    print(f"\n{'='*60}")
    print("🏆 REALISTIC ENHANCEMENT RESULTS")
    print(f"{'='*60}")
    
    for target_name, target_metrics in metrics.items():
        print(f"\n{target_name} Performance:")
        for metric, value in target_metrics.items():
            if 'Accuracy' in metric:
                print(f"  {metric}: {value*100:.1f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Performance assessment
    if 'NO2' in metrics:
        no2_acc = metrics['NO2']['Accuracy_10%'] * 100
        baseline = 42.3
        target = 46.0
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print(f"  Baseline NO2 (10%±): {baseline:.1f}%")
        print(f"  Enhanced NO2 (10%±): {no2_acc:.1f}%")
        print(f"  Target NO2 (10%±): {target:.1f}%")
        
        improvement = no2_acc - baseline
        if no2_acc >= target:
            print(f"  ✅ TARGET ACHIEVED! (+{improvement:.1f} percentage points)")
        elif improvement > 0:
            print(f"  📈 Improvement: +{improvement:.1f} percentage points")
        else:
            print(f"  📉 Change: {improvement:.1f} percentage points")
    
    if 'O3' in metrics:
        o3_acc = metrics['O3']['Accuracy_10%'] * 100
        print(f"  Enhanced O3 (10%±): {o3_acc:.1f}%")
    
    print(f"\n{'='*60}")
    print("✨ Enhancement complete!")

if __name__ == "__main__":
    main()
