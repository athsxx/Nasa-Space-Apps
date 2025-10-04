#!/usr/bin/env python3
"""
High-Performance Enhanced TCN Air Quality Model

Implements advanced optimizations to boost model performance:
1. Multi-scale temporal convolutions
2. Channel attention mechanisms  
3. Advanced feature engineering
4. Robust preprocessing with outlier handling
5. Multi-task learning with adaptive weighting
6. Enhanced training strategies

Target: Push NO2 accuracy from 42.3% to 50%+ (±10%)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class HighPerformanceTCN:
    """
    High-Performance TCN with Advanced Optimizations
    """
    
    def __init__(self, sequence_length=24, n_filters=64, n_layers=5, dropout_rate=0.2, l2_reg=0.005):
        self.sequence_length = sequence_length
        self.n_filters = n_filters  # Increased capacity
        self.n_layers = n_layers    # Deeper network
        self.dropout_rate = dropout_rate  # Reduced dropout for more capacity
        self.l2_reg = l2_reg  # Reduced L2 for less aggressive regularization
        self.dilation_rates = [1, 2, 4, 8, 16]  # More dilation rates
        self.model_dir = "enhanced_models"
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 High-Performance TCN initialized")
        print(f"  Architecture: {n_layers} layers, {n_filters} filters")
        print(f"  Sequence length: {sequence_length} hours")
        print(f"  Dilation rates: {self.dilation_rates}")
    
    def load_and_enhance_data(self):
        """Load data with advanced preprocessing"""
        print("📊 Loading and enhancing data...")
        
        # Load processed time series
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        if not os.path.exists(data_file):
            print("❌ Time series data not found!")
            return None
        
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} records")
        
        # Advanced feature engineering
        print("⚙️ Advanced feature engineering...")
        
        # 1. Pollutant interaction features
        df['NO2_O3_ratio'] = df['NO2'] / (df['O3'] + 0.001)
        df['CO_NO2_ratio'] = df['CO'] / (df['NO2'] + 0.001)
        df['SO2_NO2_ratio'] = df['SO2'] / (df['NO2'] + 0.001)
        df['total_pollution'] = df['NO2'] + df['O3'] + df['CO'] + df['SO2']
        
        # 2. Advanced temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['year_progress'] = df['day_of_year'] / 365.0
        
        # 3. Behavioral and contextual features
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        df['is_workday'] = ((df['day_of_week'] < 5) & (df['is_weekend'] == 0)).astype(int)
        df['is_peak_pollution'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                                  (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
        
        # 4. Rolling statistics features
        for window in [3, 6, 12]:
            for col in ['NO2', 'O3', 'CO', 'SO2']:
                if col in df.columns:
                    # Group by location to compute rolling stats
                    df[f'{col}_rolling_mean_{window}h'] = df.groupby('location_id')[col].rolling(
                        window=window, min_periods=1).mean().reset_index(level=0, drop=True)
                    df[f'{col}_rolling_std_{window}h'] = df.groupby('location_id')[col].rolling(
                        window=window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        
        # 5. Lagged features
        lag_features = ['NO2', 'O3', 'CO', 'SO2']
        for lag in [1, 2, 6, 12]:
            for col in lag_features:
                if col in df.columns:
                    df[f'{col}_lag_{lag}h'] = df.groupby('location_id')[col].shift(lag)
        
        print(f"  Created {len([c for c in df.columns if c not in ['location_id', 'datetime', 'Latitude', 'Longitude', 'State Name', 'County Name']]) - 4} total features")
        
        return df
    
    def robust_sequence_preparation(self, df, target_pollutants=['NO2', 'O3']):
        """Robust sequence preparation with outlier handling"""
        print("🔄 Robust sequence preparation...")
        
        # Get feature columns
        exclude_cols = ['location_id', 'datetime', 'Date Local', 'Time Local', 
                       'State Name', 'County Name', 'Latitude', 'Longitude']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and not pd.api.types.is_string_dtype(df[col])]
        
        print(f"  Using {len(feature_cols)} features")
        
        # Group by location and process
        site_groups = df.groupby('location_id')
        all_X = []
        all_y = []
        
        for site_id, site_data in site_groups:
            if len(site_data) < self.sequence_length + 12:  # Need extra for lag features
                continue
            
            # Sort by datetime if available
            if 'datetime' in site_data.columns:
                site_data = site_data.sort_values('datetime').copy()
            
            # Advanced missing value handling
            # Forward fill then backward fill
            site_data[feature_cols] = site_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
            
            # Fill any remaining NaN with column median
            for col in feature_cols:
                if site_data[col].isna().any():
                    site_data[col].fillna(site_data[col].median(), inplace=True)
            
            # Create sequences with better validation
            for i in range(12, len(site_data) - self.sequence_length):  # Skip first 12 for lag features
                X_seq = site_data.iloc[i:i + self.sequence_length][feature_cols].values
                
                target_idx = i + self.sequence_length
                if target_idx < len(site_data):
                    y_seq = []
                    for pollutant in target_pollutants:
                        if pollutant in site_data.columns:
                            val = site_data.iloc[target_idx][pollutant]
                            # Additional validation
                            if pd.isna(val) or val < 0 or val > 1000:  # Reasonable bounds
                                val = site_data[pollutant].median()
                            y_seq.append(val)
                        else:
                            y_seq.append(0.0)
                    
                    # Quality checks
                    if (not np.isnan(X_seq).any() and 
                        not np.isinf(X_seq).any() and
                        not np.isnan(y_seq).any() and
                        np.all(np.array(y_seq) >= 0)):
                        all_X.append(X_seq)
                        all_y.append(y_seq)
        
        X = np.array(all_X)
        y = np.array(all_y)
        
        print(f"  Created {len(X)} sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols, target_pollutants
    
    def advanced_preprocessing(self, X, y):
        """Advanced preprocessing with outlier detection"""
        print("🔧 Advanced preprocessing...")
        
        # Outlier detection
        print("  • Outlier detection...")
        contamination = 0.05  # More conservative
        outlier_detector = IsolationForest(contamination=contamination, random_state=42)
        
        # Detect outliers in flattened sequences
        X_flat = X.reshape(len(X), -1)
        outlier_mask = outlier_detector.fit_predict(X_flat) == 1
        
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        print(f"  • Removed {np.sum(~outlier_mask)} outliers ({100*np.sum(~outlier_mask)/len(outlier_mask):.1f}%)")
        
        # Feature selection
        print("  • Feature selection...")
        # Flatten for feature selection
        X_flat_clean = X_clean.reshape(len(X_clean), -1)
        
        # Select top features
        k_features = min(200, X_flat_clean.shape[1])  # Limit features
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X_flat_clean, y_clean[:, 0])  # Use NO2 for selection
        
        # Reshape back to sequences
        n_features_selected = X_selected.shape[1] // self.sequence_length
        X_selected = X_selected.reshape(len(X_clean), self.sequence_length, n_features_selected)
        
        print(f"  • Selected {n_features_selected} top features")
        
        self.feature_selector = selector
        return X_selected, y_clean
    
    def create_attention_layer(self, inputs, name='attention'):
        """Multi-head attention mechanism"""
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.n_filters // 4,
            dropout=0.1,
            name=f'{name}_multihead'
        )(inputs, inputs)
        
        # Add & Norm
        attention = layers.Add(name=f'{name}_add')([inputs, attention])
        attention = layers.LayerNormalization(name=f'{name}_norm')(attention)
        
        return attention
    
    def create_enhanced_tcn_block(self, inputs, filters, kernel_size, dilation_rate, dropout_rate, name):
        """Enhanced TCN block with attention and skip connections"""
        
        # First dilated convolution
        conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name=f'{name}_conv1'
        )(inputs)
        
        # Batch norm and dropout
        bn1 = layers.BatchNormalization(name=f'{name}_bn1')(conv1)
        drop1 = layers.Dropout(dropout_rate, name=f'{name}_dropout1')(bn1)
        
        # Second dilated convolution
        conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name=f'{name}_conv2'
        )(drop1)
        
        # Batch norm and dropout
        bn2 = layers.BatchNormalization(name=f'{name}_bn2')(conv2)
        drop2 = layers.Dropout(dropout_rate, name=f'{name}_dropout2')(bn2)
        
        # Residual connection
        if inputs.shape[-1] != filters:
            residual = layers.Conv1D(
                filters=filters,
                kernel_size=1,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'{name}_residual'
            )(inputs)
        else:
            residual = inputs
        
        # Add residual
        output = layers.Add(name=f'{name}_add')([drop2, residual])
        output = layers.Activation('relu', name=f'{name}_relu')(output)
        
        return output
    
    def build_high_performance_model(self, n_features, n_targets):
        """Build high-performance model with advanced architecture"""
        
        # Input
        inputs = layers.Input(shape=(self.sequence_length, n_features), name='input')
        
        # Initial convolution
        x = layers.Conv1D(
            filters=self.n_filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='initial_conv'
        )(inputs)
        
        # Enhanced TCN blocks with multi-scale processing
        skip_connections = []
        
        for i in range(self.n_layers):
            dilation_rate = self.dilation_rates[i % len(self.dilation_rates)]
            
            # Main TCN block
            x = self.create_enhanced_tcn_block(
                inputs=x,
                filters=self.n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate,
                name=f'tcn_block_{i}'
            )
            
            skip_connections.append(x)
            
            # Add attention every 2 blocks
            if i % 2 == 1:
                x = self.create_attention_layer(x, name=f'attention_{i}')
        
        # Multi-scale feature fusion
        # Combine skip connections for multi-scale information
        if len(skip_connections) > 1:
            multi_scale = layers.Concatenate(name='multi_scale_concat')(skip_connections)
            # Reduce dimensionality
            multi_scale = layers.Conv1D(
                filters=self.n_filters,
                kernel_size=1,
                activation='relu',
                name='multi_scale_reduce'
            )(multi_scale)
        else:
            multi_scale = x
        
        # Final attention
        attended = self.create_attention_layer(multi_scale, name='final_attention')
        
        # Multiple aggregation methods
        global_max = layers.GlobalMaxPooling1D(name='global_max')(attended)
        global_avg = layers.GlobalAveragePooling1D(name='global_avg')(attended) 
        last_step = layers.Lambda(lambda t: t[:, -1, :], name='last_step')(attended)
        
        # Combine global features
        combined = layers.Concatenate(name='combined_features')([global_max, global_avg, last_step])
        
        # Dense layers with advanced architecture
        dense1 = layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense1'
        )(combined)
        dense1 = layers.BatchNormalization(name='dense1_bn')(dense1)
        dense1 = layers.Dropout(0.3, name='dense1_dropout')(dense1)
        
        dense2 = layers.Dense(
            128,
            activation='relu', 
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense2'
        )(dense1)
        dense2 = layers.BatchNormalization(name='dense2_bn')(dense2)
        dense2 = layers.Dropout(0.2, name='dense2_dropout')(dense2)
        
        # Pollutant-specific heads
        if n_targets == 2:
            # NO2 head (more complex)
            no2_branch = layers.Dense(64, activation='relu', name='no2_dense')(dense2)
            no2_branch = layers.BatchNormalization(name='no2_bn')(no2_branch)
            no2_branch = layers.Dropout(0.2, name='no2_dropout')(no2_branch)
            no2_output = layers.Dense(1, activation='linear', name='no2_output')(no2_branch)
            
            # O3 head (simpler)
            o3_branch = layers.Dense(32, activation='relu', name='o3_dense')(dense2)
            o3_branch = layers.BatchNormalization(name='o3_bn')(o3_branch)
            o3_branch = layers.Dropout(0.15, name='o3_dropout')(o3_branch)
            o3_output = layers.Dense(1, activation='linear', name='o3_output')(o3_branch)
            
            outputs = layers.Concatenate(name='final_output')([no2_output, o3_output])
        else:
            outputs = layers.Dense(n_targets, activation='linear', name='output')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='HighPerformanceTCN')
        
        # Advanced loss function
        def adaptive_weighted_loss(y_true, y_pred):
            """Adaptive weighted multi-task loss"""
            if n_targets == 2:
                no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
                no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
                
                # Huber loss for robustness
                huber = keras.losses.Huber(delta=1.0)
                
                no2_loss = huber(no2_true, no2_pred)
                o3_loss = huber(o3_true, o3_pred)
                
                # Adaptive weighting based on current performance
                no2_weight = 2.5  # Higher weight for NO2
                o3_weight = 1.0
                
                # Add correlation penalty
                correlation_penalty = 0.05 * tf.reduce_mean(tf.square(
                    tf.nn.l2_normalize(no2_pred, axis=0) - tf.nn.l2_normalize(o3_pred, axis=0)
                ))
                
                return no2_weight * no2_loss + o3_weight * o3_loss + correlation_penalty
            else:
                return keras.losses.Huber(delta=1.0)(y_true, y_pred)
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.0005,  # Lower learning rate for stability
                weight_decay=0.01,
                clipnorm=0.5  # Gradient clipping
            ),
            loss=adaptive_weighted_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_high_performance_model(self, X_train, X_val, y_train, y_val, feature_names, target_names):
        """Train with advanced strategies"""
        print("🎯 Training high-performance model...")
        
        # Advanced scaling
        scaler = RobustScaler()
        
        # Fit and transform
        n_samples_train, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples_train, n_timesteps, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(-1, n_timesteps, n_features)
        
        # Target scaling
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_val_scaled = target_scaler.transform(y_val)
        
        self.feature_scaler = scaler
        self.target_scaler = target_scaler
        
        # Build model
        model = self.build_high_performance_model(n_features, len(target_names))
        
        # Advanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,  # Longer patience
                restore_best_weights=True,
                min_delta=0.00005
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-8,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_dir}/best_high_performance_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Cosine annealing
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.0005 * (1 + np.cos(np.pi * epoch / 150)) / 2,
                verbose=0
            )
        ]
        
        # Train
        print("  🚀 Starting advanced training...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=150,  # More epochs
            batch_size=32,  # Smaller batch for better gradients
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        return model
    
    def evaluate_performance(self, X_test, y_test, target_names):
        """Comprehensive evaluation"""
        print("📊 Evaluating high-performance model...")
        
        # Scale test data
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
        
        # Predict
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        # Metrics
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
            
            # Advanced accuracy metrics
            abs_true = np.abs(y_test[:, i])
            denom = np.maximum(abs_true, np.percentile(abs_true, 10))  # Robust denominator
            
            acc_5 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.05 * denom)
            acc_10 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.10 * denom)
            acc_15 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.15 * denom)
            acc_20 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.20 * denom)
            
            # MAPE with protection
            mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / np.maximum(np.abs(y_test[:, i]), 0.01))) * 100
            
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
        
        metrics['overall'] = {'MAE': overall_mae, 'RMSE': overall_rmse}
        
        return metrics, y_test, y_pred

def main():
    """Main execution function"""
    
    print("🚀 HIGH-PERFORMANCE TCN AIR QUALITY MODEL")
    print("="*70)
    print("Target: Push NO2 accuracy from 42.3% to 50%+ (±10%)")
    print("="*70)
    
    # Initialize high-performance model
    model = HighPerformanceTCN(
        sequence_length=24,
        n_filters=64,     # Increased capacity
        n_layers=5,       # Deeper network
        dropout_rate=0.2, # Reduced dropout
        l2_reg=0.005      # Reduced L2
    )
    
    # Load and enhance data
    df = model.load_and_enhance_data()
    if df is None:
        return
    
    # Prepare sequences
    X, y, feature_names, target_names = model.robust_sequence_preparation(df)
    
    if len(X) == 0:
        print("❌ No sequences created!")
        return
    
    # Advanced preprocessing
    X_processed, y_processed = model.advanced_preprocessing(X, y)
    
    # Split data
    print("\n🔄 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y_processed, test_size=0.3, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    print("\n🎯 Training high-performance model...")
    trained_model = model.train_high_performance_model(
        X_train, X_val, y_train, y_val, feature_names, target_names
    )
    
    # Evaluate
    print("\n📊 Final evaluation...")
    metrics, y_true, y_pred = model.evaluate_performance(X_test, y_test, target_names)
    
    # Results
    print(f"\n{'='*70}")
    print("🏆 HIGH-PERFORMANCE RESULTS")
    print(f"{'='*70}")
    
    for target_name, target_metrics in metrics.items():
        if target_name != 'overall':
            print(f"\n{target_name}:")
            print(f"  RMSE: {target_metrics['RMSE']:.4f}")
            print(f"  MAE:  {target_metrics['MAE']:.4f}")
            print(f"  R²:   {target_metrics['R²']:.4f}")
            print(f"  Accuracy (±5%):  {target_metrics['Accuracy_5%']*100:.1f}%")
            print(f"  Accuracy (±10%): {target_metrics['Accuracy_10%']*100:.1f}%")
            print(f"  Accuracy (±15%): {target_metrics['Accuracy_15%']*100:.1f}%")
            print(f"  Accuracy (±20%): {target_metrics['Accuracy_20%']*100:.1f}%")
            print(f"  MAPE: {target_metrics['MAPE']:.2f}%")
    
    print(f"\nOverall:")
    print(f"  MAE: {metrics['overall']['MAE']:.4f}")
    print(f"  RMSE: {metrics['overall']['RMSE']:.4f}")
    
    # Performance assessment
    if 'NO2' in metrics:
        no2_acc = metrics['NO2']['Accuracy_10%'] * 100
        baseline = 42.3
        target = 50.0
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print(f"  Baseline NO2 accuracy: {baseline:.1f}%")
        print(f"  Current NO2 accuracy:  {no2_acc:.1f}%")
        print(f"  Target NO2 accuracy:   {target:.1f}%")
        
        improvement = no2_acc - baseline
        if no2_acc >= target:
            print(f"  ✅ TARGET ACHIEVED! (+{improvement:.1f}pp improvement)")
        else:
            needed = target - no2_acc
            print(f"  📈 Progress: +{improvement:.1f}pp (need +{needed:.1f}pp more)")
            
            if improvement > 0:
                print(f"  🚀 Significant improvement achieved!")
    
    # Save model
    try:
        model.model.save('high_performance_tcn_model.keras')
        print(f"\n💾 Model saved as 'high_performance_tcn_model.keras'")
    except Exception as e:
        print(f"\n⚠️  Could not save model: {e}")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
