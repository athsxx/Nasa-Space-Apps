#!/usr/bin/env python3
"""
Quick Performance Boost for TCN Air Quality Model

Implements targeted improvements to boost NO2 accuracy:
- Multi-task learning with NO2-focused weighting
- Robust data preprocessing
- Enhanced feature engineering
- Optimized training strategy

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

class QuickBoostTCN:
    """
    Quick performance boost implementation for TCN air quality model
    """
    
    def __init__(self, sequence_length=24, n_filters=48, n_layers=4, dropout_rate=0.25, l2_reg=0.008):
        self.sequence_length = sequence_length
        self.n_filters = n_filters  # Increased filters
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate  # Slightly reduced for better capacity
        self.l2_reg = l2_reg  # Slightly reduced regularization
        self.dilation_rates = [1, 2, 4, 8]
        
        # Optimization flags
        self.use_robust_scaling = True
        self.use_enhanced_features = True
        self.use_multi_task_weighting = True
        
        print(f"🚀 Quick Boost TCN initialized")
        print(f"  Architecture: {n_layers} layers, {n_filters} filters")
        print(f"  Optimizations: Robust scaling, Enhanced features, Multi-task weighting")
    
    def load_and_prepare_data(self):
        """Load and prepare data with enhancements"""
        print("📊 Loading processed time series data...")
        
        # Load the pre-processed time series data
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        
        if not os.path.exists(data_file):
            print("❌ Processed data not found. Please run the main model first.")
            return None, None, None, None
        
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} records")
        
        # Enhanced feature engineering
        if self.use_enhanced_features:
            print("⚙️ Creating enhanced features...")
            
            # Pollutant ratios (chemical relationships)
            df['NO2_O3_ratio'] = df['NO2'] / (df['O3'] + 0.001)
            df['CO_NO2_ratio'] = df['CO'] / (df['NO2'] + 0.001) 
            df['pollution_index'] = df['NO2'] + df['O3'] + df['CO'] + df['SO2']
            
            # Enhanced temporal features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Day type features
            df['is_workday'] = ((df['day_of_week'] < 5) & (df['is_weekend'] == 0)).astype(int)
            df['is_rush_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19]))).astype(int)
            
            print(f"  Added 9 enhanced features")
        
        # Prepare sequences
        target_pollutants = ['NO2', 'O3']
        X, y, feature_names, target_names = self.prepare_sequences(df, target_pollutants)
        
        return X, y, feature_names, target_names
    
    def prepare_sequences(self, df, target_pollutants):
        """Prepare sequences with outlier handling"""
        print("🔄 Preparing sequences...")
        
        # Group by location
        site_groups = df.groupby('location_id')
        
        # Feature columns (exclude metadata)
        exclude_cols = ['location_id', 'datetime', 'Date Local', 'Time Local', 
                       'State Name', 'County Name', 'Latitude', 'Longitude']
        feature_cols = [col for col in df.columns if col not in exclude_cols and not pd.api.types.is_string_dtype(df[col])]
        
        print(f"  Using {len(feature_cols)} features")
        
        all_X = []
        all_y = []
        
        for site_id, site_data in site_groups:
            if len(site_data) < self.sequence_length + 1:
                continue
            
            # Sort by datetime if available
            if 'datetime' in site_data.columns:
                site_data = site_data.sort_values('datetime').copy()
            
            # Fill missing values
            site_data[feature_cols] = site_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
            
            # Create sequences
            for i in range(len(site_data) - self.sequence_length):
                X_seq = site_data.iloc[i:i + self.sequence_length][feature_cols].values
                
                target_idx = i + self.sequence_length
                if target_idx < len(site_data):
                    y_seq = []
                    for pollutant in target_pollutants:
                        if pollutant in site_data.columns:
                            y_seq.append(site_data.iloc[target_idx][pollutant])
                        else:
                            y_seq.append(0.0)
                    
                    if not np.isnan(X_seq).any() and not np.isnan(y_seq).any():
                        all_X.append(X_seq)
                        all_y.append(y_seq)
        
        X = np.array(all_X)
        y = np.array(all_y)
        
        print(f"  Created {len(X)} sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols, target_pollutants
    
    def preprocess_data(self, X, y):
        """Enhanced data preprocessing"""
        print("🔧 Enhanced preprocessing...")
        
        # Outlier detection and removal
        print("  • Detecting outliers...")
        outlier_detector = IsolationForest(contamination=0.08, random_state=42)
        
        # Reshape for outlier detection
        X_reshaped = X.reshape(len(X), -1)
        outlier_mask = outlier_detector.fit_predict(X_reshaped) == 1
        
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        print(f"  • Removed {np.sum(~outlier_mask)} outliers ({100*np.sum(~outlier_mask)/len(outlier_mask):.1f}%)")
        
        return X_clean, y_clean
    
    def build_optimized_tcn(self, n_features, n_targets):
        """Build optimized TCN architecture"""
        
        # Input layer
        inputs = keras.layers.Input(shape=(self.sequence_length, n_features), name='input')
        
        # Initial convolution
        x = keras.layers.Conv1D(
            filters=self.n_filters,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(inputs)
        
        # TCN blocks with residual connections
        for i in range(self.n_layers):
            dilation_rate = self.dilation_rates[i]
            
            # Dilated convolution 1
            conv1 = keras.layers.Conv1D(
                filters=self.n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            )(x)
            
            # Batch normalization and dropout
            bn1 = keras.layers.BatchNormalization()(conv1)
            drop1 = keras.layers.Dropout(self.dropout_rate)(bn1)
            
            # Dilated convolution 2
            conv2 = keras.layers.Conv1D(
                filters=self.n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            )(drop1)
            
            # Batch normalization and dropout
            bn2 = keras.layers.BatchNormalization()(conv2)
            drop2 = keras.layers.Dropout(self.dropout_rate)(bn2)
            
            # Residual connection
            if x.shape[-1] == self.n_filters:
                residual = x
            else:
                residual = keras.layers.Conv1D(
                    filters=self.n_filters,
                    kernel_size=1,
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg)
                )(x)
            
            x = keras.layers.Add()([drop2, residual])
            x = keras.layers.Activation('relu')(x)
        
        # Feature extraction - use last timestep and global pooling
        last_step = keras.layers.Lambda(lambda t: t[:, -1, :])(x)
        
        # Dense layers with better architecture
        dense1 = keras.layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(last_step)
        bn_dense1 = keras.layers.BatchNormalization()(dense1)
        drop_dense1 = keras.layers.Dropout(0.4)(bn_dense1)
        
        dense2 = keras.layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(drop_dense1)
        bn_dense2 = keras.layers.BatchNormalization()(dense2)
        drop_dense2 = keras.layers.Dropout(0.3)(bn_dense2)
        
        # Multi-task output with specialized heads
        if self.use_multi_task_weighting and n_targets == 2:
            # NO2 head (more complex for harder task)
            no2_dense = keras.layers.Dense(48, activation='relu')(drop_dense2)
            no2_bn = keras.layers.BatchNormalization()(no2_dense)
            no2_drop = keras.layers.Dropout(0.25)(no2_bn)
            no2_out = keras.layers.Dense(1, activation='linear', name='no2_output')(no2_drop)
            
            # O3 head (simpler for easier task)
            o3_dense = keras.layers.Dense(32, activation='relu')(drop_dense2)
            o3_bn = keras.layers.BatchNormalization()(o3_dense)
            o3_drop = keras.layers.Dropout(0.2)(o3_bn)
            o3_out = keras.layers.Dense(1, activation='linear', name='o3_output')(o3_drop)
            
            outputs = keras.layers.Concatenate()([no2_out, o3_out])
        else:
            outputs = keras.layers.Dense(n_targets, activation='linear')(drop_dense2)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='OptimizedTCN')
        
        # Enhanced loss function
        if self.use_multi_task_weighting and n_targets == 2:
            def weighted_loss(y_true, y_pred):
                """Weighted loss focusing more on NO2 (harder task)"""
                no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2] 
                no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
                
                # Huber loss for robustness
                huber = keras.losses.Huber(delta=1.0)
                
                no2_loss = huber(no2_true, no2_pred)
                o3_loss = huber(o3_true, o3_pred)
                
                # Higher weight for NO2 (harder to predict)
                return 2.0 * no2_loss + 1.0 * o3_loss
            
            loss = weighted_loss
        else:
            loss = keras.losses.Huber(delta=1.0)
        
        # Compile with optimized settings
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.0008,  # Slightly lower LR
                weight_decay=0.01,
                clipnorm=0.8  # Gradient clipping
            ),
            loss=loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val, feature_names, target_names):
        """Train optimized model"""
        print("🎯 Training optimized TCN...")
        
        # Robust scaling
        if self.use_robust_scaling:
            scaler = RobustScaler()
            print("  • Using RobustScaler")
        else:
            scaler = StandardScaler()
            print("  • Using StandardScaler")
        
        # Scale features
        n_samples_train, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples_train, n_timesteps, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(-1, n_timesteps, n_features)
        
        # Scale targets for balanced learning
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_val_scaled = target_scaler.transform(y_val)
        
        # Store scalers
        self.feature_scaler = scaler
        self.target_scaler = target_scaler
        
        # Build model
        model = self.build_optimized_tcn(n_features, len(target_names))
        
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
                factor=0.5,
                patience=12,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='optimized_tcn_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("  🚀 Starting training with optimizations...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=120,  # More epochs for better convergence
            batch_size=80,  # Slightly larger batch
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        return model
    
    def evaluate_model(self, X_test, y_test, target_names):
        """Evaluate model performance"""
        print("📊 Evaluating optimized model...")
        
        # Scale test data
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
        
        # Predict
        y_pred_scaled = self.model.predict(X_test_scaled)
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
            
            # Accuracy metrics
            abs_true = np.abs(y_test[:, i])
            denom = np.maximum(abs_true, 0.01)  # Minimum threshold
            
            acc_10 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.1 * denom)
            acc_20 = np.mean(np.abs(y_test[:, i] - y_pred[:, i]) <= 0.2 * denom)
            
            # MAPE
            mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + 1e-7))) * 100
            
            metrics[target_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_10%': acc_10,
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
        
        return metrics, y_test, y_pred

def main():
    """Main function for quick boost optimization"""
    
    print("🚀 QUICK BOOST TCN OPTIMIZATION")
    print("="*60)
    print("Target: Improve NO2 accuracy from 42.3% to 48%+ (±10%)")
    print("="*60)
    
    # Initialize model
    model = QuickBoostTCN(
        sequence_length=24,
        n_filters=48,  # Increased capacity
        n_layers=4,
        dropout_rate=0.25,  # Reduced dropout
        l2_reg=0.008  # Reduced L2
    )
    
    # Load data
    X, y, feature_names, target_names = model.load_and_prepare_data()
    
    if X is None:
        return
    
    # Preprocess
    X, y = model.preprocess_data(X, y)
    
    # Split data
    print("\n🔄 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples") 
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    print("\n🎯 Training optimized model...")
    trained_model = model.train_model(X_train, X_val, y_train, y_val, feature_names, target_names)
    
    # Evaluate
    print("\n📊 Final evaluation...")
    metrics, y_true, y_pred = model.evaluate_model(X_test, y_test, target_names)
    
    # Results
    print(f"\n{'='*60}")
    print("🎯 OPTIMIZED RESULTS")
    print(f"{'='*60}")
    
    for target_name, target_metrics in metrics.items():
        if target_name != 'overall':
            print(f"\n{target_name}:")
            print(f"  RMSE: {target_metrics['RMSE']:.4f}")
            print(f"  MAE:  {target_metrics['MAE']:.4f}")
            print(f"  R²:   {target_metrics['R²']:.4f}")
            print(f"  Accuracy (±10%): {target_metrics['Accuracy_10%']*100:.1f}%")
            print(f"  Accuracy (±20%): {target_metrics['Accuracy_20%']*100:.1f}%")
            print(f"  MAPE: {target_metrics['MAPE']:.2f}%")
    
    print(f"\nOverall:")
    print(f"  MAE: {metrics['overall']['MAE']:.4f}")
    print(f"  RMSE: {metrics['overall']['RMSE']:.4f}")
    
    # Check if target achieved
    if 'NO2' in metrics:
        no2_acc = metrics['NO2']['Accuracy_10%'] * 100
        baseline = 42.3
        target = 48.0
        
        print(f"\n🎯 TARGET ASSESSMENT:")
        print(f"  Baseline NO2 accuracy: {baseline:.1f}%")
        print(f"  Current NO2 accuracy:  {no2_acc:.1f}%")
        print(f"  Target NO2 accuracy:   {target:.1f}%")
        
        if no2_acc >= target:
            print(f"  ✅ TARGET ACHIEVED! (+{no2_acc-baseline:.1f} pp improvement)")
        else:
            improvement = no2_acc - baseline
            needed = target - no2_acc
            print(f"  📈 Progress: +{improvement:.1f} pp (need +{needed:.1f} more)")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
