#!/usr/bin/env python3
"""
High-Epoch Enhanced Model Training

Train the enhanced TCN model with maximum epochs for optimal performance.
Uses the most advanced optimizations with extended training.
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

class HighEpochEnhancedTrainer:
    """
    High-epoch trainer for enhanced TCN model
    """
    
    def __init__(self):
        print("🚀 HIGH-EPOCH ENHANCED MODEL TRAINER")
        print("="*60)
        print("  Strategy: Maximum epoch training with advanced optimizations")
        print("  Target: Achieve highest possible NO2 accuracy")
    
    def load_and_filter_data(self):
        """Load and filter data for training"""
        print("\n📊 Loading and filtering data...")
        
        data_file = 'tcn_models/real_air_quality_time_series.csv'
        if not os.path.exists(data_file):
            print("❌ Data file not found!")
            return None
        
        df = pd.read_csv(data_file)
        print(f"  Loaded {len(df)} records from {df['location_id'].nunique()} locations")
        
        # Select best locations with most complete data
        location_quality = []
        for location_id, group in df.groupby('location_id'):
            # Check data completeness
            no2_count = group['NO2'].count()
            o3_count = group['O3'].count()
            total_count = len(group)
            
            if total_count >= 15 and no2_count >= 3 and o3_count >= 3:
                completeness_score = (no2_count + o3_count) / (2 * total_count)
                location_quality.append({
                    'location_id': location_id,
                    'total': total_count,
                    'no2_count': no2_count,
                    'o3_count': o3_count,
                    'score': completeness_score
                })
        
        # Sort by quality and take top locations
        location_quality.sort(key=lambda x: x['score'], reverse=True)
        selected_locations = [loc['location_id'] for loc in location_quality[:300]]
        
        df_filtered = df[df['location_id'].isin(selected_locations)].copy()
        
        print(f"  • Selected {len(selected_locations)} high-quality locations")
        print(f"  • Filtered dataset: {len(df_filtered)} records")
        
        return df_filtered
    
    def create_enhanced_sequences(self, df):
        """Create sequences with enhanced preprocessing"""
        print("\n🔄 Creating enhanced sequences...")
        
        # Enhanced feature selection
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
        
        # Process each location with enhanced data handling
        for location_id, site_data in df.groupby('location_id'):
            if len(site_data) < sequence_length + 5:
                continue
            
            # Sort by datetime if available
            site_data = site_data.reset_index(drop=True)
            
            # Enhanced data cleaning - keep only rows with target data
            site_clean = site_data.dropna(subset=['NO2', 'O3']).copy()
            
            if len(site_clean) < sequence_length + 3:
                continue
            
            # Fill feature NaNs with advanced imputation
            for col in feature_cols:
                if col in site_clean.columns:
                    # Forward fill first
                    site_clean[col] = site_clean[col].fillna(method='ffill')
                    # Backward fill
                    site_clean[col] = site_clean[col].fillna(method='bfill')
                    # Final fill with median
                    if site_clean[col].isnull().any():
                        median_val = site_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        site_clean[col].fillna(median_val, inplace=True)
            
            # Create sequences from clean data
            for i in range(len(site_clean) - sequence_length):
                X_seq = site_clean.iloc[i:i + sequence_length][feature_cols].values
                
                target_idx = i + sequence_length
                if target_idx < len(site_clean):
                    no2_target = site_clean.iloc[target_idx]['NO2']
                    o3_target = site_clean.iloc[target_idx]['O3']
                    
                    # Quality validation
                    if (not np.any(np.isnan(X_seq)) and 
                        not np.any(np.isinf(X_seq)) and
                        not pd.isna(no2_target) and not pd.isna(o3_target) and
                        not np.isinf(no2_target) and not np.isinf(o3_target)):
                        sequences.append(X_seq)
                        targets.append([no2_target, o3_target])
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"  Created {len(X)} enhanced sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, feature_cols
    
    def build_ultimate_enhanced_model(self, n_features):
        """Build the ultimate enhanced model with all optimizations"""
        print("\n🏗️ Building ultimate enhanced model...")
        
        inputs = keras.layers.Input(shape=(24, n_features), name='input')
        
        # Enhanced TCN architecture with maximum optimizations
        n_filters = 48  # Increased capacity
        dilation_rates = [1, 2, 4, 8, 16]  # Extended dilation
        
        # Initial projection with batch normalization
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            name='initial_conv'
        )(inputs)
        x = keras.layers.BatchNormalization(name='initial_bn')(x)
        
        # Enhanced TCN blocks with all optimizations
        for i, dilation_rate in enumerate(dilation_rates):
            # First dilated convolution
            conv1 = keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.005),
                name=f'conv1_{i}'
            )(x)
            
            conv1 = keras.layers.BatchNormalization(name=f'bn1_{i}')(conv1)
            conv1 = keras.layers.Dropout(0.15, name=f'drop1_{i}')(conv1)
            
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
            
            # Enhanced residual connection
            if x.shape[-1] != n_filters:
                residual = keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=1,
                    kernel_regularizer=keras.regularizers.l2(0.005),
                    name=f'residual_{i}'
                )(x)
                residual = keras.layers.BatchNormalization(name=f'residual_bn_{i}')(residual)
            else:
                residual = x
            
            x = keras.layers.Add(name=f'add_{i}')([conv2, residual])
            x = keras.layers.Activation('relu', name=f'relu_{i}')(x)
        
        # Multi-head attention mechanism
        print("  Adding multi-head attention...")
        attention = keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=n_filters // 4,
            name='multi_head_attention'
        )(x, x)
        
        attention = keras.layers.BatchNormalization(name='attention_bn')(attention)
        attention = keras.layers.Dropout(0.1, name='attention_dropout')(attention)
        
        # Combine attention with original features
        x = keras.layers.Add(name='attention_add')([x, attention])
        
        # Enhanced feature extraction with multiple strategies
        global_avg = keras.layers.GlobalAveragePooling1D(name='global_avg')(x)
        global_max = keras.layers.GlobalMaxPooling1D(name='global_max')(x)
        
        # Last timestep features
        last_timestep = keras.layers.Lambda(lambda t: t[:, -1, :], name='last_timestep')(x)
        
        # Combine all feature extraction methods
        combined_features = keras.layers.Concatenate(name='combined_features')([
            global_avg, global_max, last_timestep
        ])
        
        # Enhanced dense layers with maximum capacity
        dense1 = keras.layers.Dense(
            128,  # Increased capacity
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.005),
            name='dense1'
        )(combined_features)
        dense1 = keras.layers.BatchNormalization(name='dense1_bn')(dense1)
        dense1 = keras.layers.Dropout(0.3, name='dense1_dropout')(dense1)
        
        dense2 = keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.005),
            name='dense2'
        )(dense1)
        dense2 = keras.layers.BatchNormalization(name='dense2_bn')(dense2)
        dense2 = keras.layers.Dropout(0.25, name='dense2_dropout')(dense2)
        
        # Enhanced multi-task heads
        # NO2 head (specialized for challenging pollutant)
        no2_dense1 = keras.layers.Dense(48, activation='relu', name='no2_dense1')(dense2)
        no2_dense1 = keras.layers.BatchNormalization(name='no2_bn1')(no2_dense1)
        no2_dense1 = keras.layers.Dropout(0.2, name='no2_dropout1')(no2_dense1)
        
        no2_dense2 = keras.layers.Dense(24, activation='relu', name='no2_dense2')(no2_dense1)
        no2_dense2 = keras.layers.Dropout(0.15, name='no2_dropout2')(no2_dense2)
        no2_output = keras.layers.Dense(1, activation='linear', name='no2_output')(no2_dense2)
        
        # O3 head (optimized for easier pollutant)
        o3_dense1 = keras.layers.Dense(32, activation='relu', name='o3_dense1')(dense2)
        o3_dense1 = keras.layers.BatchNormalization(name='o3_bn1')(o3_dense1)
        o3_dense1 = keras.layers.Dropout(0.15, name='o3_dropout1')(o3_dense1)
        
        o3_dense2 = keras.layers.Dense(16, activation='relu', name='o3_dense2')(o3_dense1)
        o3_dense2 = keras.layers.Dropout(0.1, name='o3_dropout2')(o3_dense2)
        o3_output = keras.layers.Dense(1, activation='linear', name='o3_output')(o3_dense2)
        
        # Combine outputs
        outputs = keras.layers.Concatenate(name='final_outputs')([no2_output, o3_output])
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='UltimateEnhancedTCN')
        
        # Ultimate enhanced loss function
        def ultimate_weighted_loss(y_true, y_pred):
            no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
            no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
            
            # Use combination of losses for robustness
            huber = keras.losses.Huber(delta=1.0)
            mse = keras.losses.MeanSquaredError()
            
            # NO2 loss (challenging pollutant - use robust Huber)
            no2_loss = huber(no2_true, no2_pred)
            
            # O3 loss (easier pollutant - use standard MSE)  
            o3_loss = mse(o3_true, o3_pred)
            
            # Enhanced weighting for NO2 improvement
            return 3.0 * no2_loss + 1.0 * o3_loss
        
        # Ultimate optimizer configuration
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.0006,  # Careful learning rate for stability
                weight_decay=0.008,    # Enhanced regularization
                clipnorm=0.8          # Gradient clipping
            ),
            loss=ultimate_weighted_loss,
            metrics=['mae', 'mse']
        )
        
        print(f"  Ultimate model created: {model.count_params():,} parameters")
        return model
    
    def train_with_maximum_epochs(self, X, y, feature_names):
        """Train with maximum epochs for best performance"""
        print("\n🎯 Training with maximum epochs...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"  Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
        
        # Enhanced preprocessing with RobustScaler
        print("  • Using RobustScaler for ultimate preprocessing")
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        # Scale features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
        X_val_scaled = feature_scaler.transform(X_val_reshaped).reshape(-1, n_timesteps, n_features)
        X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(-1, n_timesteps, n_features)
        
        # Scale targets
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_val_scaled = target_scaler.transform(y_val)
        
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        
        # Build ultimate model
        model = self.build_ultimate_enhanced_model(n_features)
        
        # Maximum epoch training configuration
        max_epochs = 200  # Maximum epochs for best performance
        
        # Ultimate callbacks for maximum performance
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,  # Extended patience for maximum training
                restore_best_weights=True,
                min_delta=0.0001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=15,  # Extended patience
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='ultimate_enhanced_tcn_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Learning rate warmup
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.0006 * min(1.0, (epoch + 1) / 10),
                verbose=0
            )
        ]
        
        # Maximum epoch training
        print(f"  🚀 Starting MAXIMUM EPOCH training ({max_epochs} epochs)...")
        print(f"  This will take longer but achieve the best possible performance!")
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=max_epochs,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        self.history = history
        
        # Final evaluation
        print(f"\n📊 Final evaluation after {max_epochs} epochs...")
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate final metrics
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
            
            # Enhanced accuracy calculation
            abs_errors = np.abs(y_true_i - y_pred_i)
            abs_true = np.abs(y_true_i)
            threshold_base = np.maximum(abs_true, np.percentile(abs_true, 5))
            
            acc_5 = np.mean(abs_errors <= 0.05 * threshold_base)
            acc_10 = np.mean(abs_errors <= 0.10 * threshold_base)
            acc_15 = np.mean(abs_errors <= 0.15 * threshold_base)
            acc_20 = np.mean(abs_errors <= 0.20 * threshold_base)
            
            results[pollutant] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Accuracy_5%': acc_5,
                'Accuracy_10%': acc_10,
                'Accuracy_15%': acc_15,
                'Accuracy_20%': acc_20
            }
        
        return results

def main():
    """Main execution"""
    
    print("🚀 ULTIMATE HIGH-EPOCH ENHANCED MODEL TRAINING")
    print("="*70)
    print("Training the most advanced model with maximum epochs")
    print("="*70)
    
    # Initialize trainer
    trainer = HighEpochEnhancedTrainer()
    
    # Load and prepare data
    df = trainer.load_and_filter_data()
    if df is None or len(df) == 0:
        print("❌ Could not load sufficient data!")
        return
    
    # Create enhanced sequences
    X, y, feature_names = trainer.create_enhanced_sequences(df)
    
    if len(X) == 0:
        print("❌ No sequences created!")
        return
    
    # Train with maximum epochs
    results = trainer.train_with_maximum_epochs(X, y, feature_names)
    
    # Display ultimate results
    print(f"\n{'='*70}")
    print("🏆 ULTIMATE ENHANCED MODEL RESULTS")
    print(f"{'='*70}")
    
    for pollutant, metrics in results.items():
        print(f"\n{pollutant} Performance:")
        for metric, value in metrics.items():
            if 'Accuracy' in metric:
                print(f"  {metric}: {value*100:.1f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Performance assessment
    baseline_no2 = 42.3
    enhanced_no2 = results['NO2']['Accuracy_10%'] * 100
    target_no2 = 50.0
    
    print(f"\n🎯 ULTIMATE PERFORMANCE ASSESSMENT:")
    print(f"  Baseline NO2 (10%±): {baseline_no2:.1f}%")
    print(f"  Ultimate NO2 (10%±): {enhanced_no2:.1f}%")
    print(f"  Stretch Target: {target_no2:.1f}%")
    
    improvement = enhanced_no2 - baseline_no2
    if enhanced_no2 >= target_no2:
        print(f"  🎉 STRETCH TARGET ACHIEVED! (+{improvement:.1f}pp)")
    elif enhanced_no2 >= 45.0:
        print(f"  ✅ EXCELLENT PERFORMANCE! (+{improvement:.1f}pp)")
    elif improvement > 0:
        print(f"  📈 Solid Improvement: +{improvement:.1f}pp")
    else:
        print(f"  📊 Results: {improvement:.1f}pp change")
    
    enhanced_o3 = results['O3']['Accuracy_10%'] * 100
    print(f"  Ultimate O3 (10%±): {enhanced_o3:.1f}%")
    
    print(f"\n{'='*70}")
    print("🎯 ULTIMATE TRAINING COMPLETE!")
    print("Model saved as: ultimate_enhanced_tcn_best.keras")

if __name__ == "__main__":
    main()
