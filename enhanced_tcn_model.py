#!/usr/bin/env python3
"""
Quick Performance Improvements for TCN Air Quality Model

Implements the most effective quick-win optimizations to boost model performance:
1. Multi-task learning with balanced loss
2. Channel attention mechanism  
3. Robust scaling for outlier handling
4. Enhanced data augmentation

Target: Increase NO2 accuracy from 42.3% to 48-52% (±10%)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# Import our base model
import sys
sys.path.append('/Users/a91788/Desktop/NASA/Nasa-Space-Apps')
from ensemble_air_quality_model import TCNAirQualityModel

class EnhancedTCNAirQualityModel(TCNAirQualityModel):
    """
    Enhanced TCN with quick-win performance improvements
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_robust_scaling = True
        self.use_channel_attention = True
        self.use_multi_task_loss = True
        self.use_data_augmentation = True
        
    def preprocess_data_enhanced(self, df, target_pollutants=['NO2', 'O3']):
        """
        Enhanced data preprocessing with outlier handling and robust scaling
        """
        print("🔧 Enhanced data preprocessing...")
        
        # Prepare sequences using base method
        X, y, feature_names, target_names = self.prepare_sequences(df, target_pollutants)
        
        if len(X) == 0:
            return X, y, feature_names, target_names
        
        # Outlier detection and handling
        print("  • Detecting and handling outliers...")
        outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Reshape for outlier detection
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(n_samples, -1)  # Flatten sequences
        
        # Detect outliers
        outlier_mask = outlier_detector.fit_predict(X_reshaped) == 1
        
        # Keep only non-outlier samples
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        print(f"  • Removed {np.sum(~outlier_mask)} outlier sequences ({100*np.sum(~outlier_mask)/len(outlier_mask):.1f}%)")
        print(f"  • Kept {len(X_clean)} clean sequences")
        
        return X_clean, y_clean, feature_names, target_names
    
    def create_channel_attention(self, inputs, reduction_ratio=16, name='channel_attention'):
        """
        Squeeze-and-Excitation channel attention mechanism
        """
        channels = inputs.shape[-1]
        
        # Global average pooling
        gap = layers.GlobalAveragePooling1D(name=f'{name}_gap')(inputs)
        
        # Squeeze: dimensionality reduction
        squeeze = layers.Dense(
            max(channels // reduction_ratio, 4),  # Ensure minimum 4 neurons
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name=f'{name}_squeeze'
        )(gap)
        
        # Excitation: dimensionality expansion with sigmoid
        excitation = layers.Dense(
            channels,
            activation='sigmoid',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name=f'{name}_excitation'
        )(squeeze)
        
        # Reshape and scale original input
        excitation = layers.Reshape((1, channels), name=f'{name}_reshape')(excitation)
        scaled = layers.Multiply(name=f'{name}_scale')([inputs, excitation])
        
        return scaled
    
    def create_enhanced_tcn_block(self, inputs, filters, kernel_size, dilation_rate, dropout_rate, name):
        """
        Enhanced TCN block with channel attention
        """
        # Standard TCN block
        x = self.create_tcn_block(inputs, filters, kernel_size, dilation_rate, dropout_rate, name)
        
        # Add channel attention if enabled
        if self.use_channel_attention:
            x = self.create_channel_attention(x, name=f'{name}_attention')
        
        return x
    
    def build_enhanced_tcn_model(self, n_features, n_targets):
        """
        Build enhanced TCN with attention and multi-task learning
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, n_features), name='input')
        
        # Initial convolution with better initialization
        x = layers.Conv1D(
            filters=self.n_filters,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='initial_conv'
        )(inputs)
        
        # Enhanced TCN blocks
        for i in range(self.n_layers):
            dilation_rate = self.dilation_rates[i % len(self.dilation_rates)]
            x = self.create_enhanced_tcn_block(
                inputs=x,
                filters=self.n_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate,
                name=f'enhanced_tcn_block_{i}'
            )
        
        # Multiple feature extraction methods
        global_max = layers.GlobalMaxPooling1D(name='global_max')(x)
        global_avg = layers.GlobalAvgPooling1D(name='global_avg')(x)
        last_timestep = layers.Lambda(lambda t: t[:, -1, :], name='last_timestep')(x)
        
        # Combine features
        combined = layers.Concatenate(name='feature_combination')([
            global_max, global_avg, last_timestep
        ])
        
        # Shared dense layers
        shared = layers.Dense(128, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                             name='shared_dense1')(combined)
        shared = layers.BatchNormalization(name='shared_bn1')(shared)
        shared = layers.Dropout(self.dropout_rate, name='shared_dropout1')(shared)
        
        shared = layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                             name='shared_dense2')(shared)
        shared = layers.BatchNormalization(name='shared_bn2')(shared)
        shared = layers.Dropout(self.dropout_rate/2, name='shared_dropout2')(shared)
        
        if self.use_multi_task_loss and n_targets == 2:
            # Separate heads for NO2 and O3
            no2_head = layers.Dense(32, activation='relu', name='no2_dense')(shared)
            no2_head = layers.BatchNormalization(name='no2_bn')(no2_head)
            no2_head = layers.Dropout(0.2, name='no2_dropout')(no2_head)
            no2_output = layers.Dense(1, activation='linear', name='no2_output')(no2_head)
            
            o3_head = layers.Dense(32, activation='relu', name='o3_dense')(shared)
            o3_head = layers.BatchNormalization(name='o3_bn')(o3_head)
            o3_head = layers.Dropout(0.2, name='o3_dropout')(o3_head)
            o3_output = layers.Dense(1, activation='linear', name='o3_output')(o3_head)
            
            # Combine outputs
            outputs = layers.Concatenate(name='final_output')([no2_output, o3_output])
        else:
            # Single output head
            outputs = layers.Dense(n_targets, activation='linear',
                                 kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                                 name='output')(shared)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='Enhanced_TCN_AirQuality')
        
        # Enhanced loss function
        if self.use_multi_task_loss and n_targets == 2:
            def balanced_multi_task_loss(y_true, y_pred):
                """Balanced loss giving more weight to harder NO2 predictions"""
                no2_true, o3_true = y_true[:, 0:1], y_true[:, 1:2]
                no2_pred, o3_pred = y_pred[:, 0:1], y_pred[:, 1:2]
                
                # Use Huber loss for robustness
                huber = tf.keras.losses.Huber(delta=1.0)
                
                no2_loss = huber(no2_true, no2_pred)
                o3_loss = huber(o3_true, o3_pred)
                
                # Adaptive weights (more weight to harder NO2 task)
                no2_weight = 1.8  # Higher weight for NO2
                o3_weight = 1.0
                
                # Optional: add correlation penalty to encourage coherent predictions
                correlation_penalty = 0.1 * tf.reduce_mean(tf.square(
                    tf.nn.l2_normalize(no2_pred, axis=0) - 
                    tf.nn.l2_normalize(o3_pred, axis=0)
                ))
                
                return no2_weight * no2_loss + o3_weight * o3_loss + correlation_penalty
            
            loss = balanced_multi_task_loss
        else:
            loss = tf.keras.losses.Huber(delta=1.0)
        
        # Enhanced optimizer with gradient clipping
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def augment_data(self, X, y):
        """
        Data augmentation for time series
        """
        if not self.use_data_augmentation:
            return X, y
        
        print("📈 Applying data augmentation...")
        
        augmented_X = []
        augmented_y = []
        
        # Original data
        augmented_X.append(X)
        augmented_y.append(y)
        
        # Gaussian noise augmentation
        noise_factor = 0.02
        X_noise = X + np.random.normal(0, noise_factor, X.shape)
        augmented_X.append(X_noise)
        augmented_y.append(y)
        
        # Time shifting augmentation (small shifts)
        shift_range = 2
        for shift in [-shift_range, shift_range]:
            if shift != 0:
                X_shifted = np.roll(X, shift, axis=1)
                # Mask the shifted regions
                if shift > 0:
                    X_shifted[:, :shift, :] = X[:, :shift, :]  # Keep original start
                else:
                    X_shifted[:, shift:, :] = X[:, shift:, :]  # Keep original end
                
                augmented_X.append(X_shifted)
                augmented_y.append(y)
        
        # Combine all augmented data
        X_aug = np.concatenate(augmented_X, axis=0)
        y_aug = np.concatenate(augmented_y, axis=0)
        
        print(f"  • Original data: {len(X)} samples")
        print(f"  • Augmented data: {len(X_aug)} samples")
        print(f"  • Augmentation factor: {len(X_aug)/len(X):.1f}x")
        
        return X_aug, y_aug
    
    def train_enhanced_tcn_model(self, X_train, X_val, y_train, y_val, feature_names, target_names):
        """
        Train enhanced TCN model with all optimizations
        """
        print("🚀 Training Enhanced TCN with Performance Optimizations...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        # Data augmentation
        X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
        
        # Enhanced scaling
        if self.use_robust_scaling:
            print("  • Using RobustScaler for outlier-resistant normalization")
            scaler = RobustScaler()
        else:
            print("  • Using StandardScaler")
            scaler = StandardScaler()
        
        # Fit scaler and transform data
        n_samples_train, n_timesteps, n_features = X_train_aug.shape
        X_train_reshaped = X_train_aug.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_val_scaled = scaler.transform(X_val_reshaped)
        
        X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
        X_val_scaled = X_val_scaled.reshape(-1, n_timesteps, n_features)
        
        # Store scaler
        self.feature_scaler = scaler
        
        # Target scaling for balanced multi-output
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train_aug)
        y_val_scaled = target_scaler.transform(y_val)
        self.target_scaler = target_scaler
        
        # Build enhanced model
        model = self.build_enhanced_tcn_model(n_features=n_features, n_targets=len(target_names))
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # Increased patience
                restore_best_weights=True,
                min_delta=0.0001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f'{self.model_dir}/enhanced_tcn_best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            # Learning rate warmup
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * min(1.0, (epoch + 1) / 5),  # 5-epoch warmup
                verbose=0
            )
        ]
        
        # Train model with optimizations
        print("🎯 Starting enhanced TCN training...")
        print("  Optimizations enabled:")
        print(f"    ✅ Channel attention: {self.use_channel_attention}")
        print(f"    ✅ Multi-task learning: {self.use_multi_task_loss}")
        print(f"    ✅ Robust scaling: {self.use_robust_scaling}")
        print(f"    ✅ Data augmentation: {self.use_data_augmentation}")
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100,  # Increased epochs for better convergence
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store trained model
        self.tcn_model = model
        
        print("✅ Enhanced TCN training complete!")
        return model

def main():
    """Main function for enhanced air quality modeling"""
    
    # Change to NASA Space Apps directory
    repo_dir = "/Users/a91788/Desktop/NASA/Nasa-Space-Apps"
    os.chdir(repo_dir)
    
    print("🚀 ENHANCED TCN AIR QUALITY MODEL")
    print("="*60)
    print("Performance Optimization Features:")
    print("  ✅ Channel attention mechanism")
    print("  ✅ Multi-task learning with balanced loss")
    print("  ✅ Robust scaling for outlier handling") 
    print("  ✅ Advanced data augmentation")
    print("  ✅ Enhanced optimizer with gradient clipping")
    print("="*60)
    
    # Initialize enhanced model
    model = EnhancedTCNAirQualityModel(
        sequence_length=24,
        n_filters=40,          # Slightly increased filters
        kernel_size=3,
        n_layers=4,
        dropout_rate=0.3,
        l2_reg=0.01,
        dilation_rates=[1, 2, 4, 8]
    )
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    df = model.load_real_data()
    
    if df is None or df.empty:
        print("❌ Error: Could not load real data!")
        return
    
    # Enhanced preprocessing
    target_pollutants = ['NO2', 'O3']
    X, y, feature_names, target_names = model.preprocess_data_enhanced(df, target_pollutants)
    
    if len(X) == 0:
        print("❌ Error: No valid sequences created!")
        return
    
    # Split data
    print("\n🔄 Splitting data...")
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
    enhanced_model = model.train_enhanced_tcn_model(
        X_train, X_val, y_train, y_val, feature_names, target_names
    )
    
    # Evaluate model
    print("\n📈 Evaluating enhanced model...")
    metrics, y_true, y_pred = model.evaluate_tcn_model(X_test, y_test, target_names)
    
    print(f"\n{'='*60}")
    print("🏆 ENHANCED TCN RESULTS")
    print(f"{'='*60}")
    
    print("Performance Improvements:")
    for target_name, target_metrics in metrics['per_target'].items():
        print(f"\n{target_name}:")
        print(f"  RMSE: {target_metrics['RMSE']:.4f}")
        print(f"  MAE:  {target_metrics['MAE']:.4f}")
        print(f"  R²:   {target_metrics['R²']:.4f}")
        print(f"  Accuracy (±10%): {target_metrics['Accuracy_10%']:.3f} ({target_metrics['Accuracy_10%']*100:.1f}%)")
        print(f"  Accuracy (±20%): {target_metrics['Accuracy_20%']:.3f} ({target_metrics['Accuracy_20%']*100:.1f}%)")
        print(f"  MAPE: {target_metrics['MAPE']:.2f}%")
    
    print(f"\nOverall Performance:")
    print(f"  RMSE: {metrics['overall']['RMSE']:.4f}")
    print(f"  MAE:  {metrics['overall']['MAE']:.4f}")
    
    # Save enhanced model
    try:
        model.tcn_model.save('enhanced_tcn_air_quality_model.keras')
        print(f"\n💾 Enhanced model saved as 'enhanced_tcn_air_quality_model.keras'")
    except Exception as e:
        print(f"\n⚠️  Could not save model: {e}")
    
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    if len(metrics['per_target']) >= 2:
        no2_acc = list(metrics['per_target'].values())[0]['Accuracy_10%'] * 100
        target_acc = 48
        if no2_acc >= target_acc:
            print(f"  ✅ NO2 accuracy target MET: {no2_acc:.1f}% ≥ {target_acc}%")
        else:
            print(f"  ⚠️  NO2 accuracy: {no2_acc:.1f}% (target: {target_acc}%+)")
        
        improvement = no2_acc - 42.3
        print(f"  📊 Improvement over baseline: +{improvement:.1f} percentage points")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
