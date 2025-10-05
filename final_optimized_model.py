#!/usr/bin/env python3
"""
Final Optimized Air Quality Model - Working with Available Clean Data
Designed to maximize performance with the 221 complete NO2/O3 records
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("🚀 FINAL OPTIMIZED AIR QUALITY MODEL")
print("=" * 50)

# Load and prepare data
print("📊 Loading data...")
df = pd.read_csv('tcn_models/real_air_quality_time_series.csv')

# Get complete records only
complete_data = df.dropna(subset=['NO2', 'O3']).copy()

# Merge synthetic satellite features (daily, by location)
try:
    sat = pd.read_csv('satellite_data/satellite_features.csv')
    # Normalize timestamps to daily for join
    if 'timestamp' in df.columns:
        complete_data['date'] = pd.to_datetime(complete_data['timestamp']).dt.floor('D')
    else:
        for c in ['time','date','datetime']:
            if c in complete_data.columns:
                complete_data['date'] = pd.to_datetime(complete_data[c]).dt.floor('D')
                break
    sat['timestamp'] = pd.to_datetime(sat['timestamp'])
    sat['date'] = sat['timestamp'].dt.floor('D')
    # Keep relevant sat columns
    keep_cols = ['location_id','date','no2_column','o3_column','aod','cloud_fraction','sza','qa_value']
    sat = sat[keep_cols]
    before = len(complete_data)
    complete_data = complete_data.merge(sat, on=['location_id','date'], how='left')
    print(f"🛰️  Merged satellite features: {before} -> {len(complete_data)} rows; sat cols added")
except Exception as e:
    print(f"[WARN] Satellite features not merged: {e}")
print(f"✅ Found {len(complete_data)} complete records")

# Enhanced feature engineering for limited data
def create_enhanced_features(data):
    """Create comprehensive features from available data"""
    features_df = data.copy()
    
    # Time-based features
    if 'timestamp' in features_df.columns:
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['month'] = features_df['timestamp'].dt.month
        features_df['season'] = (features_df['month'] % 12 + 3) // 3
        
        # Cyclical encoding
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['dow_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['dow_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
    
    # Location-based features
    if 'location_id' in features_df.columns:
        # Location frequency encoding
        location_counts = features_df['location_id'].value_counts()
        features_df['location_frequency'] = features_df['location_id'].map(location_counts)
        
        # Location target encoding (for each pollutant)
        for target in ['NO2', 'O3']:
            if target in features_df.columns:
                location_target_mean = features_df.groupby('location_id')[target].mean()
                features_df[f'location_{target}_mean'] = features_df['location_id'].map(location_target_mean)
    
    # Weather-based features (if available)
    weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']
    available_weather = [col for col in weather_cols if col in features_df.columns and features_df[col].notna().sum() > 0]
    
    for col in available_weather:
        # Fill missing values with median
        features_df[col] = features_df[col].fillna(features_df[col].median())
        
        # Create interaction features
        if len(available_weather) > 1:
            for other_col in available_weather:
                if col != other_col:
                    features_df[f'{col}_{other_col}_ratio'] = features_df[col] / (features_df[other_col] + 1e-8)
    
    # Pollutant interaction features (if CO, SO2 available)
    other_pollutants = ['CO', 'SO2', 'PM10', 'PM2_5']
    available_pollutants = [col for col in other_pollutants if col in features_df.columns and features_df[col].notna().sum() > 10]
    
    for pol in available_pollutants:
        # Fill missing values
        features_df[pol] = features_df[pol].fillna(features_df[pol].median())
        
        # Create ratios and interactions
        if 'NO2' in features_df.columns:
            features_df[f'NO2_{pol}_ratio'] = features_df['NO2'] / (features_df[pol] + 1e-8)
        if 'O3' in features_df.columns:
            features_df[f'O3_{pol}_ratio'] = features_df['O3'] / (features_df[pol] + 1e-8)
    # Satellite features and QA interactions
    sat_cols = ['no2_column','o3_column','aod','cloud_fraction','sza','qa_value']
    for c in sat_cols:
        if c in features_df.columns:
            features_df[c] = features_df[c].fillna(features_df[c].median())
    if 'qa_value' in features_df.columns:
        for c in ['no2_column','o3_column','aod','cloud_fraction']:
            if c in features_df.columns:
                features_df[f'{c}_qa'] = features_df[c] * features_df['qa_value']

    return features_df

# Create enhanced features
print("🔧 Engineering features...")
enhanced_data = create_enhanced_features(complete_data)

# Prepare feature matrix
feature_cols = [col for col in enhanced_data.columns if col not in ['NO2', 'O3', 'timestamp']]
X = enhanced_data[feature_cols].select_dtypes(include=[np.number]).fillna(0)
y_no2 = enhanced_data['NO2']
y_o3 = enhanced_data['O3']

print(f"📈 Features created: {X.shape[1]} features")
print(f"   Sample size: {X.shape[0]} records")

class EnhancedEnsembleModel:
    """Advanced ensemble model combining multiple algorithms"""
    
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.feature_names = None
        
    def _create_models(self, target_name):
        """Create diverse set of models for ensemble"""
        models = {}
        
        # Random Forest with optimized parameters
        models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        
        # Gradient Boosting
        models['gbm'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        
        # Ridge Regression
        models['ridge'] = Ridge(alpha=1.0)
        
        # Neural Network
        def create_nn():
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_dim=X.shape[1]),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        
        models['nn'] = create_nn
        
        return models
    
    def fit(self, X_train, y_train, target_name):
        """Train ensemble of models"""
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        self.scalers[target_name] = RobustScaler()
        X_scaled = self.scalers[target_name].fit_transform(X_train)
        
        # Create and train models
        self.models[target_name] = self._create_models(target_name)
        
        for name, model in self.models[target_name].items():
            print(f"   Training {name} for {target_name}...")
            
            if name == 'nn':
                # Train neural network
                nn_model = model()
                nn_model.fit(X_scaled, y_train, epochs=100, batch_size=16, verbose=0)
                self.models[target_name][name] = nn_model
            else:
                # Train sklearn models
                if name in ['rf', 'gbm']:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_scaled, y_train)
    
    def predict(self, X_test, target_name):
        """Ensemble prediction with weighted averaging"""
        if target_name not in self.models:
            raise ValueError(f"No trained models for {target_name}")
        
        X_scaled = self.scalers[target_name].transform(X_test)
        predictions = []
        weights = {'rf': 0.3, 'gbm': 0.3, 'ridge': 0.2, 'nn': 0.2}
        
        weighted_pred = np.zeros(len(X_test))
        
        for name, model in self.models[target_name].items():
            if name == 'nn':
                pred = model.predict(X_scaled).flatten()
            elif name in ['rf', 'gbm']:
                pred = model.predict(X_test)
            else:
                pred = model.predict(X_scaled)
            
            weighted_pred += weights[name] * pred
        
        return weighted_pred

# Cross-validation training
print("\n🎯 Training enhanced ensemble models...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {'NO2': [], 'O3': []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/5:")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_no2_train, y_no2_val = y_no2.iloc[train_idx], y_no2.iloc[val_idx]
    y_o3_train, y_o3_val = y_o3.iloc[train_idx], y_o3.iloc[val_idx]
    
    # Train models for each pollutant
    for target_name, y_train, y_val in [('NO2', y_no2_train, y_no2_val), ('O3', y_o3_train, y_o3_val)]:
        model = EnhancedEnsembleModel()
        model.fit(X_train, y_train, target_name)
        
        # Validate
        y_pred = model.predict(X_val, target_name)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Calculate MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_val - y_pred) / np.where(np.abs(y_val) < 1e-8, 1, y_val))) * 100
        accuracy = max(0, 100 - mape)
        
        results[target_name].append({
            'fold': fold + 1,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy
        })
        
        print(f"  {target_name}: R² = {r2:.3f}, Accuracy = {accuracy:.1f}%")

# Final results
print("\n" + "=" * 50)
print("📊 FINAL PERFORMANCE RESULTS")
print("=" * 50)

for target in ['NO2', 'O3']:
    accuracies = [r['accuracy'] for r in results[target]]
    r2_scores = [r['r2'] for r in results[target]]
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_r2 = np.mean(r2_scores)
    
    print(f"\n🎯 {target} Performance:")
    print(f"   Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"   R² Score: {mean_r2:.3f}")
    
    if target == 'NO2':
        baseline = 42.3
        improvement = mean_acc - baseline
        print(f"   📈 Improvement over baseline: {improvement:+.1f}%")
        if improvement > 0:
            print(f"   ✅ SUCCESS: {improvement:.1f}% accuracy gain!")
        else:
            print(f"   ⚠️  No improvement over baseline")

# Train final production model
print(f"\n🏭 Training final production model on all {len(X)} records...")
final_model_no2 = EnhancedEnsembleModel()
final_model_o3 = EnhancedEnsembleModel()

final_model_no2.fit(X, y_no2, 'NO2')
final_model_o3.fit(X, y_o3, 'O3')

print("✅ Production models trained and ready!")
print(f"📁 Models use {X.shape[1]} engineered features")
print(f"📊 Trained on {len(X)} complete records")

# Feature importance analysis
if 'rf' in final_model_no2.models.get('NO2', {}):
    print(f"\n🔍 Top 10 Most Important Features for NO2:")
    rf_model = final_model_no2.models['NO2']['rf']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")

print(f"\n🎉 OPTIMIZATION COMPLETE!")
print(f"   Enhanced ensemble model with {X.shape[1]} features")
print(f"   Cross-validated on {len(X)} complete records")
print(f"   Ready for production deployment!")
