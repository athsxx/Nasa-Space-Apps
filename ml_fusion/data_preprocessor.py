"""
Data Preprocessor for TEMPO satellite and OpenAQ ground station data.

This module handles data cleaning, normalization, and preparation for machine learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocesses TEMPO satellite and OpenAQ ground station data for ML training."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.tempo_columns = [
            'no2_column', 'o3_column', 'co_column', 'so2_column', 
            'hcho_column', 'aerosol_optical_depth', 'cloud_fraction',
            'observation_time', 'latitude', 'longitude'
        ]
        
        self.openaq_columns = [
            'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
            'latitude', 'longitude', 'elevation', 
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'timestamp'
        ]
        
        # Unit conversion factors (to standardize units)
        self.unit_conversions = {
            'no2': {'ppm_to_ugm3': 1880, 'ppb_to_ugm3': 1.88},
            'o3': {'ppm_to_ugm3': 1960, 'ppb_to_ugm3': 1.96},
            'co': {'ppm_to_ugm3': 1145, 'ppb_to_ugm3': 1.145},
            'so2': {'ppm_to_ugm3': 2620, 'ppb_to_ugm3': 2.62}
        }
    
    def preprocess_tempo_data(self, tempo_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess TEMPO satellite data.
        
        Args:
            tempo_data: Raw TEMPO satellite observations
            
        Returns:
            Cleaned and normalized TEMPO data
        """
        logger.info("Preprocessing TEMPO satellite data...")
        
        df = tempo_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df, data_type='tempo')
        
        # Normalize column densities (convert to standard units)
        df = self._normalize_tempo_columns(df)
        
        # Filter by data quality
        df = self._filter_tempo_quality(df)
        
        # Add derived features
        df = self._add_tempo_features(df)
        
        logger.info(f"TEMPO data preprocessed: {len(df)} records")
        # Reset index to ensure sequential indices for spatial-temporal matching
        return df.reset_index(drop=True)
    
    def preprocess_openaq_data(self, openaq_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess OpenAQ ground station data.
        
        Args:
            openaq_data: Raw OpenAQ ground measurements
            
        Returns:
            Cleaned and normalized OpenAQ data
        """
        logger.info("Preprocessing OpenAQ ground station data...")
        
        df = openaq_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df, data_type='openaq')
        
        # Convert units to standard (µg/m³)
        df = self._standardize_units(df)
        
        # Filter outliers
        df = self._filter_outliers(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        logger.info(f"OpenAQ data preprocessed: {len(df)} records")
        # Reset index to ensure sequential indices for spatial-temporal matching
        return df.reset_index(drop=True)
    
    def _handle_missing_values(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle missing values based on data type."""
        initial_count = len(df)
        
        if data_type == 'tempo':
            # Remove rows with missing critical satellite data
            critical_cols = ['no2_column', 'o3_column', 'latitude', 'longitude']
            df = df.dropna(subset=critical_cols)
            
            # Fill cloud fraction with median
            df['cloud_fraction'] = df['cloud_fraction'].fillna(df['cloud_fraction'].median())
            
        elif data_type == 'openaq':
            # Remove rows without location data
            df = df.dropna(subset=['latitude', 'longitude'])
            
            # Forward fill meteorological data (interpolation)
            weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction']
            for col in weather_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} rows with missing critical data")
        
        return df
    
    def _normalize_tempo_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize TEMPO column densities to standard ranges."""
        column_features = ['no2_column', 'o3_column', 'co_column', 'so2_column', 'hcho_column']
        
        for col in column_features:
            if col in df.columns:
                # Log transform for skewed distributions
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
                
                # Z-score normalization
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f'{col}_norm'] = (df[col] - mean_val) / std_val
        
        return df
    
    def _standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all pollutant concentrations to µg/m³."""
        for pollutant, conversions in self.unit_conversions.items():
            if pollutant in df.columns:
                # Assume input is in ppb, convert to µg/m³
                df[f'{pollutant}_ugm3'] = df[pollutant] * conversions['ppb_to_ugm3']
        
        return df
    
    def _filter_tempo_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter TEMPO data based on quality criteria."""
        initial_count = len(df)
        
        # Filter by cloud fraction (less than 30% cloud cover)
        if 'cloud_fraction' in df.columns:
            df = df[df['cloud_fraction'] < 0.3]
        
        # Filter extreme values (beyond reasonable atmospheric ranges)
        if 'no2_column' in df.columns:
            df = df[df['no2_column'] < 1e17]  # molecules/cm²
        
        if 'o3_column' in df.columns:
            df = df[df['o3_column'] < 5e18]  # molecules/cm²
        
        filtered_count = initial_count - len(df)
        logger.info(f"Filtered {filtered_count} TEMPO records for quality")
        
        return df
    
    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from ground station data."""
        initial_count = len(df)
        
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        
        for col in pollutant_cols:
            if col in df.columns:
                # Use IQR method to identify outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        filtered_count = initial_count - len(df)
        logger.info(f"Removed {filtered_count} outliers from ground data")
        
        return df
    
    def _add_tempo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from TEMPO data."""
        # Time-based features
        if 'observation_time' in df.columns:
            df['observation_time'] = pd.to_datetime(df['observation_time'])
            df['hour'] = df['observation_time'].dt.hour
            df['day_of_year'] = df['observation_time'].dt.dayofyear
            df['is_weekend'] = df['observation_time'].dt.weekday >= 5
        
        # Ratio features
        if 'no2_column' in df.columns and 'o3_column' in df.columns:
            df['no2_o3_ratio'] = df['no2_column'] / (df['o3_column'] + 1e-10)
        
        # Aerosol-pollutant interaction
        if 'aerosol_optical_depth' in df.columns and 'no2_column' in df.columns:
            df['aod_no2_interaction'] = df['aerosol_optical_depth'] * df['no2_column']
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to ground station data."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
            df['is_weekend'] = df['day_of_week'] >= 5
        
        return df
    
    def create_feature_matrix(self, tempo_df: pd.DataFrame, openaq_df: pd.DataFrame, 
                            matched_pairs: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix and target vector from matched data pairs.
        
        Args:
            tempo_df: Preprocessed TEMPO data
            openaq_df: Preprocessed OpenAQ data
            matched_pairs: List of (tempo_idx, openaq_idx) pairs
            
        Returns:
            Feature matrix X and target vector y
        """
        features = []
        targets = []
        
        for tempo_idx, openaq_idx in matched_pairs:
            # Use .iloc since DataFrames have reset sequential indices after preprocessing
            tempo_row = tempo_df.iloc[tempo_idx]
            openaq_row = openaq_df.iloc[openaq_idx]
            
            # Combine TEMPO and weather features
            feature_vector = self._extract_features(tempo_row, openaq_row)
            target_vector = self._extract_targets(openaq_row)
            
            if feature_vector is not None and target_vector is not None:
                features.append(feature_vector)
                targets.append(target_vector)
        
        X = np.array(features)
        y = np.array(targets)
        
        logger.info(f"Created feature matrix: {X.shape}, target matrix: {y.shape}")
        return X, y
    
    def _extract_features(self, tempo_row: pd.Series, openaq_row: pd.Series) -> Optional[np.ndarray]:
        """Extract feature vector from matched TEMPO and OpenAQ data."""
        try:
            features = []
            
            # TEMPO satellite features
            satellite_features = [
                'no2_column_norm', 'o3_column_norm', 'co_column_norm', 
                'so2_column_norm', 'hcho_column_norm', 'aerosol_optical_depth',
                'cloud_fraction', 'no2_o3_ratio', 'aod_no2_interaction'
            ]
            
            for feat in satellite_features:
                if feat in tempo_row.index:
                    features.append(tempo_row[feat])
                else:
                    features.append(0.0)  # Default value
            
            # Location features
            features.extend([
                tempo_row.get('latitude', 0.0),
                tempo_row.get('longitude', 0.0),
                openaq_row.get('elevation', 0.0)
            ])
            
            # Meteorological features
            weather_features = ['temperature', 'humidity', 'wind_speed', 'wind_direction']
            for feat in weather_features:
                features.append(openaq_row.get(feat, 0.0))
            
            # Temporal features
            temporal_features = ['hour', 'day_of_week', 'month', 'is_rush_hour', 'is_weekend']
            for feat in temporal_features:
                features.append(openaq_row.get(feat, 0))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return None
    
    def _extract_targets(self, openaq_row: pd.Series) -> Optional[np.ndarray]:
        """Extract target pollutant concentrations."""
        try:
            targets = []
            target_pollutants = ['pm25', 'pm10', 'no2_ugm3', 'o3_ugm3', 'so2_ugm3', 'co_ugm3']
            
            for pollutant in target_pollutants:
                if pollutant in openaq_row.index and not pd.isna(openaq_row[pollutant]):
                    targets.append(openaq_row[pollutant])
                else:
                    return None  # Skip incomplete target data
            
            return np.array(targets, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting targets: {e}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for model interpretation."""
        return [
            # TEMPO satellite features
            'no2_column_norm', 'o3_column_norm', 'co_column_norm', 
            'so2_column_norm', 'hcho_column_norm', 'aerosol_optical_depth',
            'cloud_fraction', 'no2_o3_ratio', 'aod_no2_interaction',
            
            # Location features
            'latitude', 'longitude', 'elevation',
            
            # Meteorological features
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            
            # Temporal features
            'hour', 'day_of_week', 'month', 'is_rush_hour', 'is_weekend'
        ]
    
    def get_target_names(self) -> List[str]:
        """Get list of target pollutant names."""
        return ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']