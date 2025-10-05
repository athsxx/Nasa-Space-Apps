"""
Weather Integration Module

This module handles integration with weather APIs and services
to provide meteorological data for surface concentration prediction.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class WeatherIntegration:
    """Handles weather data acquisition and processing for ML models."""
    
    def __init__(self, 
                 openweather_api_key: Optional[str] = None,
                 default_weather: Optional[Dict[str, float]] = None):
        """
        Initialize weather integration.
        
        Args:
            openweather_api_key: API key for OpenWeatherMap (optional)
            default_weather: Default weather values when API unavailable
        """
        self.openweather_api_key = openweather_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Default weather values for fallback
        self.default_weather = default_weather or {
            'temperature': 20.0,     # °C
            'humidity': 50.0,        # %
            'wind_speed': 3.0,       # m/s
            'wind_direction': 180.0, # degrees
            'pressure': 1013.25,     # hPa
            'visibility': 10000.0    # meters
        }
    
    def get_current_weather(self, 
                          latitude: float, 
                          longitude: float) -> Dict[str, float]:
        """
        Get current weather conditions for a location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            
        Returns:
            Dictionary with weather parameters
        """
        if self.openweather_api_key:
            try:
                return self._fetch_openweather_current(latitude, longitude)
            except Exception as e:
                logger.warning(f"Weather API failed: {e}. Using defaults.")
        
        # Return default weather with some location-based variation
        return self._generate_location_weather(latitude, longitude)
    
    def get_historical_weather(self, 
                             latitude: float, 
                             longitude: float,
                             start_date: datetime,
                             end_date: datetime) -> pd.DataFrame:
        """
        Get historical weather data for training.
        
        Args:
            latitude: Location latitude  
            longitude: Location longitude
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical weather data
        """
        if self.openweather_api_key:
            try:
                return self._fetch_openweather_historical(
                    latitude, longitude, start_date, end_date
                )
            except Exception as e:
                logger.warning(f"Historical weather API failed: {e}. Generating synthetic data.")
        
        # Generate synthetic historical weather
        return self._generate_synthetic_weather(latitude, longitude, start_date, end_date)
    
    def _fetch_openweather_current(self, 
                                 latitude: float, 
                                 longitude: float) -> Dict[str, float]:
        """Fetch current weather from OpenWeatherMap API."""
        url = f"{self.base_url}/weather"
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': self.openweather_api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data.get('wind', {}).get('speed', 0.0),
            'wind_direction': data.get('wind', {}).get('deg', 0.0),
            'visibility': data.get('visibility', 10000.0)
        }
    
    def _fetch_openweather_historical(self, 
                                    latitude: float, 
                                    longitude: float,
                                    start_date: datetime,
                                    end_date: datetime) -> pd.DataFrame:
        """Fetch historical weather from OpenWeatherMap One Call API."""
        # Note: Historical data requires subscription
        weather_data = []
        
        current = start_date
        while current <= end_date:
            timestamp = int(current.timestamp())
            
            url = f"{self.base_url}/onecall/timemachine"
            params = {
                'lat': latitude,
                'lon': longitude,
                'dt': timestamp,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'current' in data:
                    weather_point = {
                        'timestamp': current,
                        'latitude': latitude,
                        'longitude': longitude,
                        'temperature': data['current']['temp'],
                        'humidity': data['current']['humidity'],
                        'pressure': data['current']['pressure'],
                        'wind_speed': data['current'].get('wind_speed', 0.0),
                        'wind_direction': data['current'].get('wind_deg', 0.0),
                        'visibility': data['current'].get('visibility', 10000.0)
                    }
                    weather_data.append(weather_point)
                
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch weather for {current}: {e}")
            
            current += timedelta(hours=1)
        
        return pd.DataFrame(weather_data)
    
    def _generate_location_weather(self, 
                                 latitude: float, 
                                 longitude: float) -> Dict[str, float]:
        """Generate location-appropriate weather estimates."""
        weather = self.default_weather.copy()
        
        # Adjust temperature based on latitude (simple model)
        lat_adjustment = (45 - abs(latitude)) * 0.5  # Warmer near equator
        weather['temperature'] += lat_adjustment
        
        # Adjust humidity based on proximity to water bodies (simplified)
        # Higher humidity near coasts (this is very approximate)
        if abs(longitude) > 120 or abs(longitude) < 30:  # Near oceans
            weather['humidity'] += 10
        
        # Add some random variation
        import random
        weather['temperature'] += random.uniform(-5, 5)
        weather['humidity'] = max(20, min(90, weather['humidity'] + random.uniform(-10, 10)))
        weather['wind_speed'] = max(0, weather['wind_speed'] + random.uniform(-1, 3))
        weather['wind_direction'] = (weather['wind_direction'] + random.uniform(-45, 45)) % 360
        
        return weather
    
    def _generate_synthetic_weather(self, 
                                  latitude: float, 
                                  longitude: float,
                                  start_date: datetime,
                                  end_date: datetime) -> pd.DataFrame:
        """Generate synthetic historical weather data."""
        weather_data = []
        
        # Base weather for location
        base_weather = self._generate_location_weather(latitude, longitude)
        
        current = start_date
        while current <= end_date:
            # Seasonal variation
            day_of_year = current.timetuple().tm_yday
            seasonal_temp_adj = 10 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
            
            # Diurnal variation
            hour_temp_adj = 5 * np.sin(2 * np.pi * (current.hour - 6) / 24)
            
            # Random variation
            temp_noise = np.random.normal(0, 2)
            humidity_noise = np.random.normal(0, 5)
            wind_noise = np.random.normal(0, 1)
            
            weather_point = {
                'timestamp': current,
                'latitude': latitude,
                'longitude': longitude,
                'temperature': base_weather['temperature'] + seasonal_temp_adj + hour_temp_adj + temp_noise,
                'humidity': max(10, min(95, base_weather['humidity'] + humidity_noise)),
                'pressure': base_weather['pressure'] + np.random.normal(0, 5),
                'wind_speed': max(0, base_weather['wind_speed'] + wind_noise),
                'wind_direction': (base_weather['wind_direction'] + np.random.uniform(-30, 30)) % 360,
                'visibility': max(1000, base_weather['visibility'] + np.random.normal(0, 1000))
            }
            
            weather_data.append(weather_point)
            current += timedelta(hours=1)
        
        logger.info(f"Generated {len(weather_data)} synthetic weather records")
        return pd.DataFrame(weather_data)
    
    def process_weather_for_ml(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process weather data for machine learning features.
        
        Args:
            weather_df: Raw weather data
            
        Returns:
            Processed weather data with derived features
        """
        df = weather_df.copy()
        
        # Ensure timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add derived weather features
        df = self._add_weather_features(df)
        
        # Handle missing values
        df = self._handle_weather_missing_values(df)
        
        return df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived weather features."""
        # Wind components
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            wind_rad = np.radians(df['wind_direction'])
            df['wind_u'] = df['wind_speed'] * np.cos(wind_rad)  # East-west component
            df['wind_v'] = df['wind_speed'] * np.sin(wind_rad)  # North-south component
        
        # Temperature categories
        if 'temperature' in df.columns:
            df['temp_category'] = pd.cut(
                df['temperature'], 
                bins=[-np.inf, 0, 10, 20, 30, np.inf],
                labels=['very_cold', 'cold', 'mild', 'warm', 'hot']
            )
            df['temp_category_num'] = df['temp_category'].cat.codes
        
        # Humidity categories
        if 'humidity' in df.columns:
            df['humidity_category'] = pd.cut(
                df['humidity'],
                bins=[0, 30, 60, 80, 100],
                labels=['dry', 'moderate', 'humid', 'very_humid']
            )
            df['humidity_category_num'] = df['humidity_category'].cat.codes
        
        # Atmospheric stability indicator
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            # Simple stability classification
            df['atmospheric_stability'] = 0  # Neutral
            
            # Stable conditions (low wind, cool)
            stable_mask = (df['wind_speed'] < 2) & (df['temperature'] < 15)
            df.loc[stable_mask, 'atmospheric_stability'] = -1
            
            # Unstable conditions (high wind, warm)
            unstable_mask = (df['wind_speed'] > 5) & (df['temperature'] > 25)
            df.loc[unstable_mask, 'atmospheric_stability'] = 1
        
        # Heat index (apparent temperature)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
        
        return df
    
    def _calculate_heat_index(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index (apparent temperature)."""
        # Convert to Fahrenheit for standard heat index formula
        temp_f = temp_c * 9/5 + 32
        
        # Simplified heat index calculation
        heat_index_f = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # For higher temperatures, use more complex formula
        complex_mask = temp_f >= 80
        if complex_mask.any():
            T = temp_f[complex_mask]
            H = humidity[complex_mask]
            
            hi = (-42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H 
                  - 6.83783e-3*T**2 - 5.481717e-2*H**2 + 1.22874e-3*T**2*H 
                  + 8.5282e-4*T*H**2 - 1.99e-6*T**2*H**2)
            
            heat_index_f[complex_mask] = hi
        
        # Convert back to Celsius
        return (heat_index_f - 32) * 5/9
    
    def _handle_weather_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in weather data."""
        weather_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'visibility']
        
        for col in weather_columns:
            if col in df.columns:
                # Forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use defaults
                if df[col].isna().any():
                    default_val = self.default_weather.get(col, 0.0)
                    df[col] = df[col].fillna(default_val)
        
        return df
    
    def get_weather_summary_stats(self, weather_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for weather data."""
        if weather_df.empty:
            return {}
        
        stats = {}
        weather_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'visibility']
        
        for col in weather_columns:
            if col in weather_df.columns:
                stats[col] = {
                    'mean': weather_df[col].mean(),
                    'std': weather_df[col].std(),
                    'min': weather_df[col].min(),
                    'max': weather_df[col].max(),
                    'count': weather_df[col].count()
                }
        
        return stats
    
    def create_weather_features_for_location(self, 
                                           latitude: float, 
                                           longitude: float,
                                           timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Create weather feature set for a specific location and time.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            timestamp: Observation time (uses current time if None)
            
        Returns:
            Dictionary of weather features for ML model
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get current weather
        weather = self.get_current_weather(latitude, longitude)
        
        # Add temporal features
        weather['hour'] = timestamp.hour
        weather['day_of_week'] = timestamp.weekday()
        weather['month'] = timestamp.month
        weather['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        
        # Add seasonal features
        day_of_year = timestamp.timetuple().tm_yday
        weather['season_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        weather['season_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Add derived features
        if 'wind_speed' in weather and 'wind_direction' in weather:
            wind_rad = np.radians(weather['wind_direction'])
            weather['wind_u'] = weather['wind_speed'] * np.cos(wind_rad)
            weather['wind_v'] = weather['wind_speed'] * np.sin(wind_rad)
        
        return weather