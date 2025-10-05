"""
Spatial-Temporal Fusion Module

This module handles the matching and alignment of TEMPO satellite data
with OpenAQ ground station data in space and time.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

class SpatialTemporalFusion:
    """Handles spatial and temporal alignment of satellite and ground data."""
    
    def __init__(self, 
                 spatial_threshold_km: float = 25.0,
                 temporal_threshold_hours: float = 1.0,
                 max_matches_per_point: int = 3):
        """
        Initialize the spatial-temporal fusion module.
        
        Args:
            spatial_threshold_km: Maximum spatial distance for matching (km)
            temporal_threshold_hours: Maximum temporal difference for matching (hours)
            max_matches_per_point: Maximum number of matches per ground station
        """
        self.spatial_threshold_km = spatial_threshold_km
        self.temporal_threshold_hours = temporal_threshold_hours
        self.max_matches_per_point = max_matches_per_point
        
        # Earth's radius in km
        self.earth_radius_km = 6371.0
    
    def find_collocated_pairs(self, 
                             tempo_df: pd.DataFrame, 
                             openaq_df: pd.DataFrame) -> List[Tuple[int, int, float, float]]:
        """
        Find spatially and temporally collocated TEMPO-OpenAQ pairs.
        
        Args:
            tempo_df: TEMPO satellite data with lat, lon, observation_time
            openaq_df: OpenAQ ground data with lat, lon, timestamp
            
        Returns:
            List of (tempo_idx, openaq_idx, spatial_dist_km, temporal_diff_hours) tuples
        """
        logger.info("Finding collocated satellite-ground data pairs...")
        
        matches = []
        
        # Convert timestamps to datetime if needed
        tempo_times = pd.to_datetime(tempo_df['observation_time'])
        openaq_times = pd.to_datetime(openaq_df['timestamp'])
        
        # Build spatial index for efficient nearest neighbor search
        nn_model = NearestNeighbors(
            n_neighbors=min(len(tempo_df), 10),
            metric='haversine',
            radius=self.spatial_threshold_km / self.earth_radius_km
        )
        
        # Convert coordinates to radians for haversine distance
        tempo_coords_rad = np.radians(tempo_df[['latitude', 'longitude']].values)
        openaq_coords_rad = np.radians(openaq_df[['latitude', 'longitude']].values)
        
        nn_model.fit(tempo_coords_rad)
        
        # Find matches for each ground station
        for row_idx, (openaq_idx, openaq_row) in enumerate(openaq_df.iterrows()):
            openaq_time = openaq_times.iloc[row_idx]
            openaq_coord_rad = openaq_coords_rad[row_idx:row_idx+1]
            
            # Find spatially nearby TEMPO observations
            distances, indices = nn_model.radius_neighbors(
                openaq_coord_rad,
                radius=self.spatial_threshold_km / self.earth_radius_km
            )
            
            if len(indices[0]) == 0:
                continue
                
            # Check temporal alignment for each spatial match
            for i, tempo_idx in enumerate(indices[0]):
                tempo_time = tempo_times.iloc[tempo_idx]
                
                # Calculate temporal difference
                time_diff_hours = abs((tempo_time - openaq_time).total_seconds()) / 3600.0
                
                if time_diff_hours <= self.temporal_threshold_hours:
                    # Convert spatial distance back to km
                    spatial_dist_km = distances[0][i] * self.earth_radius_km
                    
                    matches.append((
                        tempo_idx, 
                        openaq_idx, 
                        spatial_dist_km, 
                        time_diff_hours
                    ))
        
        # Sort by quality (spatial distance + temporal difference)
        matches.sort(key=lambda x: x[2] + x[3])  # Distance + time difference
        
        # Limit matches per ground station
        matches = self._limit_matches_per_station(matches)
        
        logger.info(f"Found {len(matches)} collocated pairs")
        return matches
    
    def _limit_matches_per_station(self, 
                                  matches: List[Tuple[int, int, float, float]]) -> List[Tuple[int, int, float, float]]:
        """Limit the number of matches per ground station to avoid bias."""
        station_counts = {}
        filtered_matches = []
        
        for match in matches:
            tempo_idx, openaq_idx, spatial_dist, temporal_diff = match
            
            if openaq_idx not in station_counts:
                station_counts[openaq_idx] = 0
            
            if station_counts[openaq_idx] < self.max_matches_per_point:
                filtered_matches.append(match)
                station_counts[openaq_idx] += 1
        
        return filtered_matches
    
    def create_hourly_averages(self, 
                              tempo_df: pd.DataFrame, 
                              openaq_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create hourly averaged datasets for more stable matching.
        
        Args:
            tempo_df: Raw TEMPO data
            openaq_df: Raw OpenAQ data
            
        Returns:
            Hourly averaged TEMPO and OpenAQ dataframes
        """
        logger.info("Creating hourly averages...")
        
        # Process TEMPO data
        tempo_hourly = self._create_tempo_hourly_avg(tempo_df)
        
        # Process OpenAQ data
        openaq_hourly = self._create_openaq_hourly_avg(openaq_df)
        
        logger.info(f"Created hourly averages: TEMPO {len(tempo_hourly)}, OpenAQ {len(openaq_hourly)}")
        return tempo_hourly, openaq_hourly
    
    def _create_tempo_hourly_avg(self, tempo_df: pd.DataFrame) -> pd.DataFrame:
        """Create hourly averages for TEMPO satellite data."""
        df = tempo_df.copy()
        df['observation_time'] = pd.to_datetime(df['observation_time'])
        
        # Create spatial-temporal grouping key
        df['lat_bin'] = np.round(df['latitude'], 2)  # ~1km resolution
        df['lon_bin'] = np.round(df['longitude'], 2)
        df['hour_bin'] = df['observation_time'].dt.floor('H')
        
        # Group and average
        grouping_cols = ['lat_bin', 'lon_bin', 'hour_bin']
        
        agg_dict = {
            'latitude': 'mean',
            'longitude': 'mean',
            'observation_time': 'first'
        }
        
        # Add satellite measurement columns
        satellite_cols = [
            'no2_column', 'o3_column', 'co_column', 'so2_column',
            'hcho_column', 'aerosol_optical_depth', 'cloud_fraction'
        ]
        
        for col in satellite_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # Add derived features if they exist
        derived_cols = [
            'no2_column_norm', 'o3_column_norm', 'co_column_norm',
            'so2_column_norm', 'hcho_column_norm', 'no2_o3_ratio',
            'aod_no2_interaction'
        ]
        
        for col in derived_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        hourly_df = df.groupby(grouping_cols).agg(agg_dict).reset_index()
        
        # Clean up
        hourly_df = hourly_df.drop(['lat_bin', 'lon_bin', 'hour_bin'], axis=1)
        
        return hourly_df
    
    def _create_openaq_hourly_avg(self, openaq_df: pd.DataFrame) -> pd.DataFrame:
        """Create hourly averages for OpenAQ ground station data."""
        df = openaq_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create spatial-temporal grouping key (by station and hour)
        df['lat_bin'] = np.round(df['latitude'], 3)  # Station precision
        df['lon_bin'] = np.round(df['longitude'], 3)
        df['hour_bin'] = df['timestamp'].dt.floor('H')
        
        # Group and average
        grouping_cols = ['lat_bin', 'lon_bin', 'hour_bin']
        
        agg_dict = {
            'latitude': 'mean',
            'longitude': 'mean',
            'timestamp': 'first',
            'elevation': 'mean'
        }
        
        # Add pollutant columns
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for col in pollutant_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # Add unit-converted columns
        unit_cols = ['no2_ugm3', 'o3_ugm3', 'so2_ugm3', 'co_ugm3']
        for col in unit_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # Add meteorological columns
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction']
        for col in weather_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        # Add temporal features (take first occurrence)
        temporal_cols = ['hour', 'day_of_week', 'month', 'is_rush_hour', 'is_weekend']
        for col in temporal_cols:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        hourly_df = df.groupby(grouping_cols).agg(agg_dict).reset_index()
        
        # Clean up
        hourly_df = hourly_df.drop(['lat_bin', 'lon_bin', 'hour_bin'], axis=1)
        
        return hourly_df
    
    def create_daily_averages(self, 
                             tempo_df: pd.DataFrame, 
                             openaq_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create daily averaged datasets for broader temporal matching.
        
        Args:
            tempo_df: Raw TEMPO data
            openaq_df: Raw OpenAQ data
            
        Returns:
            Daily averaged TEMPO and OpenAQ dataframes
        """
        logger.info("Creating daily averages...")
        
        # Similar to hourly but group by date instead of hour
        tempo_daily = self._create_tempo_daily_avg(tempo_df)
        openaq_daily = self._create_openaq_daily_avg(openaq_df)
        
        logger.info(f"Created daily averages: TEMPO {len(tempo_daily)}, OpenAQ {len(openaq_daily)}")
        return tempo_daily, openaq_daily
    
    def _create_tempo_daily_avg(self, tempo_df: pd.DataFrame) -> pd.DataFrame:
        """Create daily averages for TEMPO satellite data."""
        df = tempo_df.copy()
        df['observation_time'] = pd.to_datetime(df['observation_time'])
        
        # Create spatial-temporal grouping key
        df['lat_bin'] = np.round(df['latitude'], 2)
        df['lon_bin'] = np.round(df['longitude'], 2)
        df['date_bin'] = df['observation_time'].dt.date
        
        # Group and average (similar to hourly but by date)
        grouping_cols = ['lat_bin', 'lon_bin', 'date_bin']
        
        agg_dict = {
            'latitude': 'mean',
            'longitude': 'mean',
            'observation_time': 'first'
        }
        
        # Add all measurement columns
        satellite_cols = [
            'no2_column', 'o3_column', 'co_column', 'so2_column',
            'hcho_column', 'aerosol_optical_depth', 'cloud_fraction'
        ]
        
        for col in satellite_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        daily_df = df.groupby(grouping_cols).agg(agg_dict).reset_index()
        daily_df = daily_df.drop(['lat_bin', 'lon_bin', 'date_bin'], axis=1)
        
        return daily_df
    
    def _create_openaq_daily_avg(self, openaq_df: pd.DataFrame) -> pd.DataFrame:
        """Create daily averages for OpenAQ ground station data."""
        df = openaq_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create spatial-temporal grouping key
        df['lat_bin'] = np.round(df['latitude'], 3)
        df['lon_bin'] = np.round(df['longitude'], 3)
        df['date_bin'] = df['timestamp'].dt.date
        
        # Group and average
        grouping_cols = ['lat_bin', 'lon_bin', 'date_bin']
        
        agg_dict = {
            'latitude': 'mean',
            'longitude': 'mean',
            'timestamp': 'first',
            'elevation': 'mean'
        }
        
        # Add all measurement columns
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction']
        
        for col in pollutant_cols + weather_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        daily_df = df.groupby(grouping_cols).agg(agg_dict).reset_index()
        daily_df = daily_df.drop(['lat_bin', 'lon_bin', 'date_bin'], axis=1)
        
        return daily_df
    
    def interpolate_missing_locations(self, 
                                    tempo_df: pd.DataFrame,
                                    target_locations: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        Interpolate TEMPO data for specific locations using spatial interpolation.
        
        Args:
            tempo_df: TEMPO satellite data
            target_locations: List of (lat, lon) tuples for interpolation
            
        Returns:
            Interpolated TEMPO data at target locations
        """
        logger.info(f"Interpolating TEMPO data for {len(target_locations)} locations...")
        
        from scipy.interpolate import griddata
        
        interpolated_data = []
        
        # Get TEMPO coordinates and values
        tempo_coords = tempo_df[['latitude', 'longitude']].values
        
        # Columns to interpolate
        interp_cols = [
            'no2_column', 'o3_column', 'co_column', 'so2_column',
            'hcho_column', 'aerosol_optical_depth', 'cloud_fraction'
        ]
        
        for target_lat, target_lon in target_locations:
            row_data = {
                'latitude': target_lat,
                'longitude': target_lon,
                'observation_time': tempo_df['observation_time'].iloc[0]  # Use first timestamp
            }
            
            # Interpolate each column
            for col in interp_cols:
                if col in tempo_df.columns:
                    values = tempo_df[col].values
                    
                    # Remove NaN values for interpolation
                    valid_mask = ~np.isnan(values)
                    if np.sum(valid_mask) >= 3:  # Need at least 3 points
                        interpolated_val = griddata(
                            tempo_coords[valid_mask],
                            values[valid_mask],
                            [(target_lat, target_lon)],
                            method='linear',
                            fill_value=np.nan
                        )[0]
                        row_data[col] = interpolated_val
                    else:
                        row_data[col] = np.nan
                else:
                    row_data[col] = np.nan
            
            interpolated_data.append(row_data)
        
        interpolated_df = pd.DataFrame(interpolated_data)
        logger.info(f"Completed interpolation for {len(interpolated_df)} locations")
        
        return interpolated_df
    
    def validate_matches(self, matches: List[Tuple[int, int, float, float]]) -> List[Tuple[int, int, float, float]]:
        """
        Validate and filter matches based on quality criteria.
        
        Args:
            matches: List of (tempo_idx, openaq_idx, spatial_dist, temporal_diff) tuples
            
        Returns:
            Filtered list of high-quality matches
        """
        logger.info(f"Validating {len(matches)} matches...")
        
        # Sort by combined distance metric
        quality_scores = []
        for tempo_idx, openaq_idx, spatial_dist, temporal_diff in matches:
            # Normalized quality score (lower is better)
            spatial_score = spatial_dist / self.spatial_threshold_km
            temporal_score = temporal_diff / self.temporal_threshold_hours
            combined_score = spatial_score + temporal_score
            quality_scores.append((tempo_idx, openaq_idx, spatial_dist, temporal_diff, combined_score))
        
        # Sort by quality score
        quality_scores.sort(key=lambda x: x[4])
        
        # Keep top quality matches (within reasonable limits)
        max_matches = min(len(quality_scores), 10000)  # Practical limit
        validated_matches = [
            (tempo_idx, openaq_idx, spatial_dist, temporal_diff)
            for tempo_idx, openaq_idx, spatial_dist, temporal_diff, _ in quality_scores[:max_matches]
        ]
        
        logger.info(f"Validated {len(validated_matches)} high-quality matches")
        return validated_matches