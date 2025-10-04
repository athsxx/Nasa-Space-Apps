#!/usr/bin/env python3
"""
Integrated Air Quality and Weather Data System

Combines data from multiple sources:
- EPA Air Quality Data (ground-based measurements)
- Satellite Data (NO₂, O₃, AOD)
- Weather Data (Temperature, Humidity, Wind, Precipitation, Solar Radiation, UV)

Creates a unified dataset for air quality analysis and modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom downloaders
from satellite_data_downloader import SatelliteDataDownloader
from weather_data_downloader import WeatherDataDownloader
from epa_data_processor import EPADataProcessor

class IntegratedDataSystem:
    def __init__(self, data_dir="integrated_data"):
        """
        Initialize Integrated Data System
        
        Args:
            data_dir (str): Directory for integrated datasets
        """
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize component systems
        self.epa_processor = EPADataProcessor("validation datasets")
        self.satellite_downloader = SatelliteDataDownloader("satellite_data")
        self.weather_downloader = WeatherDataDownloader("weather_data")
        
        # Data integration parameters
        self.spatial_tolerance = 0.1  # degrees for spatial matching
        self.temporal_tolerance = 1   # days for temporal matching
        
        print(f"Integrated Air Quality Data System initialized")
        print(f"Integration directory: {self.data_dir}")
    
    def load_all_datasets(self, use_sample_data=True):
        """
        Load and prepare all data sources
        
        Args:
            use_sample_data (bool): Whether to use sample data for satellite/weather
        
        Returns:
            dict: Dictionary of loaded datasets
        """
        datasets = {}
        
        print("Loading all data sources...")
        print("="*50)
        
        # 1. Load EPA air quality data
        print("\n1. Loading EPA Air Quality Data...")
        try:
            epa_datasets = self.epa_processor.extract_and_load_data()
            if epa_datasets:
                # Combine all EPA parameters into one DataFrame
                epa_combined = []
                
                for param_code, dataset in epa_datasets.items():
                    df = dataset['data'].copy()
                    df['parameter_code'] = param_code
                    df['pollutant'] = dataset['info'].get('pollutant', param_code)
                    epa_combined.append(df)
                
                if epa_combined:
                    datasets['epa'] = pd.concat(epa_combined, ignore_index=True)
                    print(f"  Loaded EPA data: {len(datasets['epa']):,} records")
                else:
                    print("  No EPA data loaded")
            else:
                print("  No EPA datasets found")
                
        except Exception as e:
            print(f"  Error loading EPA data: {e}")
        
        # 2. Load/Generate Satellite Data
        print("\n2. Loading Satellite Data...")
        try:
            # Define date range and bbox
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            north_america_bbox = (-180, 10, -50, 85)
            
            if use_sample_data:
                # Generate sample satellite data
                satellite_data = []
                
                for param in ['no2', 'o3', 'aod']:
                    df = self.satellite_downloader.create_sample_data(
                        param, north_america_bbox, start_date, end_date, num_points=500
                    )
                    satellite_data.append(df)
                
                if satellite_data:
                    datasets['satellite'] = pd.concat(satellite_data, ignore_index=True)
                    print(f"  Generated satellite data: {len(datasets['satellite']):,} records")
            else:
                # Try to download real satellite data
                satellite_results = self.satellite_downloader.download_all_satellite_data(
                    north_america_bbox, start_date, end_date, use_sample_data=False
                )
                print(f"  Satellite data download attempted - check results")
                
        except Exception as e:
            print(f"  Error with satellite data: {e}")
        
        # 3. Load/Generate Weather Data
        print("\n3. Loading Weather Data...")
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            north_america_bbox = (-180, 10, -50, 85)
            
            if use_sample_data:
                # Generate sample weather data
                datasets['weather'] = self.weather_downloader.create_sample_weather_data(
                    north_america_bbox, start_date, end_date, num_locations=100
                )
                print(f"  Generated weather data: {len(datasets['weather']):,} records")
            else:
                # Try to download real weather data
                weather_results = self.weather_downloader.download_all_weather_data(
                    north_america_bbox, start_date, end_date, use_sample_data=False
                )
                print(f"  Weather data download attempted - check results")
                
        except Exception as e:
            print(f"  Error with weather data: {e}")
        
        return datasets
    
    def create_spatial_grid(self, bbox, resolution=1.0):
        """
        Create a spatial grid for data integration
        
        Args:
            bbox (tuple): (west, south, east, north) bounding box
            resolution (float): Grid resolution in degrees
        
        Returns:
            pandas.DataFrame: Grid points with lat/lon
        """
        west, south, east, north = bbox
        
        # Create grid points
        lats = np.arange(south, north + resolution, resolution)
        lons = np.arange(west, east + resolution, resolution)
        
        grid_points = []
        for lat in lats:
            for lon in lons:
                grid_points.append({
                    'grid_lat': lat,
                    'grid_lon': lon,
                    'grid_id': f"GRID_{lat:.1f}_{lon:.1f}"
                })
        
        return pd.DataFrame(grid_points)
    
    def spatial_join_datasets(self, datasets, target_resolution=1.0):
        """
        Spatially join datasets to common grid
        
        Args:
            datasets (dict): Dictionary of datasets
            target_resolution (float): Target grid resolution in degrees
        
        Returns:
            pandas.DataFrame: Spatially integrated dataset
        """
        print(f"\nPerforming spatial integration...")
        print(f"Target resolution: {target_resolution}° grid")
        
        # Determine overall bounding box
        all_lats = []
        all_lons = []
        
        for name, df in datasets.items():
            if 'latitude' in df.columns and 'longitude' in df.columns:
                all_lats.extend(df['latitude'].dropna().tolist())
                all_lons.extend(df['longitude'].dropna().tolist())
        
        if not all_lats:
            print("No spatial data found for integration")
            return pd.DataFrame()
        
        bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
        print(f"Data bounding box: {bbox}")
        
        # Create spatial grid
        grid = self.create_spatial_grid(bbox, target_resolution)
        print(f"Created grid: {len(grid)} points")
        
        integrated_data = []
        
        for _, grid_point in grid.iterrows():
            grid_lat = grid_point['grid_lat']
            grid_lon = grid_point['grid_lon']
            grid_id = grid_point['grid_id']
            
            record = {
                'grid_id': grid_id,
                'latitude': grid_lat,
                'longitude': grid_lon
            }
            
            # Find nearest data points from each dataset
            for dataset_name, df in datasets.items():
                if 'latitude' not in df.columns or 'longitude' not in df.columns:
                    continue
                
                # Calculate distances to grid point
                df_subset = df.dropna(subset=['latitude', 'longitude']).copy()
                
                if len(df_subset) > 0:
                    distances = np.sqrt(
                        (df_subset['latitude'] - grid_lat)**2 + 
                        (df_subset['longitude'] - grid_lon)**2
                    )
                    
                    # Find points within tolerance
                    nearby_indices = distances <= self.spatial_tolerance
                    nearby_data = df_subset[nearby_indices]
                    
                    if len(nearby_data) > 0:
                        # Aggregate data for this grid cell
                        if dataset_name == 'epa':
                            # EPA data - group by pollutant
                            for pollutant in nearby_data['pollutant'].unique():
                                pollutant_data = nearby_data[nearby_data['pollutant'] == pollutant]
                                if len(pollutant_data) > 0:
                                    avg_value = pollutant_data['Sample Measurement'].mean()
                                    record[f'epa_{pollutant.lower()}_avg'] = avg_value
                                    record[f'epa_{pollutant.lower()}_count'] = len(pollutant_data)
                        
                        elif dataset_name == 'satellite':
                            # Satellite data - group by parameter
                            for param in nearby_data['parameter'].unique():
                                param_data = nearby_data[nearby_data['parameter'] == param]
                                if len(param_data) > 0:
                                    avg_value = param_data['value'].mean()
                                    record[f'sat_{param}_avg'] = avg_value
                                    record[f'sat_{param}_count'] = len(param_data)
                        
                        elif dataset_name == 'weather':
                            # Weather data - average all parameters
                            weather_params = ['temperature', 'humidity', 'wind_speed', 
                                            'wind_u', 'wind_v', 'precipitation', 
                                            'solar_radiation', 'uv_index']
                            
                            for param in weather_params:
                                if param in nearby_data.columns:
                                    param_data = nearby_data[param].dropna()
                                    if len(param_data) > 0:
                                        record[f'weather_{param}_avg'] = param_data.mean()
                                        record[f'weather_{param}_count'] = len(param_data)
            
            integrated_data.append(record)
        
        integrated_df = pd.DataFrame(integrated_data)
        
        # Save integrated dataset
        output_file = os.path.join(self.data_dir, f"integrated_air_quality_data_{target_resolution}deg.csv")
        integrated_df.to_csv(output_file, index=False)
        
        print(f"Spatial integration complete: {len(integrated_df)} grid points")
        print(f"Saved to: {output_file}")
        
        return integrated_df
    
    def create_temporal_series(self, datasets, aggregation_period='daily'):
        """
        Create temporal series from all datasets
        
        Args:
            datasets (dict): Dictionary of datasets
            aggregation_period (str): 'daily', 'weekly', or 'monthly'
        
        Returns:
            pandas.DataFrame: Temporally aggregated dataset
        """
        print(f"\nCreating temporal series ({aggregation_period} aggregation)...")
        
        temporal_data = []
        
        # Process each dataset
        for dataset_name, df in datasets.items():
            print(f"Processing {dataset_name}...")
            
            # Identify date column
            date_col = None
            for col in ['date', 'Date Local', 'datetime']:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                print(f"  No date column found in {dataset_name}")
                continue
            
            # Convert to datetime
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time[date_col])
            
            # Set aggregation period
            if aggregation_period == 'daily':
                df_time['period'] = df_time['date'].dt.date
            elif aggregation_period == 'weekly':
                df_time['period'] = df_time['date'].dt.to_period('W')
            elif aggregation_period == 'monthly':
                df_time['period'] = df_time['date'].dt.to_period('M')
            
            # Aggregate by period
            if dataset_name == 'epa':
                # Group by period and pollutant
                agg_data = df_time.groupby(['period', 'pollutant']).agg({
                    'Sample Measurement': ['mean', 'count', 'std'],
                    'State Name': lambda x: list(x.unique()),
                }).round(6)
                
                agg_data.columns = ['mean_value', 'count', 'std_value', 'states']
                agg_data = agg_data.reset_index()
                agg_data['dataset'] = 'EPA'
                
                temporal_data.append(agg_data)
            
            elif dataset_name == 'satellite':
                # Group by period and parameter
                agg_data = df_time.groupby(['period', 'parameter']).agg({
                    'value': ['mean', 'count', 'std']
                }).round(6)
                
                agg_data.columns = ['mean_value', 'count', 'std_value']
                agg_data = agg_data.reset_index()
                agg_data['dataset'] = 'Satellite'
                
                temporal_data.append(agg_data)
            
            elif dataset_name == 'weather':
                # Aggregate all weather parameters
                weather_cols = ['temperature', 'humidity', 'wind_speed', 
                               'precipitation', 'solar_radiation', 'uv_index']
                
                for param in weather_cols:
                    if param in df_time.columns:
                        param_data = df_time.groupby('period')[param].agg(['mean', 'count', 'std']).round(6)
                        param_data = param_data.reset_index()
                        param_data['parameter'] = param
                        param_data['dataset'] = 'Weather'
                        param_data.columns = ['period', 'mean_value', 'count', 'std_value', 'parameter', 'dataset']
                        
                        temporal_data.append(param_data)
        
        if temporal_data:
            # Combine all temporal data
            combined_temporal = pd.concat(temporal_data, ignore_index=True)
            
            # Save temporal series
            output_file = os.path.join(self.data_dir, f"temporal_series_{aggregation_period}.csv")
            combined_temporal.to_csv(output_file, index=False)
            
            print(f"Temporal series created: {len(combined_temporal)} records")
            print(f"Saved to: {output_file}")
            
            return combined_temporal
        
        return pd.DataFrame()
    
    def create_correlation_analysis(self, integrated_df):
        """
        Analyze correlations between air quality, satellite, and weather data
        
        Args:
            integrated_df (pandas.DataFrame): Integrated dataset
        
        Returns:
            pandas.DataFrame: Correlation matrix
        """
        print(f"\nPerforming correlation analysis...")
        
        # Select numeric columns for correlation
        numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
        
        if len(correlation_cols) < 2:
            print("Insufficient numeric data for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = integrated_df[correlation_cols].corr()
        
        # Save correlation matrix
        output_file = os.path.join(self.data_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(output_file)
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # Filter for interesting correlations (remove perfect self-correlations)
        mask = np.triu(np.ones_like(corr_matrix))
        
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Air Quality, Satellite, and Weather Data Correlations', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.data_dir, "correlation_heatmap.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f"Correlation analysis complete")
        print(f"Matrix saved to: {output_file}")
        print(f"Heatmap saved to: {plot_file}")
        
        return corr_matrix
    
    def generate_summary_report(self, datasets, integrated_df, temporal_df, correlation_matrix):
        """
        Generate comprehensive summary report
        
        Args:
            datasets (dict): Original datasets
            integrated_df (pandas.DataFrame): Integrated spatial dataset
            temporal_df (pandas.DataFrame): Temporal series
            correlation_matrix (pandas.DataFrame): Correlation matrix
        
        Returns:
            dict: Summary statistics
        """
        print(f"\nGenerating summary report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'integration': {},
            'correlations': {}
        }
        
        # Original dataset summary
        for name, df in datasets.items():
            report['datasets'][name] = {
                'records': len(df),
                'columns': list(df.columns),
                'date_range': None,
                'spatial_extent': None
            }
            
            # Add date range if available
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols and len(df) > 0:
                try:
                    dates = pd.to_datetime(df[date_cols[0]]).dropna()
                    if len(dates) > 0:
                        report['datasets'][name]['date_range'] = {
                            'start': dates.min().isoformat(),
                            'end': dates.max().isoformat()
                        }
                except:
                    pass
            
            # Add spatial extent if available
            if 'latitude' in df.columns and 'longitude' in df.columns:
                lat_data = df['latitude'].dropna()
                lon_data = df['longitude'].dropna()
                if len(lat_data) > 0 and len(lon_data) > 0:
                    report['datasets'][name]['spatial_extent'] = {
                        'lat_min': float(lat_data.min()),
                        'lat_max': float(lat_data.max()),
                        'lon_min': float(lon_data.min()),
                        'lon_max': float(lon_data.max())
                    }
        
        # Integration summary
        if len(integrated_df) > 0:
            report['integration'] = {
                'grid_points': len(integrated_df),
                'data_coverage': {},
                'parameter_summary': {}
            }
            
            # Count data coverage
            for col in integrated_df.columns:
                if col.endswith('_count'):
                    param = col.replace('_count', '')
                    coverage = (integrated_df[col] > 0).sum()
                    report['integration']['data_coverage'][param] = {
                        'grid_points_with_data': int(coverage),
                        'coverage_percent': float(coverage / len(integrated_df) * 100)
                    }
        
        # Correlation summary
        if len(correlation_matrix) > 0:
            # Find strongest correlations
            corr_values = correlation_matrix.values
            np.fill_diagonal(corr_values, 0)  # Remove self-correlations
            
            # Get indices of strongest positive and negative correlations
            max_pos_idx = np.unravel_index(np.nanargmax(corr_values), corr_values.shape)
            min_neg_idx = np.unravel_index(np.nanargmin(corr_values), corr_values.shape)
            
            report['correlations'] = {
                'strongest_positive': {
                    'variables': [correlation_matrix.index[max_pos_idx[0]], 
                                correlation_matrix.columns[max_pos_idx[1]]],
                    'correlation': float(corr_values[max_pos_idx])
                },
                'strongest_negative': {
                    'variables': [correlation_matrix.index[min_neg_idx[0]], 
                                correlation_matrix.columns[min_neg_idx[1]]],
                    'correlation': float(corr_values[min_neg_idx])
                }
            }
        
        # Save report
        report_file = os.path.join(self.data_dir, "integration_summary_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Summary report saved to: {report_file}")
        
        return report

def main():
    """Main function for integrated data system"""
    
    # Change to NASA Space Apps directory
    repo_dir = "/Users/a91788/Desktop/NASA/Nasa-Space-Apps"
    os.chdir(repo_dir)
    
    print("Integrated Air Quality and Weather Data System")
    print("="*60)
    
    # Initialize system
    system = IntegratedDataSystem()
    
    # Load all datasets
    print("\nStep 1: Loading all data sources...")
    datasets = system.load_all_datasets(use_sample_data=True)
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return
    
    # Spatial integration
    print("\nStep 2: Spatial integration...")
    integrated_df = system.spatial_join_datasets(datasets, target_resolution=2.0)
    
    # Temporal analysis
    print("\nStep 3: Temporal analysis...")
    temporal_df = system.create_temporal_series(datasets, aggregation_period='daily')
    
    # Correlation analysis
    print("\nStep 4: Correlation analysis...")
    correlation_matrix = pd.DataFrame()
    if len(integrated_df) > 0:
        correlation_matrix = system.create_correlation_analysis(integrated_df)
    
    # Generate summary report
    print("\nStep 5: Summary report...")
    report = system.generate_summary_report(datasets, integrated_df, temporal_df, correlation_matrix)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("INTEGRATED DATA SYSTEM SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, dataset_info in report['datasets'].items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"  Records: {dataset_info['records']:,}")
        if dataset_info['date_range']:
            print(f"  Date Range: {dataset_info['date_range']['start']} to {dataset_info['date_range']['end']}")
        if dataset_info['spatial_extent']:
            extent = dataset_info['spatial_extent']
            print(f"  Spatial Extent: ({extent['lat_min']:.1f}, {extent['lon_min']:.1f}) to ({extent['lat_max']:.1f}, {extent['lon_max']:.1f})")
    
    if report['integration']:
        print(f"\nSPATIAL INTEGRATION:")
        print(f"  Grid Points: {report['integration']['grid_points']:,}")
        
        if report['integration']['data_coverage']:
            print(f"  Data Coverage:")
            for param, coverage in report['integration']['data_coverage'].items():
                print(f"    {param}: {coverage['coverage_percent']:.1f}% of grid points")
    
    if report['correlations']:
        print(f"\nKEY CORRELATIONS:")
        pos_corr = report['correlations']['strongest_positive']
        neg_corr = report['correlations']['strongest_negative']
        print(f"  Strongest Positive: {pos_corr['variables'][0]} ↔ {pos_corr['variables'][1]} (r = {pos_corr['correlation']:.3f})")
        print(f"  Strongest Negative: {neg_corr['variables'][0]} ↔ {neg_corr['variables'][1]} (r = {neg_corr['correlation']:.3f})")
    
    print(f"\n{'='*60}")
    print("Integration complete! Check 'integrated_data' directory for outputs.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
