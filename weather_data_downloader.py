#!/usr/bin/env python3
"""
Weather Data Downloader for Air Quality Analysis

Downloads meteorological data for:
- Temperature
- Relative Humidity  
- Wind Speed & Direction (converted to u/v components)
- Precipitation
- Solar Radiation / UV Index

Data Sources:
- NOAA National Weather Service API
- OpenWeatherMap API
- ECMWF ERA5 Reanalysis (via Copernicus CDS)
- NASA POWER API
- Weather Underground API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import math
from pathlib import Path

class WeatherDataDownloader:
    def __init__(self, data_dir="weather_data"):
        """
        Initialize Weather Data Downloader
        
        Args:
            data_dir (str): Directory to store downloaded data
        """
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # API endpoints and keys
        self.apis = {
            'noaa': {
                'base_url': 'https://api.weather.gov',
                'requires_key': False
            },
            'openweather': {
                'base_url': 'https://api.openweathermap.org/data/2.5',
                'requires_key': True,
                'key_env': 'OPENWEATHER_API_KEY'
            },
            'nasa_power': {
                'base_url': 'https://power.larc.nasa.gov/api/temporal/daily/point',
                'requires_key': False
            },
            'weatherbit': {
                'base_url': 'https://api.weatherbit.io/v2.0',
                'requires_key': True,
                'key_env': 'WEATHERBIT_API_KEY'
            }
        }
        
        # Weather parameters
        self.weather_params = {
            'temperature': {'units': '°C', 'nasa_param': 'T2M'},
            'humidity': {'units': '%', 'nasa_param': 'RH2M'},
            'wind_speed': {'units': 'm/s', 'nasa_param': 'WS10M'},
            'wind_direction': {'units': 'degrees', 'nasa_param': 'WD10M'},
            'precipitation': {'units': 'mm', 'nasa_param': 'PRECTOTCORR'},
            'solar_radiation': {'units': 'W/m²', 'nasa_param': 'ALLSKY_SFC_SW_DWN'},
            'uv_index': {'units': 'index', 'nasa_param': 'UV_INDEX'}
        }
        
        print(f"Weather Data Downloader initialized")
        print(f"Data directory: {self.data_dir}")
    
    def wind_components(self, speed, direction):
        """
        Convert wind speed and direction to u,v components
        
        Args:
            speed (float): Wind speed in m/s
            direction (float): Wind direction in degrees (meteorological convention)
        
        Returns:
            tuple: (u, v) components in m/s
        """
        # Convert meteorological direction to mathematical angle
        # Meteorological: 0° = North, 90° = East
        # Mathematical: 0° = East, 90° = North
        math_angle = (270 - direction) % 360
        
        # Convert to radians
        angle_rad = math.radians(math_angle)
        
        # Calculate components
        u = speed * math.cos(angle_rad)  # Eastward component
        v = speed * math.sin(angle_rad)  # Northward component
        
        return u, v
    
    def download_nasa_power_data(self, lat, lon, start_date, end_date, parameters=None):
        """
        Download weather data from NASA POWER API
        
        Args:
            lat (float): Latitude in degrees
            lon (float): Longitude in degrees
            start_date (str): 'YYYYMMDD' format
            end_date (str): 'YYYYMMDD' format
            parameters (list): List of parameters to download
        
        Returns:
            pandas.DataFrame: Weather data
        """
        if parameters is None:
            parameters = ['T2M', 'RH2M', 'WS10M', 'WD10M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN']
        
        # Build NASA POWER API URL
        params_str = ','.join(parameters)
        url = f"{self.apis['nasa_power']['base_url']}"
        
        params = {
            'parameters': params_str,
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': start_date,
            'end': end_date,
            'format': 'json'
        }
        
        print(f"Downloading NASA POWER data for ({lat:.2f}, {lon:.2f})...")
        print(f"  Parameters: {params_str}")
        print(f"  Date range: {start_date} - {end_date}")
        
        try:
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse the NASA POWER response format
                if 'properties' in data and 'parameter' in data['properties']:
                    param_data = data['properties']['parameter']
                    
                    # Convert to DataFrame
                    df_data = []
                    
                    # Get dates from one of the parameters
                    first_param = list(param_data.keys())[0]
                    dates = list(param_data[first_param].keys())
                    
                    for date_str in dates:
                        row = {
                            'date': datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d'),
                            'latitude': lat,
                            'longitude': lon
                        }
                        
                        # Add all parameters
                        for param in parameters:
                            if param in param_data and date_str in param_data[param]:
                                value = param_data[param][date_str]
                                # Handle missing values
                                if value != -999.0:
                                    row[param] = value
                                else:
                                    row[param] = np.nan
                        
                        # Calculate wind components if wind data available
                        if 'WS10M' in row and 'WD10M' in row and not pd.isna(row['WS10M']) and not pd.isna(row['WD10M']):
                            u, v = self.wind_components(row['WS10M'], row['WD10M'])
                            row['wind_u'] = u
                            row['wind_v'] = v
                        
                        df_data.append(row)
                    
                    df = pd.DataFrame(df_data)
                    
                    # Save to file
                    output_file = os.path.join(self.data_dir, f"nasa_power_{lat}_{lon}_{start_date}_{end_date}.csv")
                    df.to_csv(output_file, index=False)
                    
                    print(f"  Downloaded {len(df)} records")
                    print(f"  Saved to: {output_file}")
                    
                    return df
                
            else:
                print(f"  Error: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error downloading NASA POWER data: {e}")
            return None
    
    def download_openweather_data(self, lat, lon, api_key=None, days_back=5):
        """
        Download current and historical weather data from OpenWeatherMap
        
        Args:
            lat (float): Latitude in degrees
            lon (float): Longitude in degrees  
            api_key (str): OpenWeatherMap API key
            days_back (int): Number of days of historical data to fetch
        
        Returns:
            pandas.DataFrame: Weather data
        """
        if not api_key:
            api_key = os.getenv('OPENWEATHER_API_KEY')
            if not api_key:
                print("OpenWeatherMap API key not found. Set OPENWEATHER_API_KEY environment variable.")
                return None
        
        base_url = self.apis['openweather']['base_url']
        all_data = []
        
        print(f"Downloading OpenWeatherMap data for ({lat:.2f}, {lon:.2f})...")
        
        # Get current weather
        try:
            current_url = f"{base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric'
            }
            
            response = requests.get(current_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                current_record = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'latitude': lat,
                    'longitude': lon,
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', np.nan),
                    'wind_direction': data.get('wind', {}).get('deg', np.nan),
                    'precipitation': data.get('rain', {}).get('1h', 0) + data.get('snow', {}).get('1h', 0),
                    'uv_index': np.nan,  # Not available in current weather
                    'source': 'OpenWeatherMap Current'
                }
                
                # Add wind components
                if not pd.isna(current_record['wind_speed']) and not pd.isna(current_record['wind_direction']):
                    u, v = self.wind_components(current_record['wind_speed'], current_record['wind_direction'])
                    current_record['wind_u'] = u
                    current_record['wind_v'] = v
                
                all_data.append(current_record)
                print(f"  Current weather: {current_record['temperature']}°C, {current_record['humidity']}% RH")
            
        except Exception as e:
            print(f"  Error getting current weather: {e}")
        
        # Get historical data (requires separate API calls for each day)
        for days_ago in range(1, days_back + 1):
            try:
                # Historical weather endpoint
                target_date = datetime.now() - timedelta(days=days_ago)
                timestamp = int(target_date.timestamp())
                
                hist_url = f"{base_url}/onecall/timemachine"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'dt': timestamp,
                    'appid': api_key,
                    'units': 'metric'
                }
                
                response = requests.get(hist_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'current' in data:
                        current = data['current']
                        
                        hist_record = {
                            'date': target_date.strftime('%Y-%m-%d'),
                            'latitude': lat,
                            'longitude': lon,
                            'temperature': current['temp'],
                            'humidity': current['humidity'],
                            'pressure': current['pressure'],
                            'wind_speed': current.get('wind_speed', np.nan),
                            'wind_direction': current.get('wind_deg', np.nan),
                            'precipitation': sum([item.get('rain', {}).get('1h', 0) + item.get('snow', {}).get('1h', 0) 
                                                 for item in data.get('hourly', [])]),
                            'uv_index': current.get('uvi', np.nan),
                            'source': 'OpenWeatherMap Historical'
                        }
                        
                        # Add wind components
                        if not pd.isna(hist_record['wind_speed']) and not pd.isna(hist_record['wind_direction']):
                            u, v = self.wind_components(hist_record['wind_speed'], hist_record['wind_direction'])
                            hist_record['wind_u'] = u
                            hist_record['wind_v'] = v
                        
                        all_data.append(hist_record)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  Error getting historical data for {days_ago} days ago: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Save to file
            output_file = os.path.join(self.data_dir, f"openweather_{lat}_{lon}_{datetime.now().strftime('%Y%m%d')}.csv")
            df.to_csv(output_file, index=False)
            
            print(f"  Downloaded {len(df)} records total")
            print(f"  Saved to: {output_file}")
            
            return df
        
        return None
    
    def download_noaa_data(self, station_id, start_date, end_date):
        """
        Download weather data from NOAA API
        
        Args:
            station_id (str): NOAA weather station ID
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
        
        Returns:
            pandas.DataFrame: Weather data
        """
        base_url = self.apis['noaa']['base_url']
        
        print(f"Downloading NOAA data for station {station_id}...")
        print(f"  Date range: {start_date} to {end_date}")
        
        # First, get station metadata
        try:
            station_url = f"{base_url}/stations/{station_id}"
            response = requests.get(station_url, timeout=30)
            
            if response.status_code == 200:
                station_data = response.json()
                lat = station_data['geometry']['coordinates'][1]
                lon = station_data['geometry']['coordinates'][0]
                station_name = station_data['properties']['name']
                
                print(f"  Station: {station_name} ({lat:.2f}, {lon:.2f})")
            else:
                print(f"  Could not get station metadata: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error getting station metadata: {e}")
            return None
        
        # Get observations
        try:
            obs_url = f"{base_url}/stations/{station_id}/observations"
            params = {
                'start': f"{start_date}T00:00:00Z",
                'end': f"{end_date}T23:59:59Z"
            }
            
            response = requests.get(obs_url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'features' in data:
                    observations = data['features']
                    
                    records = []
                    for obs in observations:
                        props = obs['properties']
                        
                        # Extract timestamp
                        timestamp = props.get('timestamp', '')
                        if timestamp:
                            date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            continue
                        
                        record = {
                            'datetime': date,
                            'date': date.split(' ')[0],
                            'latitude': lat,
                            'longitude': lon,
                            'station_id': station_id,
                            'station_name': station_name
                        }
                        
                        # Extract meteorological data
                        if props.get('temperature', {}).get('value'):
                            record['temperature'] = props['temperature']['value']
                        
                        if props.get('relativeHumidity', {}).get('value'):
                            record['humidity'] = props['relativeHumidity']['value']
                        
                        if props.get('windSpeed', {}).get('value'):
                            record['wind_speed'] = props['windSpeed']['value']
                        
                        if props.get('windDirection', {}).get('value'):
                            record['wind_direction'] = props['windDirection']['value']
                        
                        if props.get('precipitationLastHour', {}).get('value'):
                            record['precipitation'] = props['precipitationLastHour']['value']
                        
                        # Calculate wind components
                        if 'wind_speed' in record and 'wind_direction' in record:
                            if not pd.isna(record['wind_speed']) and not pd.isna(record['wind_direction']):
                                u, v = self.wind_components(record['wind_speed'], record['wind_direction'])
                                record['wind_u'] = u
                                record['wind_v'] = v
                        
                        record['source'] = 'NOAA NWS'
                        records.append(record)
                    
                    if records:
                        df = pd.DataFrame(records)
                        
                        # Save to file
                        output_file = os.path.join(self.data_dir, f"noaa_{station_id}_{start_date}_{end_date}.csv")
                        df.to_csv(output_file, index=False)
                        
                        print(f"  Downloaded {len(df)} observations")
                        print(f"  Saved to: {output_file}")
                        
                        return df
                
        except Exception as e:
            print(f"  Error downloading NOAA observations: {e}")
        
        return None
    
    def create_sample_weather_data(self, bbox, start_date, end_date, num_locations=20):
        """
        Create sample weather data for testing/demonstration
        
        Args:
            bbox (tuple): (west, south, east, north) in degrees
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
            num_locations (int): Number of sample locations
        
        Returns:
            pandas.DataFrame: Sample weather data
        """
        west, south, east, north = bbox
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        print(f"Creating sample weather data...")
        print(f"  Locations: {num_locations}")
        print(f"  Date range: {start_date} to {end_date}")
        
        np.random.seed(42)  # For reproducible results
        
        data = []
        
        # Generate random locations within bbox
        locations = []
        for i in range(num_locations):
            lat = np.random.uniform(south, north)
            lon = np.random.uniform(west, east)
            locations.append((lat, lon))
        
        # Generate data for each date and location
        current_dt = start_dt
        while current_dt <= end_dt:
            for i, (lat, lon) in enumerate(locations):
                # Seasonal temperature variation (simplified)
                base_temp = 15 + 15 * math.cos((current_dt.timetuple().tm_yday - 172) * 2 * math.pi / 365)
                # Add latitude effect (cooler toward poles)
                temp_adjustment = -(abs(lat) - 30) * 0.5
                # Add random variation
                temperature = base_temp + temp_adjustment + np.random.normal(0, 5)
                
                # Humidity (inversely related to temperature with noise)
                humidity = max(10, min(100, 80 - temperature * 0.8 + np.random.normal(0, 15)))
                
                # Wind (random but realistic)
                wind_speed = np.random.exponential(3)  # Exponential distribution
                wind_direction = np.random.uniform(0, 360)
                
                # Precipitation (random, higher in some seasons)
                precip_prob = 0.3 + 0.2 * math.sin((current_dt.timetuple().tm_yday - 80) * 2 * math.pi / 365)
                precipitation = np.random.exponential(2) if np.random.random() < precip_prob else 0
                
                # Solar radiation (seasonal and latitude dependent)
                solar_base = 200 + 300 * math.cos((current_dt.timetuple().tm_yday - 172) * 2 * math.pi / 365)
                solar_lat_effect = math.cos(math.radians(abs(lat))) * 0.8 + 0.2
                solar_radiation = max(0, solar_base * solar_lat_effect + np.random.normal(0, 50))
                
                # UV Index (related to solar radiation)
                uv_index = max(0, min(12, solar_radiation / 80 + np.random.normal(0, 1)))
                
                # Wind components
                u, v = self.wind_components(wind_speed, wind_direction)
                
                record = {
                    'date': current_dt.strftime('%Y-%m-%d'),
                    'latitude': lat,
                    'longitude': lon,
                    'location_id': f'LOC_{i:03d}',
                    'temperature': round(temperature, 2),
                    'humidity': round(humidity, 1),
                    'wind_speed': round(wind_speed, 2),
                    'wind_direction': round(wind_direction, 1),
                    'wind_u': round(u, 2),
                    'wind_v': round(v, 2),
                    'precipitation': round(precipitation, 2),
                    'solar_radiation': round(solar_radiation, 1),
                    'uv_index': round(uv_index, 1),
                    'source': 'Sample Data'
                }
                
                data.append(record)
            
            current_dt += timedelta(days=1)
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_file = os.path.join(self.data_dir, f"sample_weather_{start_date}_{end_date}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"  Created {len(df)} weather records")
        print(f"  Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")
        print(f"  Wind speed range: {df['wind_speed'].min():.1f} - {df['wind_speed'].max():.1f} m/s")
        print(f"  Saved to: {output_file}")
        
        return df
    
    def download_all_weather_data(self, bbox, start_date, end_date, use_sample_data=False):
        """
        Download comprehensive weather data from multiple sources
        
        Args:
            bbox (tuple): (west, south, east, north) in degrees
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
            use_sample_data (bool): Whether to use sample data instead of real downloads
        
        Returns:
            dict: Summary of downloaded weather data
        """
        print(f"Downloading weather data for North America")
        print(f"Bounding box: {bbox}")
        print(f"Date range: {start_date} to {end_date}")
        print("="*60)
        
        results = {}
        
        if use_sample_data:
            # Create comprehensive sample weather data
            df = self.create_sample_weather_data(bbox, start_date, end_date, num_locations=50)
            results['sample_weather'] = {
                'status': 'sample_created',
                'records': len(df),
                'file': os.path.join(self.data_dir, f"sample_weather_{start_date}_{end_date}.csv"),
                'parameters': ['temperature', 'humidity', 'wind_speed', 'wind_direction', 
                              'wind_u', 'wind_v', 'precipitation', 'solar_radiation', 'uv_index']
            }
            
            return results
        
        # Try NASA POWER API for several key locations
        key_locations = [
            (40.7128, -74.0060, "New York"),
            (34.0522, -118.2437, "Los Angeles"), 
            (41.8781, -87.6298, "Chicago"),
            (29.7604, -95.3698, "Houston"),
            (43.6532, -79.3832, "Toronto"),
            (49.2827, -123.1207, "Vancouver")
        ]
        
        nasa_start = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
        nasa_end = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
        
        nasa_results = []
        for lat, lon, city in key_locations:
            print(f"\nTrying NASA POWER for {city}...")
            df = self.download_nasa_power_data(lat, lon, nasa_start, nasa_end)
            if df is not None:
                nasa_results.append({
                    'city': city,
                    'records': len(df),
                    'lat': lat,
                    'lon': lon
                })
        
        if nasa_results:
            results['nasa_power'] = {
                'status': 'downloaded',
                'locations': nasa_results,
                'total_records': sum([r['records'] for r in nasa_results])
            }
        
        # Try OpenWeatherMap if API key available
        owm_key = os.getenv('OPENWEATHER_API_KEY')
        if owm_key:
            print(f"\nTrying OpenWeatherMap...")
            # Test with one location
            df = self.download_openweather_data(40.7128, -74.0060, owm_key)
            if df is not None:
                results['openweathermap'] = {
                    'status': 'downloaded',
                    'records': len(df),
                    'location': 'New York (sample)'
                }
        else:
            results['openweathermap'] = {
                'status': 'api_key_required',
                'note': 'Set OPENWEATHER_API_KEY environment variable'
            }
        
        # NOAA stations (sample stations for testing)
        noaa_stations = ['KNYC', 'KLAX', 'KORD', 'KIAH']  # Major airport weather stations
        
        print(f"\nTrying NOAA stations...")
        noaa_results = []
        for station in noaa_stations:
            df = self.download_noaa_data(station, start_date, end_date)
            if df is not None:
                noaa_results.append({
                    'station': station,
                    'records': len(df)
                })
        
        if noaa_results:
            results['noaa'] = {
                'status': 'downloaded',
                'stations': noaa_results,
                'total_records': sum([r['records'] for r in noaa_results])
            }
        
        return results

def main():
    """Main function for weather data download"""
    
    # Change to NASA Space Apps directory
    repo_dir = "/Users/a91788/Desktop/NASA/Nasa-Space-Apps"
    os.chdir(repo_dir)
    
    print("Weather Data Downloader for Air Quality Analysis")
    print("="*60)
    
    # Initialize downloader
    downloader = WeatherDataDownloader()
    
    # Define North American bounding box
    north_america_bbox = (-180, 10, -50, 85)
    
    # Set date range (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"\nTarget region: North America {north_america_bbox}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Download weather data
    print(f"\nStarting weather data download...")
    
    # Use sample data for demonstration (set to False for real downloads)
    use_sample = True
    
    if use_sample:
        print("Note: Using sample data for demonstration")
        print("Set use_sample=False in code for real downloads")
    
    results = downloader.download_all_weather_data(
        bbox=north_america_bbox,
        start_date=start_date,
        end_date=end_date,
        use_sample_data=use_sample
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("WEATHER DATA DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    total_records = 0
    
    for source, result in results.items():
        print(f"\n{source.upper().replace('_', ' ')}:")
        print(f"  Status: {result['status']}")
        
        if 'records' in result:
            print(f"  Records: {result['records']:,}")
            total_records += result['records']
        
        if 'total_records' in result:
            print(f"  Total Records: {result['total_records']:,}")
            total_records += result['total_records']
        
        if 'file' in result:
            print(f"  File: {result['file']}")
        
        if 'parameters' in result:
            print(f"  Parameters: {', '.join(result['parameters'])}")
        
        if 'locations' in result:
            print(f"  Locations: {len(result['locations'])}")
        
        if 'note' in result:
            print(f"  Note: {result['note']}")
    
    print(f"\nTOTAL WEATHER RECORDS: {total_records:,}")
    print(f"\n{'='*60}")
    print("Weather data download completed!")
    print("Check the 'weather_data' directory for output files.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
