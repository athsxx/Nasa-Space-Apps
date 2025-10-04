#!/usr/bin/env python3
"""
Satellite Data Downloader for Air Quality Analysis

Downloads satellite data for:
- NO₂ Column (TROPOMI/OMI)
- O₃ Column (TROPOMI/OMI) 
- AOD (MODIS/VIIRS)

Data Sources:
- NASA Giovanni (https://giovanni.gsfc.nasa.gov/)
- NASA Earthdata (https://earthdata.nasa.gov/)
- Google Earth Engine (requires authentication)
- Copernicus Atmosphere Monitoring Service (CAMS)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import zipfile
from pathlib import Path

class SatelliteDataDownloader:
    def __init__(self, data_dir="satellite_data"):
        """
        Initialize Satellite Data Downloader
        
        Args:
            data_dir (str): Directory to store downloaded data
        """
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # NASA Giovanni API endpoints
        self.giovanni_base = "https://giovanni.gsfc.nasa.gov/giovanni/daac-bin"
        
        # Common satellite data parameters
        self.satellite_params = {
            'no2': {
                'name': 'Nitrogen Dioxide Column',
                'giovanni_id': 'OMNO2d_003_ColumnAmountNO2TropCloudScreened',
                'tropomi_id': 'S5P_L3__NO2____HiR',
                'units': 'molecules/cm²'
            },
            'o3': {
                'name': 'Ozone Column',
                'giovanni_id': 'OMO3PR_003_ColumnAmountO3',
                'tropomi_id': 'S5P_L3__O3_____HiR', 
                'units': 'Dobson Units'
            },
            'aod': {
                'name': 'Aerosol Optical Depth',
                'giovanni_id': 'MOD08_D3_6_1_Aerosol_Optical_Depth_Land_Ocean_Mean_Mean',
                'modis_id': 'MOD04_L2',
                'units': 'dimensionless'
            }
        }
        
        print(f"Satellite Data Downloader initialized")
        print(f"Data directory: {self.data_dir}")
    
    def download_giovanni_data(self, parameter, bbox, start_date, end_date, resolution='0.25'):
        """
        Download data from NASA Giovanni service
        
        Args:
            parameter (str): 'no2', 'o3', or 'aod'
            bbox (tuple): (west, south, east, north) in degrees
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD' 
            resolution (str): Grid resolution in degrees
        
        Returns:
            dict: Downloaded data information
        """
        if parameter not in self.satellite_params:
            print(f"Parameter {parameter} not supported")
            return None
        
        param_info = self.satellite_params[parameter]
        west, south, east, north = bbox
        
        # Giovanni API request parameters
        giovanni_params = {
            'service': 'ArAvTs',  # Area-averaged time series
            'version': '1.02',
            'data': param_info['giovanni_id'],
            'starttime': f"{start_date}T00:00:00Z",
            'endtime': f"{end_date}T23:59:59Z",
            'bbox': f"{west},{south},{east},{north}",
            'format': 'json'
        }
        
        print(f"Requesting {param_info['name']} data from Giovanni...")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Bounding box: {bbox}")
        
        try:
            response = requests.get(self.giovanni_base, params=giovanni_params, timeout=120)
            
            if response.status_code == 200:
                # Parse the response (Giovanni returns complex format)
                output_file = os.path.join(self.data_dir, f"giovanni_{parameter}_{start_date}_{end_date}.json")
                
                with open(output_file, 'w') as f:
                    json.dump(response.json(), f, indent=2)
                
                print(f"  Downloaded to: {output_file}")
                return {
                    'parameter': parameter,
                    'source': 'NASA Giovanni',
                    'file': output_file,
                    'bbox': bbox,
                    'date_range': (start_date, end_date)
                }
            else:
                print(f"  Error: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error downloading from Giovanni: {e}")
            return None
    
    def download_earthdata_urls(self, parameter, bbox, start_date, end_date):
        """
        Generate URLs for NASA Earthdata downloads
        Note: Requires authentication for actual download
        
        Args:
            parameter (str): 'no2', 'o3', or 'aod'
            bbox (tuple): (west, south, east, north) in degrees
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
        
        Returns:
            list: List of download URLs
        """
        if parameter not in self.satellite_params:
            return []
        
        param_info = self.satellite_params[parameter]
        
        # Generate approximate URLs (actual URLs require CMR search)
        urls = []
        
        # TROPOMI S5P data URLs (example format)
        if parameter in ['no2', 'o3']:
            base_url = "https://s5phub.copernicus.eu/dhus/search"
            tropomi_id = param_info['tropomi_id']
            
            # Convert dates to datetime for iteration
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime('%Y%m%d')
                url = f"https://s5phub.copernicus.eu/dhus/odata/v1/Products?$filter=contains(Name,'{tropomi_id}') and contains(Name,'{date_str}')"
                urls.append(url)
                current_dt += timedelta(days=1)
        
        # MODIS AOD data URLs
        elif parameter == 'aod':
            modis_id = param_info['modis_id']
            base_url = "https://ladsweb.modaps.eosdis.nasa.gov/search/order"
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            current_dt = start_dt
            while current_dt <= end_dt:
                year = current_dt.strftime('%Y')
                doy = current_dt.strftime('%j')  # Day of year
                url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/{modis_id}/{year}/{doy}/"
                urls.append(url)
                current_dt += timedelta(days=1)
        
        # Save URLs to file
        urls_file = os.path.join(self.data_dir, f"earthdata_urls_{parameter}_{start_date}_{end_date}.txt")
        with open(urls_file, 'w') as f:
            for url in urls:
                f.write(f"{url}\n")
        
        print(f"Generated {len(urls)} URLs for {param_info['name']}")
        print(f"URLs saved to: {urls_file}")
        
        return urls
    
    def download_copernicus_ads_data(self, parameter, bbox, start_date, end_date):
        """
        Download data from Copernicus Atmosphere Data Store (ADS)
        Note: Requires cdsapi authentication
        
        Args:
            parameter (str): 'no2', 'o3', or 'aod'
            bbox (tuple): (west, south, east, north) in degrees  
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
        
        Returns:
            dict: Download information
        """
        try:
            import cdsapi
        except ImportError:
            print("cdsapi not installed. Install with: pip install cdsapi")
            print("Also requires authentication setup: https://cds.climate.copernicus.eu/api-how-to")
            return None
        
        c = cdsapi.Client()
        
        # Map parameters to ADS dataset names
        ads_datasets = {
            'no2': 'cams-global-reanalysis-eac4',
            'o3': 'cams-global-reanalysis-eac4', 
            'aod': 'cams-global-reanalysis-eac4'
        }
        
        # Map parameters to ADS variable names
        ads_variables = {
            'no2': 'nitrogen_dioxide',
            'o3': 'ozone',
            'aod': 'total_aerosol_optical_depth_469nm'
        }
        
        if parameter not in ads_datasets:
            print(f"Parameter {parameter} not available from ADS")
            return None
        
        dataset = ads_datasets[parameter]
        variable = ads_variables[parameter]
        west, south, east, north = bbox
        
        print(f"Requesting {parameter} data from Copernicus ADS...")
        
        # Build request
        request = {
            'format': 'netcdf',
            'variable': variable,
            'date': f"{start_date}/{end_date}",
            'time': [
                '00:00', '03:00', '06:00', '09:00',
                '12:00', '15:00', '18:00', '21:00'
            ],
            'area': [north, west, south, east],  # North, West, South, East
        }
        
        # Add level for 3D variables
        if parameter in ['no2', 'o3']:
            request['pressure_level'] = '1000'  # Surface level
        
        output_file = os.path.join(self.data_dir, f"copernicus_{parameter}_{start_date}_{end_date}.nc")
        
        try:
            c.retrieve(dataset, request, output_file)
            print(f"  Downloaded to: {output_file}")
            
            return {
                'parameter': parameter,
                'source': 'Copernicus ADS',
                'file': output_file,
                'bbox': bbox,
                'date_range': (start_date, end_date)
            }
            
        except Exception as e:
            print(f"  Error downloading from ADS: {e}")
            return None
    
    def create_sample_data(self, parameter, bbox, start_date, end_date, num_points=100):
        """
        Create sample satellite data for testing/demonstration
        
        Args:
            parameter (str): 'no2', 'o3', or 'aod'
            bbox (tuple): (west, south, east, north) in degrees
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
            num_points (int): Number of sample points to generate
        
        Returns:
            pandas.DataFrame: Sample satellite data
        """
        west, south, east, north = bbox
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate sample data
        np.random.seed(42)  # For reproducible results
        
        data = []
        for i in range(num_points):
            # Random location within bbox
            lon = np.random.uniform(west, east)
            lat = np.random.uniform(south, north)
            
            # Random date within range
            days_diff = (end_dt - start_dt).days
            random_days = np.random.randint(0, days_diff + 1)
            date = start_dt + timedelta(days=random_days)
            
            # Generate realistic values based on parameter
            if parameter == 'no2':
                # NO2 column density (molecules/cm²)
                value = np.random.lognormal(15.5, 0.5)  # Typical range: 1e15 - 1e16
            elif parameter == 'o3':
                # Ozone column (Dobson Units)
                value = np.random.normal(300, 50)  # Typical range: 200-400 DU
            elif parameter == 'aod':
                # Aerosol Optical Depth
                value = np.random.exponential(0.2)  # Typical range: 0-2
            else:
                value = np.random.normal(0, 1)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'latitude': lat,
                'longitude': lon,
                'parameter': parameter,
                'value': max(0, value),  # Ensure positive values
                'units': self.satellite_params[parameter]['units'],
                'source': 'Sample Data'
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_file = os.path.join(self.data_dir, f"sample_{parameter}_{start_date}_{end_date}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"Created sample {parameter} data: {len(df)} points")
        print(f"  Saved to: {output_file}")
        print(f"  Value range: {df['value'].min():.3f} - {df['value'].max():.3f} {self.satellite_params[parameter]['units']}")
        
        return df
    
    def download_all_satellite_data(self, bbox, start_date, end_date, use_sample_data=False):
        """
        Download all satellite parameters for a given region and time period
        
        Args:
            bbox (tuple): (west, south, east, north) in degrees
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
            use_sample_data (bool): Whether to use sample data instead of real downloads
        
        Returns:
            dict: Summary of downloaded data
        """
        print(f"Downloading satellite data for North America")
        print(f"Bounding box: {bbox}")
        print(f"Date range: {start_date} to {end_date}")
        print("="*60)
        
        results = {}
        
        for param in ['no2', 'o3', 'aod']:
            print(f"\nProcessing {param.upper()}...")
            
            if use_sample_data:
                # Create sample data
                df = self.create_sample_data(param, bbox, start_date, end_date)
                results[param] = {
                    'status': 'sample_created',
                    'records': len(df),
                    'file': os.path.join(self.data_dir, f"sample_{param}_{start_date}_{end_date}.csv")
                }
            else:
                # Try different data sources
                download_result = None
                
                # Try Giovanni first
                download_result = self.download_giovanni_data(param, bbox, start_date, end_date)
                
                if not download_result:
                    # Try Copernicus ADS
                    download_result = self.download_copernicus_ads_data(param, bbox, start_date, end_date)
                
                if not download_result:
                    # Generate URLs for manual download
                    urls = self.download_earthdata_urls(param, bbox, start_date, end_date)
                    results[param] = {
                        'status': 'urls_generated',
                        'urls': len(urls),
                        'requires_manual_download': True
                    }
                else:
                    results[param] = download_result
                    results[param]['status'] = 'downloaded'
        
        return results

def main():
    """Main function for satellite data download"""
    
    # Change to NASA Space Apps directory
    repo_dir = "/Users/a91788/Desktop/NASA/Nasa-Space-Apps"
    os.chdir(repo_dir)
    
    print("Satellite Data Downloader for Air Quality Analysis")
    print("="*60)
    
    # Initialize downloader
    downloader = SatelliteDataDownloader()
    
    # Define North American bounding box
    # (west, south, east, north) - covers USA, Canada, Mexico
    north_america_bbox = (-180, 10, -50, 85)
    
    # Set date range (last 30 days as example)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"\nTarget region: North America {north_america_bbox}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Download satellite data
    print(f"\nStarting satellite data download...")
    
    # Use sample data for demonstration (set to False for real downloads)
    use_sample = True
    
    if use_sample:
        print("Note: Using sample data for demonstration")
        print("Set use_sample=False in code for real downloads")
    
    results = downloader.download_all_satellite_data(
        bbox=north_america_bbox,
        start_date=start_date, 
        end_date=end_date,
        use_sample_data=use_sample
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SATELLITE DATA DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    for param, result in results.items():
        param_info = downloader.satellite_params[param]
        print(f"\n{param.upper()} - {param_info['name']}:")
        print(f"  Status: {result['status']}")
        
        if 'records' in result:
            print(f"  Records: {result['records']}")
        if 'file' in result:
            print(f"  File: {result['file']}")
        if 'urls' in result:
            print(f"  URLs generated: {result['urls']}")
        if result.get('requires_manual_download'):
            print(f"  Note: Requires manual download or API authentication")
    
    print(f"\n{'='*60}")
    print("Satellite data download completed!")
    print("Check the 'satellite_data' directory for output files.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
