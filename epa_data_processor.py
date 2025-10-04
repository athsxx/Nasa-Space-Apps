#!/usr/bin/env python3
"""
EPA Air Quality Data Analysis for North America
Processes EPA AQS (Air Quality System) hourly data for various pollutants

This script works with EPA datasets containing:
- Ozone (O3) - Parameter Code 44201
- Carbon Monoxide (CO) - Parameter Code 42101  
- Sulfur Dioxide (SO2) - Parameter Code 42401
- Nitrogen Dioxide (NO2) - Parameter Code 42602
- PM2.5 and PM10 data (additional parameter codes)

Data Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html
"""

import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EPADataProcessor:
    def __init__(self, data_directory="validation datasets"):
        """
        Initialize EPA Data Processor
        
        Args:
            data_directory (str): Directory containing EPA data files
        """
        self.data_dir = data_directory
        self.datasets = {}
        
        # EPA Parameter codes and names
        self.parameter_info = {
            '44201': {'name': 'Ozone', 'units': 'ppm', 'pollutant': 'O3'},
            '42101': {'name': 'Carbon Monoxide', 'units': 'ppm', 'pollutant': 'CO'},
            '42401': {'name': 'Sulfur Dioxide', 'units': 'ppb', 'pollutant': 'SO2'},
            '42602': {'name': 'Nitrogen Dioxide', 'units': 'ppb', 'pollutant': 'NO2'},
            '88101': {'name': 'PM2.5 LC', 'units': 'µg/m³', 'pollutant': 'PM2.5'},
            '81102': {'name': 'PM10', 'units': 'µg/m³', 'pollutant': 'PM10'}
        }
        
        print(f"EPA Data Processor initialized")
        print(f"Data directory: {self.data_dir}")
        
    def extract_and_load_data(self, zip_file_pattern=None):
        """
        Extract ZIP files and load EPA data
        
        Args:
            zip_file_pattern (str): Pattern to match ZIP files (e.g., 'hourly_44201' for ozone)
        
        Returns:
            dict: Dictionary of loaded datasets by parameter code
        """
        zip_files = []
        
        # Find ZIP files in directory
        for file in os.listdir(self.data_dir):
            if file.endswith('.zip'):
                if zip_file_pattern is None or zip_file_pattern in file:
                    zip_files.append(file)
        
        print(f"Found {len(zip_files)} ZIP files to process")
        
        for zip_file in zip_files:
            zip_path = os.path.join(self.data_dir, zip_file)
            
            # Extract parameter code from filename
            # Format: hourly_PARAMCODE_YEAR_POLLUTANT.zip
            parts = zip_file.replace('.zip', '').split('_')
            if len(parts) >= 2:
                param_code = parts[1]
            else:
                continue
                
            print(f"Processing {zip_file} (Parameter: {param_code})...")
            
            try:
                # Extract and read CSV
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Get the CSV file name (should match the ZIP name pattern)
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if csv_files:
                        csv_file = csv_files[0]
                        zip_ref.extract(csv_file, self.data_dir)
                        
                        # Load the CSV
                        csv_path = os.path.join(self.data_dir, csv_file)
                        df = pd.read_csv(csv_path)
                        
                        # Store in datasets
                        self.datasets[param_code] = {
                            'data': df,
                            'info': self.parameter_info.get(param_code, {'name': f'Parameter {param_code}'}),
                            'file': zip_file
                        }
                        
                        print(f"  Loaded {len(df)} records for {self.parameter_info.get(param_code, {}).get('name', param_code)}")
                        
                        # Clean up extracted CSV
                        os.remove(csv_path)
                        
            except Exception as e:
                print(f"  Error processing {zip_file}: {e}")
        
        return self.datasets
    
    def get_data_summary(self):
        """Get summary of all loaded datasets"""
        print(f"\n{'='*60}")
        print(f"EPA AIR QUALITY DATA SUMMARY")
        print(f"{'='*60}")
        
        if not self.datasets:
            print("No datasets loaded. Run extract_and_load_data() first.")
            return
        
        total_records = 0
        
        for param_code, dataset in self.datasets.items():
            df = dataset['data']
            info = dataset['info']
            
            print(f"\n{info.get('name', param_code)} ({info.get('pollutant', param_code)}):")
            print(f"  Parameter Code: {param_code}")
            print(f"  Total Records: {len(df):,}")
            print(f"  Date Range: {df['Date Local'].min()} to {df['Date Local'].max()}")
            print(f"  States Covered: {df['State Name'].nunique()}")
            print(f"  Monitoring Sites: {df[['State Code', 'County Code', 'Site Num']].drop_duplicates().shape[0]}")
            
            # Value statistics
            measurement_col = 'Sample Measurement'
            if measurement_col in df.columns:
                valid_measurements = df[measurement_col].dropna()
                if len(valid_measurements) > 0:
                    print(f"  Measurement Range: {valid_measurements.min():.4f} - {valid_measurements.max():.4f} {info.get('units', '')}")
                    print(f"  Average: {valid_measurements.mean():.4f} {info.get('units', '')}")
            
            total_records += len(df)
        
        print(f"\nTOTAL RECORDS ACROSS ALL POLLUTANTS: {total_records:,}")
        print(f"{'='*60}")
    
    def filter_data_by_region(self, param_code, states=None, cities=None, date_range=None):
        """
        Filter data by geographic region and time period
        
        Args:
            param_code (str): EPA parameter code
            states (list): List of state names to include
            cities (list): List of cities to include
            date_range (tuple): (start_date, end_date) in 'YYYY-MM-DD' format
        
        Returns:
            pandas.DataFrame: Filtered dataset
        """
        if param_code not in self.datasets:
            print(f"Parameter code {param_code} not found in loaded datasets")
            return pd.DataFrame()
        
        df = self.datasets[param_code]['data'].copy()
        
        # Filter by states
        if states:
            df = df[df['State Name'].isin(states)]
            print(f"Filtered to states: {states}")
        
        # Filter by cities  
        if cities:
            df = df[df['County Name'].isin(cities)]
            print(f"Filtered to cities: {cities}")
        
        # Filter by date range
        if date_range:
            start_date, end_date = date_range
            df = df[(df['Date Local'] >= start_date) & (df['Date Local'] <= end_date)]
            print(f"Filtered to date range: {start_date} to {end_date}")
        
        print(f"Filtered dataset contains {len(df)} records")
        return df
    
    def get_north_american_summary(self):
        """Get comprehensive summary of North American air quality data"""
        print(f"\n{'='*60}")
        print(f"NORTH AMERICAN AIR QUALITY SUMMARY")
        print(f"{'='*60}")
        
        for param_code, dataset in self.datasets.items():
            df = dataset['data']
            info = dataset['info']
            pollutant = info.get('pollutant', param_code)
            
            print(f"\n{pollutant} ({info.get('name', '')}):")
            
            # State-level summary
            state_summary = df.groupby('State Name').agg({
                'Sample Measurement': ['count', 'mean', 'max'],
                'Site Num': 'nunique'
            }).round(4)
            
            state_summary.columns = ['Measurements', 'Average', 'Maximum', 'Sites']
            state_summary = state_summary.sort_values('Average', ascending=False)
            
            print("  Top 10 States by Average Concentration:")
            print(state_summary.head(10).to_string())
            
            # Temporal patterns
            df['Date Local'] = pd.to_datetime(df['Date Local'])
            df['Month'] = df['Date Local'].dt.month
            df['Hour'] = pd.to_numeric(df['Time Local'].str.split(':').str[0])
            
            monthly_avg = df.groupby('Month')['Sample Measurement'].mean()
            hourly_avg = df.groupby('Hour')['Sample Measurement'].mean()
            
            print(f"\n  Seasonal Pattern (Monthly Averages):")
            for month, avg in monthly_avg.items():
                month_name = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month]
                print(f"    {month_name}: {avg:.4f} {info.get('units', '')}")
    
    def export_filtered_data(self, param_code, output_file, **filter_kwargs):
        """
        Export filtered data to CSV
        
        Args:
            param_code (str): EPA parameter code
            output_file (str): Output CSV filename
            **filter_kwargs: Arguments for filter_data_by_region()
        """
        filtered_df = self.filter_data_by_region(param_code, **filter_kwargs)
        
        if not filtered_df.empty:
            filtered_df.to_csv(output_file, index=False)
            print(f"Exported {len(filtered_df)} records to {output_file}")
            return True
        else:
            print("No data to export")
            return False
    
    def create_state_summary(self):
        """Create a summary CSV of average pollutant levels by state"""
        summary_data = []
        
        for param_code, dataset in self.datasets.items():
            df = dataset['data']
            info = dataset['info']
            pollutant = info.get('pollutant', param_code)
            
            # Calculate state averages
            state_avg = df.groupby('State Name').agg({
                'Sample Measurement': ['mean', 'count', 'max'],
                'Site Num': 'nunique'
            }).round(6)
            
            state_avg.columns = ['Average', 'Measurements', 'Maximum', 'Sites']
            
            for state in state_avg.index:
                summary_data.append({
                    'State': state,
                    'Pollutant': pollutant,
                    'Parameter_Code': param_code,
                    'Average_Concentration': state_avg.loc[state, 'Average'],
                    'Max_Concentration': state_avg.loc[state, 'Maximum'],
                    'Total_Measurements': state_avg.loc[state, 'Measurements'],
                    'Monitoring_Sites': state_avg.loc[state, 'Sites'],
                    'Units': info.get('units', '')
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('north_america_air_quality_summary.csv', index=False)
        print(f"State summary exported to: north_america_air_quality_summary.csv")
        
        return summary_df

def main():
    """Main function demonstrating EPA data processing for North America"""
    
    # Change to the repository directory
    repo_dir = "/Users/a91788/Desktop/NASA/Nasa-Space-Apps"
    os.chdir(repo_dir)
    
    print("EPA Air Quality Data Analysis - North America")
    print("=" * 50)
    
    # Initialize processor
    processor = EPADataProcessor()
    
    # Load all available datasets
    print("\nStep 1: Loading EPA datasets...")
    datasets = processor.extract_and_load_data()
    
    if not datasets:
        print("No datasets found. Please check that ZIP files are in the 'validation datasets' directory.")
        return
    
    # Get overall summary
    print("\nStep 2: Data Summary...")
    processor.get_data_summary()
    
    # Get North American summary with state breakdowns
    print("\nStep 3: North American Analysis...")
    processor.get_north_american_summary()
    
    # Create state summary file
    print("\nStep 4: Creating State Summary...")
    state_summary = processor.create_state_summary()
    
    # Example: Export California ozone data
    if '44201' in datasets:  # Ozone data
        print("\nStep 5: Example - California Ozone Data...")
        ca_ozone_file = "california_ozone_2024.csv"
        processor.export_filtered_data(
            param_code='44201',
            output_file=ca_ozone_file,
            states=['California']
        )
    
    # Example: Export high-pollution states data
    if '42101' in datasets:  # CO data
        print("\nStep 6: Example - Major Cities CO Data...")
        major_cities_co_file = "major_cities_co_2024.csv"
        # Note: County names in EPA data, not city names
        processor.export_filtered_data(
            param_code='42101',
            output_file=major_cities_co_file,
            states=['California', 'New York', 'Texas', 'Illinois', 'Pennsylvania']
        )
    
    print(f"\n{'='*50}")
    print("Analysis complete! Generated files:")
    print("- north_america_air_quality_summary.csv")
    if '44201' in datasets:
        print(f"- california_ozone_2024.csv")
    if '42101' in datasets:
        print(f"- major_cities_co_2024.csv")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
