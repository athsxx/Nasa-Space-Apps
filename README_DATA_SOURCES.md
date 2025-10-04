# Multi-Source Air Quality Data Collection System

A comprehensive system for collecting, processing, and integrating air quality data from multiple sources including EPA ground measurements, satellite observations, and meteorological data.

## 🌟 Overview

This system provides:

### 📡 **Satellite Data**
- **NO₂ Column Density** from TROPOMI/OMI satellites
- **O₃ Column Density** from TROPOMI/OMI satellites  
- **AOD (Aerosol Optical Depth)** from MODIS/VIIRS satellites

### 🌤️ **Weather Data**
- **Temperature** (°C)
- **Relative Humidity** (%)
- **Wind Speed & Direction** (converted to u/v components)
- **Precipitation** (mm)
- **Solar Radiation** (W/m²)
- **UV Index**

### 🏭 **Ground-Based Air Quality**
- **EPA AQS Data** (SO₂, O₃, NO₂, CO)
- **17.3 million measurements** from 2,331 monitoring sites
- **Full year 2024** coverage across North America

## 🗂️ File Structure

```
NASA-Space-Apps/
├── 📊 EPA Data Processing
│   ├── epa_data_processor.py          # EPA AQS data analysis
│   ├── visualize_air_quality.py       # EPA data visualization
│   └── validation datasets/           # EPA ZIP files
│
├── 🛰️ Satellite Data Collection  
│   ├── satellite_data_downloader.py   # Multi-source satellite data
│   └── satellite_data/               # Downloaded satellite files
│
├── 🌤️ Weather Data Collection
│   ├── weather_data_downloader.py     # Multi-source weather data  
│   └── weather_data/                 # Downloaded weather files
│
├── 🔗 Data Integration
│   ├── integrated_data_system.py      # Unified data integration
│   └── integrated_data/              # Combined datasets
│
└── 📋 Configuration
    ├── requirements.txt               # Python dependencies
    └── README_DATA_SOURCES.md         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Individual Systems

**EPA Air Quality Analysis:**
```bash
python epa_data_processor.py          # Process EPA data
python visualize_air_quality.py       # Create visualizations
```

**Satellite Data Collection:**
```bash
python satellite_data_downloader.py   # Download satellite data
```

**Weather Data Collection:**
```bash
python weather_data_downloader.py     # Download weather data
```

**Integrated Analysis:**
```bash
python integrated_data_system.py      # Combine all data sources
```

## 📡 Satellite Data Sources

### 🎯 **Primary Sources:**

1. **NASA Giovanni** (https://giovanni.gsfc.nasa.gov/)
   - NO₂: OMI/TROPOMI Level 3 products
   - O₃: OMI/TROPOMI ozone columns
   - AOD: MODIS Terra/Aqua products
   - **Advantage**: Free, no registration required
   - **Coverage**: Global, daily

2. **Copernicus Atmosphere Data Store** (https://ads.atmosphere.copernicus.eu/)
   - CAMS global reanalysis products
   - Near real-time atmospheric composition
   - **Advantage**: High quality, research-grade
   - **Coverage**: Global, hourly/daily
   - **Requirement**: Free registration + API key

3. **NASA Earthdata** (https://earthdata.nasa.gov/)
   - TROPOMI S5P Level 2/3 products
   - MODIS/VIIRS aerosol products
   - **Advantage**: Highest quality, latest algorithms
   - **Coverage**: Global, daily
   - **Requirement**: NASA Earthdata login

### 📊 **Data Parameters:**

| Parameter | Satellite | Product ID | Units | Resolution |
|-----------|-----------|------------|-------|------------|
| NO₂ Column | TROPOMI | S5P_L3__NO2____HiR | molecules/cm² | 1km × 1km |
| O₃ Column | TROPOMI | S5P_L3__O3_____HiR | Dobson Units | 1km × 1km |
| AOD | MODIS | MOD04_L2 | dimensionless | 1km × 1km |

## 🌤️ Weather Data Sources

### 🎯 **Primary Sources:**

1. **NASA POWER** (https://power.larc.nasa.gov/)
   - Global meteorology from satellites/reanalysis
   - **Parameters**: Temperature, humidity, wind, precipitation, solar radiation
   - **Advantage**: Free, no API key required
   - **Coverage**: Global, daily since 1981
   - **Resolution**: 0.5° × 0.625°

2. **NOAA National Weather Service** (https://api.weather.gov/)
   - Real-time observations from weather stations
   - **Parameters**: Temperature, humidity, wind, precipitation
   - **Advantage**: High temporal resolution, quality controlled
   - **Coverage**: United States, hourly
   - **Requirement**: None

3. **OpenWeatherMap** (https://openweathermap.org/api)
   - Current and historical weather data
   - **Parameters**: All standard meteorological variables
   - **Advantage**: Easy API, good coverage
   - **Coverage**: Global, hourly
   - **Requirement**: API key (free tier available)

### 📊 **Weather Parameters:**

| Parameter | Units | NASA Code | Description |
|-----------|-------|-----------|-------------|
| Temperature | °C | T2M | 2-meter air temperature |
| Humidity | % | RH2M | 2-meter relative humidity |
| Wind Speed | m/s | WS10M | 10-meter wind speed |
| Wind Direction | degrees | WD10M | 10-meter wind direction |
| Precipitation | mm/day | PRECTOTCORR | Total precipitation |
| Solar Radiation | W/m² | ALLSKY_SFC_SW_DWN | Downward shortwave flux |
| UV Index | index | UV_INDEX | UV index at noon |

## 🔗 Data Integration Features

### 🗺️ **Spatial Integration:**
- Automatic spatial gridding (1-5° resolution)
- Nearest neighbor matching within tolerance
- Multi-source data fusion per grid cell

### ⏰ **Temporal Integration:**
- Daily/weekly/monthly aggregation
- Time series alignment
- Missing data interpolation

### 📊 **Analysis Capabilities:**
- Cross-parameter correlation analysis
- Spatial pattern identification  
- Temporal trend analysis
- Data quality assessment

## 🔧 Configuration & Setup

### 🔑 **API Keys (Optional but Recommended):**

1. **OpenWeatherMap API Key:**
   ```bash
   export OPENWEATHER_API_KEY="your_api_key_here"
   ```

2. **Copernicus ADS API Setup:**
   ```bash
   pip install cdsapi
   # Create ~/.cdsapirc with credentials from:
   # https://cds.climate.copernicus.eu/api-how-to
   ```

3. **NASA Earthdata Login:**
   - Register at: https://urs.earthdata.nasa.gov/
   - Required for direct TROPOMI/MODIS downloads

### 📂 **Directory Structure:**
```bash
# Automatically created by scripts:
satellite_data/     # Satellite data files
weather_data/       # Weather data files  
integrated_data/    # Combined analysis outputs
```

## 💡 Usage Examples

### 🎯 **Example 1: California Air Quality Analysis**
```python
from epa_data_processor import EPADataProcessor

processor = EPADataProcessor()
datasets = processor.extract_and_load_data()

# Get California ozone data
ca_ozone = processor.filter_data_by_region(
    param_code='44201',  # Ozone
    states=['California'],
    date_range=('2024-06-01', '2024-08-31')
)

processor.save_to_csv(ca_ozone, 'california_summer_ozone.csv')
```

### 🛰️ **Example 2: Satellite NO₂ Data**
```python
from satellite_data_downloader import SatelliteDataDownloader

downloader = SatelliteDataDownloader()

# Get NO2 data for Los Angeles area
la_bbox = (-118.7, 33.7, -117.9, 34.3)  # LA metropolitan area
no2_data = downloader.download_giovanni_data(
    parameter='no2',
    bbox=la_bbox, 
    start_date='2024-07-01',
    end_date='2024-07-31'
)
```

### 🌤️ **Example 3: Weather Data Collection**
```python
from weather_data_downloader import WeatherDataDownloader

downloader = WeatherDataDownloader()

# Get weather data for New York City
nyc_weather = downloader.download_nasa_power_data(
    lat=40.7128, 
    lon=-74.0060,
    start_date='20240701',
    end_date='20240731'
)
```

### 🔗 **Example 4: Integrated Analysis**
```python
from integrated_data_system import IntegratedDataSystem

system = IntegratedDataSystem()

# Load all data sources
datasets = system.load_all_datasets()

# Create spatially integrated dataset
integrated_data = system.spatial_join_datasets(datasets)

# Analyze correlations
correlations = system.create_correlation_analysis(integrated_data)
```

## 📈 Expected Output Files

### 📊 **EPA Analysis:**
- `north_america_air_quality_summary.csv` - State-level statistics
- `california_ozone_2024.csv` - Sample regional data
- `north_america_air_quality_analysis.png` - Comprehensive plots

### 🛰️ **Satellite Data:**
- `sample_no2_YYYY-MM-DD_YYYY-MM-DD.csv` - NO₂ column data
- `sample_o3_YYYY-MM-DD_YYYY-MM-DD.csv` - Ozone column data  
- `sample_aod_YYYY-MM-DD_YYYY-MM-DD.csv` - Aerosol optical depth

### 🌤️ **Weather Data:**
- `sample_weather_YYYY-MM-DD_YYYY-MM-DD.csv` - Multi-parameter weather
- `nasa_power_LAT_LON_YYYYMMDD_YYYYMMDD.csv` - NASA POWER data
- `openweather_LAT_LON_YYYYMMDD.csv` - OpenWeatherMap data

### 🔗 **Integrated Analysis:**
- `integrated_air_quality_data_2.0deg.csv` - Spatially gridded dataset
- `temporal_series_daily.csv` - Daily time series
- `correlation_matrix.csv` - Cross-parameter correlations
- `correlation_heatmap.png` - Correlation visualization
- `integration_summary_report.json` - Comprehensive summary

## ⚡ Performance & Limitations

### 🚀 **Performance:**
- EPA data: ~17M records processed in ~2 minutes
- Satellite data: Sample generation for 500 points in ~10 seconds
- Weather data: Sample generation for 100 locations in ~5 seconds
- Integration: 2° grid processing in ~30 seconds

### ⚠️ **Limitations:**
- Real satellite data downloads require API authentication
- Some APIs have rate limits (handled automatically)
- Large temporal ranges may require chunked downloads
- Spatial resolution limited by source data resolution

## 🆘 Troubleshooting

### ❓ **Common Issues:**

1. **"No datasets loaded"**
   - Ensure EPA ZIP files are in `validation datasets/`
   - Check file permissions

2. **"API key not found"**
   - Set environment variables for API keys
   - Use sample data mode for testing

3. **"HTTP 410 Gone" for satellite APIs**
   - API endpoints may have changed
   - Check service status pages
   - Use sample data for development

4. **Memory issues with large datasets**
   - Process data in chunks
   - Increase spatial/temporal resolution
   - Use data filtering options

## 🔗 Additional Resources

- **EPA AQS Documentation**: https://www.epa.gov/aqs
- **NASA Giovanni Tutorial**: https://giovanni.gsfc.nasa.gov/giovanni/
- **TROPOMI Data Guide**: https://s5phub.copernicus.eu/
- **MODIS Documentation**: https://modis.gsfc.nasa.gov/
- **NASA POWER Documentation**: https://power.larc.nasa.gov/docs/
- **OpenWeatherMap API Docs**: https://openweathermap.org/api

---

**Need help?** Check the troubleshooting section above or review the comprehensive error handling built into each script.
