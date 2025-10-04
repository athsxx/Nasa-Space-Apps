# NASA Space Apps - Air Quality Data Analysis

This repository contains comprehensive air quality data analysis tools for North America using EPA AQS (Air Quality System) data.

## 🌍 Project Overview

This project analyzes air quality data from the EPA Air Quality System, covering all monitoring stations across the United States and territories. The analysis includes four major air pollutants:

- **SO₂** (Sulfur Dioxide) - 394 monitoring sites, 3.1M measurements
- **O₃** (Ozone) - 1,241 monitoring sites, 9.0M measurements  
- **NO₂** (Nitrogen Dioxide) - 460 monitoring sites, 3.5M measurements
- **CO** (Carbon Monoxide) - 236 monitoring sites, 1.7M measurements

**Total Dataset**: 17.3 million hourly measurements from 2,331 monitoring sites across 53 states/territories for the year 2024.

## 📊 Key Findings

### Highest Pollutant Concentrations by Region:

**Ozone (O₃)**:
1. Colorado - 0.0411 ppm
2. Wyoming - 0.0391 ppm  
3. Nevada - 0.0380 ppm
4. Arizona - 0.0375 ppm
5. Utah - 0.0361 ppm

**Sulfur Dioxide (SO₂)**:
1. Idaho - 1.77 ppb
2. Alaska - 1.47 ppb
3. North Dakota - 1.18 ppb
4. Hawaii - 1.17 ppb
5. Georgia - 1.06 ppb

**Nitrogen Dioxide (NO₂)**:
1. Arizona - 13.25 ppb
2. Nevada - 13.13 ppb
3. Georgia - 12.95 ppb
4. Washington - 12.48 ppb
5. Illinois - 12.26 ppb

**Carbon Monoxide (CO)**:
1. Georgia - 0.43 ppm
2. Michigan - 0.40 ppm
3. Oklahoma - 0.36 ppm
4. Washington D.C. - 0.34 ppm
5. New Jersey - 0.34 ppm

## 🗂️ Repository Structure

### Data Files
- `validation datasets/` - Contains EPA AQS data files
  - `hourly_44201_2024_ozone.zip` - Ozone measurements
  - `hourly_42401_2024_SO2.zip` - Sulfur Dioxide measurements  
  - `hourly_42602_2024_NO2.zip` - Nitrogen Dioxide measurements
  - `hourly_42101_2024_CO.zip` - Carbon Monoxide measurements
  - `source.txt` - Data source URL

### Analysis Scripts
- `epa_data_processor.py` - EPA AQS data processing and analysis
- `visualize_air_quality.py` - EPA data visualization and plotting
- **NEW**: `satellite_data_downloader.py` - Multi-source satellite data collection
- **NEW**: `weather_data_downloader.py` - Multi-source meteorological data
- **NEW**: `integrated_data_system.py` - Unified multi-source data integration
- `requirements.txt` - Python package dependencies

### Generated Output Files

#### EPA Analysis:
- `north_america_air_quality_summary.csv` - State-level summary statistics
- `california_ozone_2024.csv` - Example: California ozone data
- `major_cities_co_2024.csv` - Example: CO data from major states
- `north_america_air_quality_analysis.png` - Comprehensive visualization

#### **NEW**: Satellite Data:
- `satellite_data/sample_no2_YYYY-MM-DD_YYYY-MM-DD.csv` - NO₂ column measurements
- `satellite_data/sample_o3_YYYY-MM-DD_YYYY-MM-DD.csv` - Ozone column measurements  
- `satellite_data/sample_aod_YYYY-MM-DD_YYYY-MM-DD.csv` - Aerosol optical depth

#### **NEW**: Weather Data:
- `weather_data/sample_weather_YYYY-MM-DD_YYYY-MM-DD.csv` - Multi-parameter meteorology
- `weather_data/nasa_power_LAT_LON_YYYYMMDD_YYYYMMDD.csv` - NASA POWER data
- `weather_data/openweather_LAT_LON_YYYYMMDD.csv` - OpenWeatherMap data

#### **NEW**: Integrated Analysis:
- `integrated_data/integrated_air_quality_data_2.0deg.csv` - Spatially gridded multi-source dataset
- `integrated_data/temporal_series_daily.csv` - Daily time series across all sources
- `integrated_data/correlation_matrix.csv` - Cross-parameter correlation analysis
- `integrated_data/correlation_heatmap.png` - Correlation visualization
- `integrated_data/integration_summary_report.json` - Comprehensive analysis summary

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.5.0
- numpy >= 1.24.0
- matplotlib >= 3.6.0
- seaborn >= 0.11.0

### Running the Analysis

#### **EPA Air Quality Analysis**:
```bash
python epa_data_processor.py          # Process EPA AQS data
python visualize_air_quality.py       # Create EPA visualizations
```

#### **NEW: Multi-Source Data Collection**:
```bash
python satellite_data_downloader.py   # Download satellite data (NO₂, O₃, AOD)
python weather_data_downloader.py     # Download weather data (Temp, Wind, etc.)
python integrated_data_system.py      # Integrate all data sources
```

#### **Complete Analysis Pipeline**:
```bash
# Run all analyses in sequence
python epa_data_processor.py
python satellite_data_downloader.py  
python weather_data_downloader.py
python integrated_data_system.py
python visualize_air_quality.py
```

#### **Data Source Configuration**:
```bash
# Optional: Set API keys for enhanced data access
export OPENWEATHER_API_KEY="your_api_key"
export WEATHERBIT_API_KEY="your_api_key"

# For Copernicus data, create ~/.cdsapirc with credentials
# See: https://cds.climate.copernicus.eu/api-how-to
```

## 📈 Analysis Features

### EPADataProcessor Class Methods:

- `extract_and_load_data()` - Extracts ZIP files and loads CSV data
- `get_data_summary()` - Provides comprehensive data statistics
- `get_north_american_summary()` - State-level and temporal analysis
- `filter_data_by_region()` - Filter by states, cities, date ranges
- `export_filtered_data()` - Export custom datasets to CSV
- `create_state_summary()` - Generate state-level summary file

### Filtering Options:

```python
# Example: Get California ozone data
processor.filter_data_by_region(
    param_code='44201',  # Ozone
    states=['California'],
    date_range=('2024-06-01', '2024-08-31')  # Summer months
)

# Example: Get data from multiple states
processor.filter_data_by_region(
    param_code='42101',  # Carbon Monoxide
    states=['California', 'New York', 'Texas', 'Florida'],
    limit=1000
)
```

## 🗺️ Data Coverage

### Geographic Coverage:
- **53** States and Territories
- **2,331** Total monitoring sites
- Coverage includes all major metropolitan areas
- Rural and urban monitoring stations

### Temporal Coverage:
- **Full year 2024** (January 1 - December 31)
- **Hourly measurements** (8760 hours per year)
- **17.3 million** total data points

### Monitoring Network:
- **EPA AQS certified** monitoring equipment
- **Quality assured** data with standardized procedures
- **Real-time** and **research-grade** monitors

## 📊 Data Structure

Each measurement record contains:
```
- State Code, County Code, Site Number (location identifiers)  
- Parameter Code, Parameter Name (pollutant type)
- Date Local, Time Local (measurement timestamp)
- Sample Measurement (concentration value)
- Units of Measure (ppm, ppb, μg/m³)
- Latitude, Longitude (geographic coordinates)
- Method Type, Method Code (measurement technique)
- State Name, County Name (location names)
- Quality flags and metadata
```

## �️ Multi-Source Data Integration System

### **NEW: Comprehensive Data Collection**

In addition to EPA ground-based measurements, the system now includes:

#### 📡 **Satellite Data Collection:**
- **NO₂ Column Density** - TROPOMI/OMI satellites (molecules/cm²)
- **O₃ Column Density** - TROPOMI/OMI satellites (Dobson Units)  
- **AOD (Aerosol Optical Depth)** - MODIS/VIIRS satellites (dimensionless)

#### 🌤️ **Meteorological Data:**
- **Temperature** (°C) 
- **Relative Humidity** (%)
- **Wind Speed & Direction** (m/s, converted to u/v components)
- **Precipitation** (mm)
- **Solar Radiation** (W/m²)
- **UV Index**

#### 🔗 **Integrated Analysis:**
- **Spatial Integration**: 2° grid system across North America
- **Temporal Analysis**: Daily/weekly/monthly aggregation
- **Correlation Analysis**: Cross-parameter relationships
- **Multi-source Data Fusion**: EPA + Satellite + Weather

### **Data Sources:**
- **NASA Giovanni** - Free satellite data access
- **NASA POWER** - Global meteorological data  
- **NOAA NWS API** - Real-time weather observations
- **OpenWeatherMap** - Current/historical weather (API key)
- **Copernicus ADS** - European atmospheric data (registration)

## �🔬 Scientific Applications

### Environmental Research:
- Air pollution trend analysis
- Regional comparison studies  
- Seasonal pattern identification
- Policy effectiveness assessment
- **NEW**: Satellite-ground data validation
- **NEW**: Weather-pollution correlation studies

### Public Health:
- Health risk assessment
- Exposure analysis
- Epidemiological studies
- Air quality index calculation
- **NEW**: Multi-parameter health impact modeling

### Climate Studies:
- Atmospheric chemistry research
- Weather pattern correlations
- Long-term trend analysis
- Model validation
- **NEW**: Satellite-based emission monitoring
- **NEW**: Meteorological influence on air quality

## 🌟 Key Insights

### Seasonal Patterns:

**Ozone**: Peaks in spring/summer (0.0375 ppm in April)
**NO₂**: Higher in winter months (10.68 ppb in December)  
**CO**: Elevated in winter (0.345 ppm in December)
**SO₂**: Relatively stable year-round (0.6-0.8 ppb)

### Regional Hotspots:

- **Western States**: Higher ozone concentrations (elevation/sunlight effects)
- **Industrial Areas**: Elevated SO₂ and NO₂ levels
- **Urban Centers**: Higher CO and NO₂ from traffic
- **Coastal Areas**: Generally lower pollutant levels

## 📚 Data Sources

- **EPA Air Quality System (AQS)**: https://aqs.epa.gov/
- **Download Portal**: https://aqs.epa.gov/aqsweb/airdata/download_files.html
- **Data Quality**: EPA-certified, quality-assured measurements
- **Update Frequency**: Data updated annually with previous year's measurements

## 🔗 Related Resources

- [EPA AQS Documentation](https://www.epa.gov/aqs)
- [Air Quality Standards](https://www.epa.gov/criteria-air-pollutants/naaqs-table)
- [NASA Air Quality Applications](https://airquality.gsfc.nasa.gov/)
- [OpenAQ Global Data](https://openaq.org/)

## 📝 License

This project is for educational and research purposes. EPA data is in the public domain. Please cite appropriate sources when using this analysis.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional pollutant parameters (PM2.5, PM10)
- Time series forecasting models
- Interactive web dashboards
- Integration with satellite data
- Machine learning predictions

## 📧 Contact

For questions about this analysis or collaboration opportunities, please open an issue in this repository.
