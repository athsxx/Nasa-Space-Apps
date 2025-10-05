# Air Quality Monitoring Agent

A comprehensive air quality monitoring system that ingests data from TEMPO satellite and OpenAQ ground stations to provide real-time AQI calculations and health advisories.

## Features

- **Dual data sources**: TEMPO satellite data + OpenAQ ground monitoring
- **Real-time AQI calculations** using EPA standards
- **Machine Learning Fusion**: Trained Random Forest models predict surface concentrations
- **Location-based queries** (GPS/IP detection or manual input)
- **Health advisory messages** based on AQI categories
- **Web API and CLI interfaces** with ML prediction endpoints
- **Support for major pollutants**: PM2.5, PM10, CO, O3, NO2, SO2
- **Weather integration** for enhanced predictions
- **Model training pipeline** with synthetic and real data

## Installation

### Quick Start (Windows)
1. Double-click `start.bat` and choose option [1] to install
2. Or run manually:
   ```bash
   python install.py
   ```

### Manual Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Quick Start (Windows)
- **Interactive Menu**: Double-click `start.bat`
- **Web API**: Choose option [2] from menu or run `python main.py`
- **CLI**: Choose option [3] from menu for interactive mode

### Web API
```bash
python main.py
```
Then visit `http://localhost:8000/docs` for the interactive API documentation.

### CLI
```bash
python cli.py --location "New York, NY"
# or
python cli.py --lat 40.7128 --lon -74.0060
# or
python cli.py --auto
```

### Demo
```bash
python demo.py
```

## API Endpoints

### Traditional AQI Endpoints
- `GET /aqi/location/{location}` - Get AQI by location name
- `GET /aqi/coordinates?lat={lat}&lon={lon}` - Get AQI by coordinates
- `GET /aqi/auto` - Get AQI for auto-detected location

### ML Fusion Endpoints (New!)
- `GET /ml/predict/location/{location}` - Get ML-predicted AQI by location
- `GET /ml/predict/coordinates?lat={lat}&lon={lon}` - Get ML-predicted AQI by coordinates
- `GET /ml/info` - Get ML model information and performance
- `GET /ml/features` - Get feature importance from trained models

## Machine Learning Fusion

### Training ML Models
```bash
# Train ML fusion models with synthetic data
python train_ml_fusion.py

# Train with custom parameters
python train_ml_fusion.py --n-tempo 2000 --n-openaq 1000 --optimize
```

### ML Fusion Demo
```bash
# Run complete ML fusion demonstration
python demo_ml_fusion.py
```

### Using ML Predictions
```bash
# Get ML prediction for a location
curl "http://localhost:8000/ml/predict/location/New York"

# Get ML model information
curl "http://localhost:8000/ml/info"
```

## Project Structure

```
Pipeline/
├── main.py              # FastAPI web application
├── cli.py               # Command-line interface
├── train_ml_fusion.py   # ML model training pipeline
├── demo_ml_fusion.py    # ML fusion demonstration
├── aqi_agent/
│   ├── __init__.py
│   ├── core.py          # Main AQI calculation logic (with ML integration)
│   ├── data_sources.py  # TEMPO and OpenAQ data fetching
│   ├── location.py      # Location detection and geocoding
│   ├── aqi_calculator.py # EPA AQI formula implementation
│   └── models.py        # Data models and schemas
├── ml_fusion/           # Machine Learning Fusion Module
│   ├── __init__.py
│   ├── data_preprocessor.py     # Data cleaning and preparation
│   ├── spatial_temporal_fusion.py # Data matching and alignment
│   ├── model_trainer.py         # Random Forest training
│   ├── surface_estimator.py     # ML prediction interface
│   └── weather_integration.py   # Weather data integration
├── config/
│   ├── __init__.py
│   └── settings.py      # Configuration and constants
├── models/              # Trained ML models (created after training)
│   └── surface_estimator.pkl
├── tests/
│   ├── __init__.py
│   └── test_aqi.py      # Unit tests
└── requirements.txt
```