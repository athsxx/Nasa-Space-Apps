# Air Quality Monitoring Agent

A comprehensive air quality monitoring system that ingests data from TEMPO satellite and OpenAQ ground stations to provide real-time AQI calculations and health advisories.

## Features

- Dual data sources: TEMPO satellite data + OpenAQ ground monitoring
- Real-time AQI calculations using EPA standards
- Location-based queries (GPS/IP detection or manual input)
- Health advisory messages based on AQI categories
- Web API and CLI interfaces
- Support for major pollutants: PM2.5, PM10, CO, O3, NO2, SO2

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

- `GET /aqi/location/{location}` - Get AQI by location name
- `GET /aqi/coordinates?lat={lat}&lon={lon}` - Get AQI by coordinates
- `GET /aqi/auto` - Get AQI for auto-detected location

## Project Structure

```
Pipeline/
├── main.py              # FastAPI web application
├── cli.py               # Command-line interface
├── aqi_agent/
│   ├── __init__.py
│   ├── core.py          # Main AQI calculation logic
│   ├── data_sources.py  # TEMPO and OpenAQ data fetching
│   ├── location.py      # Location detection and geocoding
│   ├── aqi_calculator.py # EPA AQI formula implementation
│   └── models.py        # Data models and schemas
├── config/
│   ├── __init__.py
│   └── settings.py      # Configuration and constants
├── tests/
│   ├── __init__.py
│   └── test_aqi.py      # Unit tests
└── requirements.txt
```