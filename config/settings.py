"""Configuration settings and constants for the AQI monitoring agent."""

import os
from typing import Dict, List, Tuple

# API Configuration
OPENAQ_BASE_URL = "https://api.openaq.org/v3"  # Updated to v3
TEMPO_DATA_URL = "https://data.gesdisc.earthdata.nasa.gov"  # Placeholder - actual TEMPO API endpoint

# Demo/Fallback mode (when APIs are unavailable)
DEMO_MODE = True  # Enable demo mode with mock data

# Location Detection
IP_GEOLOCATION_API = "http://ip-api.com/json"
DEFAULT_LOCATION = {"lat": 40.7128, "lon": -74.0060}  # New York City

# Pollutant Configuration
POLLUTANTS = ["pm25", "pm10", "co", "o3", "no2", "so2"]

# EPA AQI Breakpoints (concentration ranges and corresponding AQI ranges)
# Format: {pollutant: [(c_low, c_high, aqi_low, aqi_high), ...]}
AQI_BREAKPOINTS = {
    "pm25": [  # PM2.5 (µg/m³)
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "pm10": [  # PM10 (µg/m³)
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "co": [  # CO (ppm)
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
    "o3": [  # O3 (ppm)
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
        (0.201, 0.400, 301, 400),
        (0.401, 0.500, 401, 500),
    ],
    "no2": [  # NO2 (ppb)
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
    "so2": [  # SO2 (ppb)
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
}

# AQI Categories
AQI_CATEGORIES = {
    (0, 50): {
        "category": "Good",
        "color": "#00E400",
        "health_message": "Air quality is considered satisfactory, and air pollution poses little or no risk."
    },
    (51, 100): {
        "category": "Moderate",
        "color": "#FFFF00",
        "health_message": "Air quality is acceptable for most people. However, sensitive people may experience minor issues."
    },
    (101, 150): {
        "category": "Unhealthy for Sensitive Groups",
        "color": "#FF7E00",
        "health_message": "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
    },
    (151, 200): {
        "category": "Unhealthy",
        "color": "#FF0000",
        "health_message": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
    },
    (201, 300): {
        "category": "Very Unhealthy",
        "color": "#8F3F97",
        "health_message": "Health warnings of emergency conditions. The entire population is more likely to be affected."
    },
    (301, 500): {
        "category": "Hazardous",
        "color": "#7E0023",
        "health_message": "Health alert: everyone may experience more serious health effects."
    },
}

# Unit Conversions
UNIT_CONVERSIONS = {
    # Convert to standard units used in AQI calculations
    "co": {
        "mg/m³": lambda x: x * 0.873,  # mg/m³ to ppm
        "µg/m³": lambda x: x * 0.000873,  # µg/m³ to ppm
    },
    "o3": {
        "µg/m³": lambda x: x * 0.0005,  # µg/m³ to ppm (approx)
    },
    "no2": {
        "µg/m³": lambda x: x * 0.532,  # µg/m³ to ppb
        "ppm": lambda x: x * 1000,  # ppm to ppb
    },
    "so2": {
        "µg/m³": lambda x: x * 0.382,  # µg/m³ to ppb
        "ppm": lambda x: x * 1000,  # ppm to ppb
    },
}

# Data source priorities (higher = more reliable)
DATA_SOURCE_PRIORITIES = {
    "openaq": 2,
    "tempo": 1,
}

# Request timeouts and retries
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3