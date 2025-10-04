"""Air Quality Monitoring Agent Package."""

from .core import aqi_agent, AQIAgent
from .models import *
from .aqi_calculator import AQICalculator
from .location import location_service, LocationService
from .data_sources import data_manager, DataSourceManager

__version__ = "1.0.0"
__author__ = "NASA Air Quality Monitoring Team"
__description__ = "Air Quality Monitoring Agent with TEMPO satellite and OpenAQ integration"

# Export main components
__all__ = [
    "aqi_agent",
    "AQIAgent", 
    "AQICalculator",
    "location_service",
    "LocationService",
    "data_manager",
    "DataSourceManager",
    # Models
    "Location",
    "PollutantReading",
    "AQICalculation", 
    "AQIResponse",
    "LocationRequest",
    "ErrorResponse",
    "StationInfo",
    "HealthAdvisory"
]