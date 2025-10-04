"""Data models and schemas for the AQI monitoring agent."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class Location(BaseModel):
    """Location information."""
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    city: Optional[str] = Field(None, description="City name")
    country: Optional[str] = Field(None, description="Country name")
    address: Optional[str] = Field(None, description="Full address")


class PollutantReading(BaseModel):
    """Single pollutant measurement."""
    pollutant: str = Field(..., description="Pollutant name (pm25, pm10, co, o3, no2, so2)")
    value: float = Field(..., description="Concentration value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    source: str = Field(..., description="Data source (openaq, tempo)")
    station_name: Optional[str] = Field(None, description="Monitoring station name")
    distance_km: Optional[float] = Field(None, description="Distance from requested location")


class AQICalculation(BaseModel):
    """AQI calculation result for a single pollutant."""
    pollutant: str = Field(..., description="Pollutant name")
    concentration: float = Field(..., description="Standardized concentration value")
    unit: str = Field(..., description="Standard unit for AQI calculation")
    aqi_value: int = Field(..., description="Calculated AQI value")
    category: str = Field(..., description="AQI category")
    color: str = Field(..., description="Category color code")


class AQIResponse(BaseModel):
    """Complete AQI response."""
    location: Location = Field(..., description="Location information")
    timestamp: datetime = Field(..., description="Response timestamp")
    overall_aqi: int = Field(..., description="Overall AQI value (maximum of all pollutants)")
    overall_category: str = Field(..., description="Overall AQI category")
    overall_color: str = Field(..., description="Overall category color")
    dominant_pollutant: str = Field(..., description="Pollutant causing highest AQI")
    health_message: str = Field(..., description="Health advisory message")
    pollutant_details: List[AQICalculation] = Field(..., description="Detailed AQI for each pollutant")
    raw_readings: List[PollutantReading] = Field(..., description="Raw pollutant measurements")
    data_sources: List[str] = Field(..., description="Data sources used")


class LocationRequest(BaseModel):
    """Location request model."""
    location: Optional[str] = Field(None, description="Location name (city, address)")
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")
    use_auto_location: bool = Field(False, description="Use IP-based location detection")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    code: int = Field(..., description="Error code")


class DataSourceConfig(BaseModel):
    """Data source configuration."""
    name: str = Field(..., description="Data source name")
    enabled: bool = Field(True, description="Whether source is enabled")
    priority: int = Field(1, description="Source priority (higher = preferred)")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_distance_km: float = Field(50.0, description="Maximum distance for station selection")


class StationInfo(BaseModel):
    """Monitoring station information."""
    id: str = Field(..., description="Station ID")
    name: str = Field(..., description="Station name")
    latitude: float = Field(..., description="Station latitude")
    longitude: float = Field(..., description="Station longitude")
    distance_km: float = Field(..., description="Distance from requested location")
    source: str = Field(..., description="Data source")
    parameters: List[str] = Field(..., description="Available parameters")
    last_updated: Optional[datetime] = Field(None, description="Last data update")


class HealthAdvisory(BaseModel):
    """Health advisory information."""
    aqi_range: tuple = Field(..., description="AQI range (min, max)")
    category: str = Field(..., description="Category name")
    color: str = Field(..., description="Color code")
    sensitive_groups: List[str] = Field(..., description="Affected sensitive groups")
    general_message: str = Field(..., description="General health message")
    recommendations: List[str] = Field(..., description="Recommended actions")


# Type aliases for convenience
PollutantData = Dict[str, List[PollutantReading]]
AQIResults = Dict[str, AQICalculation]