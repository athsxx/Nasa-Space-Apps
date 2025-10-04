"""Data source integrations for TEMPO satellite and OpenAQ ground monitoring."""

import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

try:
    import xarray as xr
    import netCDF4 as nc
    XARRAY_AVAILABLE = True
except ImportError:
    print("Info: xarray/netCDF4 not available. TEMPO satellite data processing disabled.")
    print("      Install with: pip install netCDF4 xarray")
    print("      Note: May require HDF5 system libraries on Windows")
    XARRAY_AVAILABLE = False
    # Create dummy variables to prevent import errors
    xr = None
    nc = None

from .models import PollutantReading, StationInfo, Location
from .location import location_service
from .mock_data import mock_data_source
from config.settings import (
    OPENAQ_BASE_URL, TEMPO_DATA_URL, REQUEST_TIMEOUT, MAX_RETRIES,
    POLLUTANTS, DATA_SOURCE_PRIORITIES, DEMO_MODE
)


class OpenAQDataSource:
    """OpenAQ API data source integration."""
    
    def __init__(self):
        self.base_url = OPENAQ_BASE_URL
        self.timeout = REQUEST_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.priority = DATA_SOURCE_PRIORITIES.get("openaq", 2)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                print(f"OpenAQ request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return None
        return None
    
    def find_nearby_stations(self, location: Location, radius_km: float = 50) -> List[StationInfo]:
        """Find monitoring stations near the given location."""
        params = {
            "coordinates": f"{location.latitude},{location.longitude}",
            "radius": int(radius_km * 1000),  # Convert to meters
            "limit": 100,
            "sort": "distance"
        }
        
        data = self._make_request("locations", params)
        if not data or "results" not in data:
            return []
        
        stations = []
        for station_data in data["results"]:
            try:
                # Calculate distance
                distance = location_service.calculate_distance(
                    location.latitude, location.longitude,
                    station_data["coordinates"]["latitude"],
                    station_data["coordinates"]["longitude"]
                )
                
                # Get available parameters
                parameters = [param["parameter"] for param in station_data.get("parameters", [])]
                
                station = StationInfo(
                    id=str(station_data["id"]),
                    name=station_data.get("name", "Unknown Station"),
                    latitude=station_data["coordinates"]["latitude"],
                    longitude=station_data["coordinates"]["longitude"],
                    distance_km=distance,
                    source="openaq",
                    parameters=parameters,
                    last_updated=datetime.fromisoformat(
                        station_data.get("lastUpdated", "").replace("Z", "+00:00")
                    ) if station_data.get("lastUpdated") else None
                )
                stations.append(station)
            except (KeyError, ValueError) as e:
                print(f"Error parsing station data: {e}")
                continue
        
        return sorted(stations, key=lambda s: s.distance_km)
    
    def get_latest_measurements(self, location: Location, 
                              pollutants: Optional[List[str]] = None,
                              max_distance_km: float = 50) -> List[PollutantReading]:
        """Get latest pollutant measurements near the location."""
        if pollutants is None:
            pollutants = POLLUTANTS
        
        readings = []
        
        # Find nearby stations
        stations = self.find_nearby_stations(location, max_distance_km)
        if not stations:
            print(f"No OpenAQ stations found within {max_distance_km}km")
            return readings
        
        # Get measurements from the closest stations
        for station in stations[:5]:  # Limit to 5 closest stations
            station_readings = self._get_station_measurements(station, pollutants)
            readings.extend(station_readings)
        
        return readings
    
    def _get_station_measurements(self, station: StationInfo, 
                                 pollutants: List[str]) -> List[PollutantReading]:
        """Get measurements from a specific station."""
        readings = []
        
        # Get recent measurements
        params = {
            "location_id": station.id,
            "limit": 100,
            "sort": "desc",
            "order_by": "datetime"
        }
        
        data = self._make_request("measurements", params)
        if not data or "results" not in data:
            return readings
        
        # Process measurements
        for measurement in data["results"]:
            try:
                parameter = measurement.get("parameter")
                if not parameter or parameter not in pollutants:
                    continue
                
                # Parse timestamp
                timestamp_str = measurement.get("date", {}).get("utc")
                if not timestamp_str:
                    continue
                
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                
                # Only include recent measurements (within 24 hours)
                if datetime.now(timestamp.tzinfo) - timestamp > timedelta(hours=24):
                    continue
                
                reading = PollutantReading(
                    pollutant=parameter,
                    value=float(measurement["value"]),
                    unit=measurement.get("unit", "µg/m³"),
                    timestamp=timestamp,
                    source="openaq",
                    station_name=station.name,
                    distance_km=station.distance_km
                )
                readings.append(reading)
                
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error parsing measurement: {e}")
                continue
        
        return readings


class TEMPODataSource:
    """TEMPO satellite data source integration."""
    
    def __init__(self):
        self.base_url = TEMPO_DATA_URL
        self.timeout = REQUEST_TIMEOUT
        self.priority = DATA_SOURCE_PRIORITIES.get("tempo", 1)
    
    def get_satellite_data(self, location: Location,
                          pollutants: Optional[List[str]] = None) -> List[PollutantReading]:
        """
        Get satellite-based pollutant data for the location.
        Note: This is a placeholder implementation. Actual TEMPO data access
        would require specific API credentials and data processing.
        """
        if not XARRAY_AVAILABLE:
            print("Warning: xarray not available for TEMPO data processing")
            return []
        
        if pollutants is None:
            pollutants = POLLUTANTS
        
        readings = []
        
        # This is a mock implementation - actual TEMPO integration would involve:
        # 1. Authenticating with NASA Earthdata
        # 2. Querying TEMPO data products
        # 3. Processing NetCDF files with xarray
        # 4. Extracting data for specific coordinates
        
        # For now, return empty list with warning
        print("TEMPO satellite data integration not yet implemented")
        print("This would require NASA Earthdata credentials and specific data processing")
        
        return readings
    
    def _process_netcdf_data(self, file_path: str, location: Location) -> List[PollutantReading]:
        """Process TEMPO NetCDF data file (placeholder)."""
        if not XARRAY_AVAILABLE:
            return []
        
        try:
            # This would be the actual implementation for processing TEMPO data
            # ds = xr.open_dataset(file_path)
            # Extract data for location coordinates
            # Convert to PollutantReading objects
            pass
        except Exception as e:
            print(f"Error processing TEMPO data: {e}")
        
        return []


class DataSourceManager:
    """Manager for coordinating multiple data sources."""
    
    def __init__(self):
        self.openaq = OpenAQDataSource()
        self.tempo = TEMPODataSource()
        self.sources = {
            "openaq": self.openaq,
            "tempo": self.tempo
        }
    
    async def get_all_pollutant_data(self, location: Location,
                                   pollutants: Optional[List[str]] = None,
                                   sources: Optional[List[str]] = None) -> List[PollutantReading]:
        """Get pollutant data from all available sources."""
        if pollutants is None:
            pollutants = POLLUTANTS
        
        if sources is None:
            sources = list(self.sources.keys())
        
        all_readings = []
        
        # Get data from each source
        for source_name in sources:
            if source_name not in self.sources:
                continue
            
            try:
                if source_name == "openaq":
                    readings = self.openaq.get_latest_measurements(location, pollutants)
                elif source_name == "tempo":
                    readings = self.tempo.get_satellite_data(location, pollutants)
                else:
                    continue
                
                all_readings.extend(readings)
                print(f"Retrieved {len(readings)} readings from {source_name}")
                
            except Exception as e:
                print(f"Error getting data from {source_name}: {e}")
                continue
        
        # If no real data was obtained and demo mode is enabled, use mock data
        if not all_readings and DEMO_MODE:
            print("No real data available, using mock data for demonstration...")
            mock_readings = mock_data_source.get_mock_readings(location, pollutants)
            all_readings.extend(mock_readings)
            print(f"Generated {len(mock_readings)} mock readings")
        
        # Remove duplicates and prioritize by source
        return self._deduplicate_readings(all_readings)
    
    def _deduplicate_readings(self, readings: List[PollutantReading]) -> List[PollutantReading]:
        """Remove duplicate readings, prioritizing by source and recency."""
        if not readings:
            return []
        
        # Group by pollutant
        pollutant_groups = {}
        for reading in readings:
            if reading.pollutant not in pollutant_groups:
                pollutant_groups[reading.pollutant] = []
            pollutant_groups[reading.pollutant].append(reading)
        
        # Select best reading for each pollutant
        best_readings = []
        for _, group in pollutant_groups.items():
            # Sort by priority (source) and timestamp (recency)
            def sort_key(r):
                source_priority = DATA_SOURCE_PRIORITIES.get(r.source, 0)
                return (-source_priority, -r.timestamp.timestamp())
            
            best_reading = sorted(group, key=sort_key)[0]
            best_readings.append(best_reading)
        
        return best_readings


# Global data source manager instance
data_manager = DataSourceManager()