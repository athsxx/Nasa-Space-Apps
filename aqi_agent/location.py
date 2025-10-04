"""Location detection and geocoding functionality."""

import requests
from typing import Optional

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    print("Warning: geopy not available. Geocoding features will be limited.")
    GEOPY_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class Nominatim:
        def __init__(self, *args, **kwargs): pass
        def geocode(self, *args, **kwargs): return None
        def reverse(self, *args, **kwargs): return None
    
    class GeocoderTimedOut(Exception): pass
    class GeocoderServiceError(Exception): pass

from .models import Location
from config.settings import IP_GEOLOCATION_API, DEFAULT_LOCATION, REQUEST_TIMEOUT


class LocationService:
    """Service for location detection and geocoding."""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="aqi_monitoring_agent")
        self.timeout = REQUEST_TIMEOUT
    
    def get_location_from_ip(self) -> Optional[Location]:
        """Detect location based on IP address."""
        try:
            response = requests.get(IP_GEOLOCATION_API, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success':
                return Location(
                    latitude=data.get('lat', DEFAULT_LOCATION['lat']),
                    longitude=data.get('lon', DEFAULT_LOCATION['lon']),
                    city=data.get('city'),
                    country=data.get('country'),
                    address=f"{data.get('city', '')}, {data.get('regionName', '')}, {data.get('country', '')}"
                )
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"IP geolocation failed: {e}")
        
        # Return default location on failure
        return Location(
            latitude=DEFAULT_LOCATION['lat'],
            longitude=DEFAULT_LOCATION['lon'],
            city="New York City",
            country="United States",
            address="New York City, NY, United States"
        )
    
    def geocode_address(self, address: str) -> Optional[Location]:
        """Convert address string to coordinates."""
        try:
            location = self.geolocator.geocode(address, timeout=self.timeout)
            if location:
                # Parse address components
                address_parts = location.address.split(', ')
                city = address_parts[0] if address_parts else None
                country = address_parts[-1] if len(address_parts) > 1 else None
                
                return Location(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    city=city,
                    country=country,
                    address=location.address
                )
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding failed for '{address}': {e}")
        
        return None
    
    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Location]:
        """Convert coordinates to address."""
        try:
            location = self.geolocator.reverse(
                (latitude, longitude), 
                timeout=self.timeout,
                language='en'
            )
            if location:
                # Parse address components
                address_parts = location.address.split(', ')
                city = address_parts[0] if address_parts else None
                country = address_parts[-1] if len(address_parts) > 1 else None
                
                return Location(
                    latitude=latitude,
                    longitude=longitude,
                    city=city,
                    country=country,
                    address=location.address
                )
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Reverse geocoding failed for ({latitude}, {longitude}): {e}")
        
        # Return basic location with coordinates only
        return Location(
            latitude=latitude,
            longitude=longitude,
            city=None,
            country=None,
            address=f"{latitude:.4f}, {longitude:.4f}"
        )
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate coordinate ranges."""
        return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def resolve_location(self, location_str: Optional[str] = None, 
                        latitude: Optional[float] = None, 
                        longitude: Optional[float] = None,
                        use_auto_location: bool = False) -> Location:
        """
        Resolve location from various inputs.
        Priority: coordinates > address > auto-detection
        """
        # Option 1: Use provided coordinates
        if latitude is not None and longitude is not None:
            if self.validate_coordinates(latitude, longitude):
                location = self.reverse_geocode(latitude, longitude)
                if location:
                    return location
                # Fallback to basic coordinate location
                return Location(
                    latitude=latitude,
                    longitude=longitude,
                    address=f"{latitude:.4f}, {longitude:.4f}"
                )
            else:
                raise ValueError(f"Invalid coordinates: ({latitude}, {longitude})")
        
        # Option 2: Geocode address string
        if location_str:
            location = self.geocode_address(location_str)
            if location:
                return location
            else:
                raise ValueError(f"Could not geocode address: {location_str}")
        
        # Option 3: Auto-detect from IP
        if use_auto_location:
            return self.get_location_from_ip()
        
        # Default: raise error for insufficient input
        raise ValueError("Must provide either coordinates, address, or enable auto-location")


# Global location service instance
location_service = LocationService()