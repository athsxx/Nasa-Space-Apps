"""Mock data source for demonstration when real APIs are unavailable."""

from typing import List
from datetime import datetime, timedelta
import random

from aqi_agent.models import PollutantReading, Location


class MockDataSource:
    """Mock data source providing realistic pollutant readings for demonstration."""
    
    def __init__(self):
        self.name = "mock"
        self.priority = 3
        
        # City-specific pollution profiles (typical ranges)
        self.city_profiles = {
            "new york": {"pm25": (15, 45), "pm10": (20, 60), "co": (1, 6), "o3": (0.04, 0.08), "no2": (20, 80), "so2": (2, 15)},
            "los angeles": {"pm25": (20, 55), "pm10": (30, 80), "co": (2, 8), "o3": (0.06, 0.12), "no2": (30, 100), "so2": (1, 10)},
            "chicago": {"pm25": (12, 40), "pm10": (18, 65), "co": (1, 5), "o3": (0.03, 0.07), "no2": (15, 70), "so2": (3, 20)},
            "houston": {"pm25": (18, 50), "pm10": (25, 75), "co": (2, 7), "o3": (0.05, 0.10), "no2": (25, 90), "so2": (5, 25)},
            "phoenix": {"pm25": (25, 60), "pm10": (40, 100), "co": (2, 6), "o3": (0.07, 0.13), "no2": (20, 85), "so2": (1, 8)},
            "london": {"pm25": (10, 35), "pm10": (15, 50), "co": (1, 4), "o3": (0.02, 0.06), "no2": (25, 95), "so2": (2, 12)},
            "paris": {"pm25": (12, 38), "pm10": (18, 55), "co": (1, 5), "o3": (0.03, 0.07), "no2": (30, 100), "so2": (3, 15)},
            "tokyo": {"pm25": (8, 30), "pm10": (12, 45), "co": (1, 3), "o3": (0.02, 0.05), "no2": (15, 60), "so2": (1, 8)},
            "beijing": {"pm25": (50, 150), "pm10": (80, 200), "co": (3, 10), "o3": (0.04, 0.08), "no2": (40, 120), "so2": (10, 50)},
            "mumbai": {"pm25": (40, 120), "pm10": (60, 180), "co": (2, 8), "o3": (0.03, 0.07), "no2": (35, 110), "so2": (8, 40)},
            "default": {"pm25": (15, 45), "pm10": (20, 60), "co": (1, 6), "o3": (0.04, 0.08), "no2": (20, 80), "so2": (2, 15)}
        }
    
    def get_city_profile(self, location: Location) -> dict:
        """Get pollution profile for a city based on location."""
        if location.city:
            city_name = location.city.lower()
            for profile_city in self.city_profiles:
                if profile_city in city_name:
                    return self.city_profiles[profile_city]
        
        # Default profile if city not found
        return self.city_profiles["default"]
    
    def generate_realistic_reading(self, pollutant: str, base_range: tuple, time_of_day: int) -> float:
        """Generate realistic pollutant reading with time-of-day variation."""
        min_val, max_val = base_range
        
        # Time of day effects (simplified)
        time_factors = {
            "pm25": 1.0 + 0.3 * (abs(time_of_day - 12) / 12),  # Higher during rush hours
            "pm10": 1.0 + 0.2 * (abs(time_of_day - 12) / 12),
            "co": 1.0 + 0.4 * (abs(time_of_day - 8) / 12 if time_of_day < 10 or time_of_day > 17 else 0),
            "o3": 0.5 + 0.5 * max(0, (time_of_day - 6) / 12) if 6 <= time_of_day <= 18 else 0.3,
            "no2": 1.0 + 0.5 * (1 if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19 else 0),
            "so2": 1.0 + 0.1 * random.random()
        }
        
        factor = time_factors.get(pollutant, 1.0)
        base_value = random.uniform(min_val, max_val)
        return max(0, base_value * factor)
    
    def get_mock_readings(self, location: Location, pollutants: List[str] = None) -> List[PollutantReading]:
        """Generate mock pollutant readings for a location."""
        if pollutants is None:
            pollutants = ["pm25", "pm10", "co", "o3", "no2", "so2"]
        
        profile = self.get_city_profile(location)
        current_time = datetime.now()
        hour = current_time.hour
        
        readings = []
        
        # Standard units for each pollutant
        units = {
            "pm25": "µg/m³",
            "pm10": "µg/m³", 
            "co": "ppm",
            "o3": "ppm",
            "no2": "ppb",
            "so2": "ppb"
        }
        
        for pollutant in pollutants:
            if pollutant in profile:
                concentration = self.generate_realistic_reading(
                    pollutant, profile[pollutant], hour
                )
                
                # Add some stations at different distances
                stations = [
                    ("Central Monitor", 1.2),
                    ("Downtown Station", 2.8),
                    ("Suburban Monitor", 5.1)
                ]
                
                # Create readings from different stations (with slight variations)
                for station_name, distance in stations:
                    # Vary readings slightly between stations
                    station_variation = 1.0 + random.uniform(-0.15, 0.15)
                    station_concentration = max(0, concentration * station_variation)
                    
                    reading = PollutantReading(
                        pollutant=pollutant,
                        value=round(station_concentration, 2),
                        unit=units[pollutant],
                        timestamp=current_time - timedelta(minutes=random.randint(0, 30)),
                        source="mock",
                        station_name=station_name,
                        distance_km=distance
                    )
                    readings.append(reading)
        
        return readings


# Global mock data source instance
mock_data_source = MockDataSource()