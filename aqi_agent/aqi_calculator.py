"""AQI calculation engine using EPA standards."""

from typing import Dict, List, Tuple
from datetime import datetime

from .models import AQICalculation, AQIResponse, PollutantReading
from config.settings import AQI_BREAKPOINTS, AQI_CATEGORIES, UNIT_CONVERSIONS


class AQICalculator:
    """EPA AQI calculation engine."""
    
    def __init__(self):
        self.breakpoints = AQI_BREAKPOINTS
        self.categories = AQI_CATEGORIES
        self.conversions = UNIT_CONVERSIONS
    
    def convert_units(self, pollutant: str, value: float, from_unit: str, to_unit: str) -> float:
        """Convert pollutant concentration between units."""
        if from_unit == to_unit:
            return value
        
        if pollutant in self.conversions and from_unit in self.conversions[pollutant]:
            return self.conversions[pollutant][from_unit](value)
        
        # Default: assume no conversion needed (same units)
        return value
    
    def standardize_concentration(self, pollutant: str, value: float, unit: str) -> Tuple[float, str]:
        """Standardize pollutant concentration to EPA AQI standard units."""
        # Standard units for EPA AQI calculations
        standard_units = {
            "pm25": "µg/m³",
            "pm10": "µg/m³", 
            "co": "ppm",
            "o3": "ppm",
            "no2": "ppb",
            "so2": "ppb"
        }
        
        if pollutant not in standard_units:
            return value, unit
        
        target_unit = standard_units[pollutant]
        converted_value = self.convert_units(pollutant, value, unit, target_unit)
        
        return converted_value, target_unit
    
    def calculate_aqi_for_pollutant(self, pollutant: str, concentration: float) -> int:
        """
        Calculate AQI for a single pollutant using EPA formula.
        AQI = ((I_hi - I_lo)/(C_hi - C_lo)) * (C - C_lo) + I_lo
        """
        if pollutant not in self.breakpoints:
            raise ValueError(f"Unknown pollutant: {pollutant}")
        
        breakpoints = self.breakpoints[pollutant]
        
        # Find the appropriate breakpoint range
        for c_low, c_high, aqi_low, aqi_high in breakpoints:
            if c_low <= concentration <= c_high:
                # Apply EPA AQI formula
                aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
                return round(aqi)
        
        # If concentration exceeds all breakpoints, use the highest range
        if concentration > breakpoints[-1][1]:
            # Extrapolate using the highest range
            c_low, c_high, aqi_low, aqi_high = breakpoints[-1]
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
            return min(round(aqi), 500)  # Cap at 500
        
        # If concentration is below all breakpoints, return 0
        return 0
    
    def get_aqi_category(self, aqi_value: int) -> Dict[str, str]:
        """Get AQI category information for a given AQI value."""
        for (low, high), info in self.categories.items():
            if low <= aqi_value <= high:
                return info
        
        # Default to hazardous if above all ranges
        return self.categories[(301, 500)]
    
    def process_pollutant_readings(self, readings: List[PollutantReading]) -> Dict[str, AQICalculation]:
        """Process multiple pollutant readings and calculate AQI for each."""
        aqi_results = {}
        
        # Group readings by pollutant
        pollutant_groups = {}
        for reading in readings:
            if reading.pollutant not in pollutant_groups:
                pollutant_groups[reading.pollutant] = []
            pollutant_groups[reading.pollutant].append(reading)
        
        # Calculate AQI for each pollutant
        for pollutant, pollutant_readings in pollutant_groups.items():
            # Use the most recent reading or average if multiple readings
            if len(pollutant_readings) == 1:
                latest_reading = pollutant_readings[0]
            else:
                # Use reading with highest priority source or most recent
                latest_reading = max(pollutant_readings, 
                                   key=lambda x: (x.timestamp, getattr(x, 'priority', 0)))
            
            # Standardize concentration
            std_concentration, std_unit = self.standardize_concentration(
                pollutant, latest_reading.value, latest_reading.unit
            )
            
            # Calculate AQI
            try:
                aqi_value = self.calculate_aqi_for_pollutant(pollutant, std_concentration)
                category_info = self.get_aqi_category(aqi_value)
                
                aqi_results[pollutant] = AQICalculation(
                    pollutant=pollutant,
                    concentration=std_concentration,
                    unit=std_unit,
                    aqi_value=aqi_value,
                    category=category_info["category"],
                    color=category_info["color"]
                )
            except ValueError as e:
                print(f"Warning: Could not calculate AQI for {pollutant}: {e}")
                continue
        
        return aqi_results
    
    def calculate_overall_aqi(self, aqi_results: Dict[str, AQICalculation]) -> Tuple[int, str, str]:
        """Calculate overall AQI as the maximum of all pollutant AQIs."""
        if not aqi_results:
            return 0, "unknown", "#808080"
        
        # Find the pollutant with the highest AQI
        max_aqi = 0
        dominant_pollutant = ""
        
        for pollutant, result in aqi_results.items():
            if result.aqi_value > max_aqi:
                max_aqi = result.aqi_value
                dominant_pollutant = pollutant
        
        return max_aqi, dominant_pollutant, aqi_results[dominant_pollutant].color
    
    def get_health_advisory(self, aqi_value: int, dominant_pollutant: str) -> str:
        """Get detailed health advisory message."""
        category_info = self.get_aqi_category(aqi_value)
        base_message = category_info["health_message"]
        
        # Add pollutant-specific advice
        pollutant_advice = {
            "pm25": "Fine particles can penetrate deep into lungs and bloodstream.",
            "pm10": "Coarse particles can irritate eyes, nose, and throat.",
            "co": "Carbon monoxide reduces oxygen delivery to body tissues.",
            "o3": "Ground-level ozone can cause respiratory irritation.",
            "no2": "Nitrogen dioxide can irritate airways and reduce lung function.",
            "so2": "Sulfur dioxide can cause breathing difficulties."
        }
        
        if dominant_pollutant in pollutant_advice:
            return f"{base_message} {pollutant_advice[dominant_pollutant]}"
        
        return base_message
    
    def create_aqi_response(self, location, readings: List[PollutantReading]) -> AQIResponse:
        """Create complete AQI response from pollutant readings."""
        # Calculate AQI for each pollutant
        aqi_results = self.process_pollutant_readings(readings)
        
        # Calculate overall AQI
        overall_aqi, dominant_pollutant, overall_color = self.calculate_overall_aqi(aqi_results)
        
        # Get category information
        category_info = self.get_aqi_category(overall_aqi)
        
        # Get health advisory
        health_message = self.get_health_advisory(overall_aqi, dominant_pollutant)
        
        # Extract data sources
        data_sources = list(set(reading.source for reading in readings))
        
        return AQIResponse(
            location=location,
            timestamp=datetime.now(),
            overall_aqi=overall_aqi,
            overall_category=category_info["category"],
            overall_color=overall_color,
            dominant_pollutant=dominant_pollutant,
            health_message=health_message,
            pollutant_details=list(aqi_results.values()),
            raw_readings=readings,
            data_sources=data_sources
        )