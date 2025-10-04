"""Unit tests for the AQI monitoring agent."""

import unittest
from unittest.mock import Mock
from datetime import datetime

from aqi_agent import AQICalculator, Location, PollutantReading
from config.settings import AQI_BREAKPOINTS


class TestAQICalculator(unittest.TestCase):
    """Test cases for AQI calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = AQICalculator()
        self.test_location = Location(
            latitude=40.7128,
            longitude=-74.0060,
            city="New York City",
            country="United States",
            address="New York City, NY, United States"
        )
    
    def test_pm25_aqi_calculation(self):
        """Test PM2.5 AQI calculation."""
        # Test Good range (0-50 AQI)
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 10.0)
        self.assertGreaterEqual(aqi, 0)
        self.assertLessEqual(aqi, 50)
        
        # Test Moderate range (51-100 AQI)
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 25.0)
        self.assertGreaterEqual(aqi, 51)
        self.assertLessEqual(aqi, 100)
        
        # Test Unhealthy range (151-200 AQI)
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 100.0)
        self.assertGreaterEqual(aqi, 151)
        self.assertLessEqual(aqi, 200)
    
    def test_aqi_formula_accuracy(self):
        """Test AQI formula accuracy with known values."""
        # EPA example: PM2.5 = 55.5 µg/m³ should give AQI = 151
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 55.5)
        self.assertEqual(aqi, 151)
        
        # Test boundary values
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 12.0)
        self.assertEqual(aqi, 50)
        
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 35.4)
        self.assertEqual(aqi, 100)
    
    def test_co_aqi_calculation(self):
        """Test CO AQI calculation."""
        # Test Good range
        aqi = self.calculator.calculate_aqi_for_pollutant("co", 2.0)
        self.assertGreaterEqual(aqi, 0)
        self.assertLessEqual(aqi, 50)
        
        # Test Moderate range
        aqi = self.calculator.calculate_aqi_for_pollutant("co", 7.0)
        self.assertGreaterEqual(aqi, 51)
        self.assertLessEqual(aqi, 100)
    
    def test_invalid_pollutant(self):
        """Test handling of invalid pollutant names."""
        with self.assertRaises(ValueError):
            self.calculator.calculate_aqi_for_pollutant("invalid_pollutant", 50.0)
    
    def test_extreme_values(self):
        """Test handling of extreme concentration values."""
        # Very low value
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 0.0)
        self.assertEqual(aqi, 0)
        
        # Very high value (should be capped at 500)
        aqi = self.calculator.calculate_aqi_for_pollutant("pm25", 1000.0)
        self.assertLessEqual(aqi, 500)
    
    def test_aqi_category_mapping(self):
        """Test AQI category assignment."""
        # Good
        category = self.calculator.get_aqi_category(25)
        self.assertEqual(category["category"], "Good")
        
        # Moderate
        category = self.calculator.get_aqi_category(75)
        self.assertEqual(category["category"], "Moderate")
        
        # Unhealthy
        category = self.calculator.get_aqi_category(175)
        self.assertEqual(category["category"], "Unhealthy")
        
        # Hazardous
        category = self.calculator.get_aqi_category(450)
        self.assertEqual(category["category"], "Hazardous")
    
    def test_unit_conversion(self):
        """Test pollutant unit conversions."""
        # Test CO conversion (mg/m³ to ppm)
        converted, unit = self.calculator.standardize_concentration("co", 10.0, "mg/m³")
        self.assertAlmostEqual(converted, 8.73, places=2)
        self.assertEqual(unit, "ppm")
        
        # Test same unit (no conversion)
        converted, unit = self.calculator.standardize_concentration("pm25", 25.0, "µg/m³")
        self.assertEqual(converted, 25.0)
        self.assertEqual(unit, "µg/m³")
    
    def test_pollutant_readings_processing(self):
        """Test processing of multiple pollutant readings."""
        # Create mock readings
        readings = [
            PollutantReading(
                pollutant="pm25",
                value=25.0,
                unit="µg/m³",
                timestamp=datetime.now(),
                source="openaq",
                station_name="Test Station"
            ),
            PollutantReading(
                pollutant="pm10",
                value=50.0,
                unit="µg/m³",
                timestamp=datetime.now(),
                source="openaq",
                station_name="Test Station"
            )
        ]
        
        aqi_results = self.calculator.process_pollutant_readings(readings)
        
        # Check that both pollutants were processed
        self.assertIn("pm25", aqi_results)
        self.assertIn("pm10", aqi_results)
        
        # Check AQI values are reasonable
        pm25_aqi = aqi_results["pm25"].aqi_value
        pm10_aqi = aqi_results["pm10"].aqi_value
        
        self.assertGreater(pm25_aqi, 0)
        self.assertGreater(pm10_aqi, 0)
        self.assertLessEqual(pm25_aqi, 500)
        self.assertLessEqual(pm10_aqi, 500)
    
    def test_overall_aqi_calculation(self):
        """Test overall AQI calculation (maximum of all pollutants)."""
        # Mock AQI results
        aqi_results = {
            "pm25": Mock(aqi_value=75, pollutant="pm25"),
            "pm10": Mock(aqi_value=50, pollutant="pm10"),
            "o3": Mock(aqi_value=120, pollutant="o3", color="#FF7E00")
        }
        
        overall_aqi, dominant, color = self.calculator.calculate_overall_aqi(aqi_results)
        
        # Overall AQI should be the maximum (120 from O3)
        self.assertEqual(overall_aqi, 120)
        self.assertEqual(dominant, "o3")
        self.assertEqual(color, "#FF7E00")
    
    def test_health_advisory_generation(self):
        """Test health advisory message generation."""
        # Test different AQI levels
        message = self.calculator.get_health_advisory(25, "pm25")
        self.assertIn("satisfactory", message.lower())
        
        message = self.calculator.get_health_advisory(150, "o3")
        self.assertIn("sensitive", message.lower())
        
        message = self.calculator.get_health_advisory(250, "pm25")
        self.assertIn("emergency", message.lower())


class TestAQIBreakpoints(unittest.TestCase):
    """Test cases for AQI breakpoint data integrity."""
    
    def test_breakpoint_completeness(self):
        """Test that all required pollutants have breakpoints."""
        required_pollutants = ["pm25", "pm10", "co", "o3", "no2", "so2"]
        
        for pollutant in required_pollutants:
            self.assertIn(pollutant, AQI_BREAKPOINTS, f"Missing breakpoints for {pollutant}")
    
    def test_breakpoint_ranges(self):
        """Test that breakpoint ranges are valid and continuous."""
        for pollutant, breakpoints in AQI_BREAKPOINTS.items():
            # Check that ranges are ascending
            prev_c_high = -1
            prev_aqi_high = -1
            
            for c_low, c_high, aqi_low, aqi_high in breakpoints:
                # Concentration ranges should be ascending
                self.assertGreater(c_high, c_low, f"Invalid concentration range for {pollutant}")
                self.assertGreater(c_low, prev_c_high, f"Non-continuous ranges for {pollutant}")
                
                # AQI ranges should be ascending
                self.assertGreater(aqi_high, aqi_low, f"Invalid AQI range for {pollutant}")
                self.assertGreater(aqi_low, prev_aqi_high, f"Non-continuous AQI ranges for {pollutant}")
                
                prev_c_high = c_high
                prev_aqi_high = aqi_high


if __name__ == "__main__":
    unittest.main()