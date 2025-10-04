"""Demo script showcasing Air Quality Monitoring Agent features."""

import asyncio
import json
from datetime import datetime

from aqi_agent import aqi_agent, LocationRequest, AQICalculator, PollutantReading


async def demo_aqi_calculations():
    """Demonstrate AQI calculation functionality."""
    print("🧮 AQI CALCULATION DEMO")
    print("=" * 40)
    
    calculator = AQICalculator()
    
    # Test different pollutants and concentrations
    test_cases = [
        ("pm25", 25.0, "µg/m³"),  # PM2.5 - Moderate
        ("pm10", 75.0, "µg/m³"),  # PM10 - Moderate
        ("co", 8.0, "ppm"),       # CO - Moderate
        ("o3", 0.080, "ppm"),     # O3 - Unhealthy for Sensitive Groups
        ("no2", 200, "ppb"),      # NO2 - Unhealthy for Sensitive Groups
        ("so2", 150, "ppb"),      # SO2 - Unhealthy for Sensitive Groups
    ]
    
    for pollutant, concentration, unit in test_cases:
        try:
            aqi_value = calculator.calculate_aqi_for_pollutant(pollutant, concentration)
            category_info = calculator.get_aqi_category(aqi_value)
            
            print(f"🔹 {pollutant.upper()}: {concentration} {unit}")
            print(f"   AQI: {aqi_value} ({category_info['category']})")
            print(f"   Color: {category_info['color']}")
            print()
        except Exception as e:
            print(f"❌ Error calculating AQI for {pollutant}: {e}")


def demo_mock_readings():
    """Demonstrate processing of mock pollutant readings."""
    print("📊 MOCK DATA PROCESSING DEMO")
    print("=" * 40)
    
    # Create mock readings simulating real data
    mock_readings = [
        PollutantReading(
            pollutant="pm25",
            value=35.2,
            unit="µg/m³",
            timestamp=datetime.now(),
            source="openaq",
            station_name="Downtown Monitor",
            distance_km=2.5
        ),
        PollutantReading(
            pollutant="pm10",
            value=58.7,
            unit="µg/m³",
            timestamp=datetime.now(),
            source="openaq",
            station_name="Downtown Monitor",
            distance_km=2.5
        ),
        PollutantReading(
            pollutant="o3",
            value=0.075,
            unit="ppm",
            timestamp=datetime.now(),
            source="tempo",
            station_name="Satellite Data",
            distance_km=0.0
        ),
        PollutantReading(
            pollutant="no2",
            value=45.3,
            unit="ppb",
            timestamp=datetime.now(),
            source="openaq",
            station_name="Traffic Monitor",
            distance_km=1.8
        )
    ]
    
    # Process readings
    calculator = AQICalculator()
    aqi_results = calculator.process_pollutant_readings(mock_readings)
    
    print("Raw Readings:")
    for reading in mock_readings:
        print(f"  📍 {reading.pollutant.upper()}: {reading.value} {reading.unit} ({reading.source})")
    
    print("\nAQI Results:")
    for pollutant, result in aqi_results.items():
        print(f"  🎯 {pollutant.upper()}: {result.aqi_value} AQI ({result.category})")
    
    # Calculate overall AQI
    overall_aqi, dominant, _ = calculator.calculate_overall_aqi(aqi_results)
    print(f"\n🏆 Overall AQI: {overall_aqi} (Dominant: {dominant.upper()})")


async def demo_location_resolution():
    """Demonstrate location resolution features."""
    print("📍 LOCATION RESOLUTION DEMO")
    print("=" * 40)
    
    test_locations = [
        LocationRequest(location="New York, NY"),
        LocationRequest(latitude=34.0522, longitude=-118.2437),  # Los Angeles
        LocationRequest(location="Paris, France"),
        LocationRequest(use_auto_location=True),
    ]
    
    for i, request in enumerate(test_locations, 1):
        try:
            print(f"Test {i}:")
            if request.location:
                print(f"  Input: Location name '{request.location}'")
            elif request.latitude and request.longitude:
                print(f"  Input: Coordinates ({request.latitude}, {request.longitude})")
            elif request.use_auto_location:
                print(f"  Input: Auto-detection (IP-based)")
            
            # This would normally resolve the location, but we'll simulate it
            print(f"  Status: Location resolution would occur here")
            print(f"  Note: Actual resolution requires internet connectivity")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()


def demo_health_recommendations():
    """Demonstrate health recommendation generation."""
    print("🏥 HEALTH RECOMMENDATIONS DEMO")
    print("=" * 40)
    
    test_aqi_values = [25, 75, 125, 175, 250, 400]
    
    for aqi in test_aqi_values:
        recommendations = aqi_agent.get_health_recommendations(
            aqi, 
            sensitive_groups=["children", "elderly", "asthma"]
        )
        
        print(f"AQI {aqi}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        print()


def demo_json_output():
    """Demonstrate JSON output format."""
    print("📄 JSON OUTPUT DEMO")
    print("=" * 40)
    
    # Mock AQI response structure
    mock_response = {
        "location": {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "city": "New York City",
            "country": "United States",
            "address": "New York City, NY, United States"
        },
        "timestamp": datetime.now().isoformat(),
        "overall_aqi": 87,
        "overall_category": "Moderate",
        "overall_color": "#FFFF00",
        "dominant_pollutant": "pm25",
        "health_message": "Air quality is acceptable for most people. However, sensitive people may experience minor issues.",
        "pollutant_details": [
            {
                "pollutant": "pm25",
                "concentration": 28.5,
                "unit": "µg/m³",
                "aqi_value": 87,
                "category": "Moderate",
                "color": "#FFFF00"
            }
        ],
        "data_sources": ["openaq", "tempo"]
    }
    
    print("Sample JSON Response:")
    print(json.dumps(mock_response, indent=2))


async def main():
    """Run all demo functions."""
    print("🌍 AIR QUALITY MONITORING AGENT - DEMO")
    print("=" * 50)
    print("This demo showcases the key features of the AQI monitoring system")
    print("Note: Some features require internet connectivity for real data")
    print("=" * 50)
    print()
    
    # Run demos
    await demo_aqi_calculations()
    print()
    
    demo_mock_readings()
    print()
    
    await demo_location_resolution()
    print()
    
    demo_health_recommendations()
    print()
    
    demo_json_output()
    print()
    
    print("🎉 Demo complete!")
    print("💡 To start the web server: python main.py")
    print("💡 To use the CLI: python cli.py --help")


if __name__ == "__main__":
    asyncio.run(main())