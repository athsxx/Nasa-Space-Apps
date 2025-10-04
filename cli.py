"""Command-line interface for Air Quality Monitoring Agent."""

import argparse
import asyncio
import json
import sys

from aqi_agent import aqi_agent, LocationRequest


def print_aqi_report(aqi_response):
    """Print formatted AQI report to console."""
    print("\n" + "="*60)
    print("🌍 AIR QUALITY MONITORING REPORT")
    print("="*60)
    
    # Location information
    location = aqi_response.location
    location_str = location.address or f"{location.latitude:.4f}, {location.longitude:.4f}"
    print(f"📍 Location: {location_str}")
    
    # Overall AQI
    aqi_color = get_aqi_color_emoji(aqi_response.overall_aqi)
    print(f"\n{aqi_color} Overall AQI: {aqi_response.overall_aqi}")
    print(f"   Category: {aqi_response.overall_category}")
    print(f"   Dominant Pollutant: {aqi_response.dominant_pollutant.upper()}")
    
    # Health message
    print(f"\n💡 Health Advisory:")
    print(f"   {aqi_response.health_message}")
    
    # Pollutant details
    print(f"\n📊 Pollutant Details:")
    for detail in aqi_response.pollutant_details:
        emoji = get_aqi_color_emoji(detail.aqi_value)
        print(f"   {emoji} {detail.pollutant.upper()}: {detail.aqi_value} AQI ({detail.category})")
        print(f"      Concentration: {detail.concentration:.2f} {detail.unit}")
    
    # Data sources
    print(f"\n📡 Data Sources: {', '.join(aqi_response.data_sources)}")
    print(f"⏰ Last Updated: {aqi_response.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Health recommendations
    recommendations = aqi_agent.get_health_recommendations(aqi_response.overall_aqi)
    if recommendations:
        print(f"\n🏥 Health Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("="*60)


def get_aqi_color_emoji(aqi_value: int) -> str:
    """Get emoji representing AQI level."""
    if aqi_value <= 50:
        return "🟢"  # Good - Green
    elif aqi_value <= 100:
        return "🟡"  # Moderate - Yellow
    elif aqi_value <= 150:
        return "🟠"  # Unhealthy for Sensitive Groups - Orange
    elif aqi_value <= 200:
        return "🔴"  # Unhealthy - Red
    elif aqi_value <= 300:
        return "🟣"  # Very Unhealthy - Purple
    else:
        return "🔴"  # Hazardous - Maroon (using red as fallback)


async def main():
    """Main CLI application."""
    parser = argparse.ArgumentParser(
        description="Air Quality Monitoring Agent - Get real-time AQI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --location "New York, NY"
  %(prog)s --lat 40.7128 --lon -74.0060
  %(prog)s --auto
  %(prog)s --location "Paris" --json
        """
    )
    
    # Location options (mutually exclusive)
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument(
        "--location", "-l",
        type=str,
        help="Location name (city, address, etc.)"
    )
    location_group.add_argument(
        "--coordinates", "-c",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        help="Latitude and longitude coordinates"
    )
    location_group.add_argument(
        "--lat",
        type=float,
        help="Latitude coordinate (use with --lon)"
    )
    location_group.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Auto-detect location from IP address"
    )
    
    # Optional longitude for --lat
    parser.add_argument(
        "--lon",
        type=float,
        help="Longitude coordinate (use with --lat)"
    )
    
    # Output options
    parser.add_argument(
        "--json", "-j",
        action="store_true", 
        help="Output in JSON format"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show summary format"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress banner and extra information"
    )
    
    args = parser.parse_args()
    
    # Validate coordinate arguments
    if args.lat is not None and args.lon is None:
        parser.error("--lat requires --lon")
    if args.lon is not None and args.lat is None:
        parser.error("--lon requires --lat")
    
    try:
        # Create location request
        if args.location:
            request = LocationRequest(location=args.location)
        elif args.coordinates:
            lat, lon = args.coordinates
            request = LocationRequest(latitude=lat, longitude=lon)
        elif args.lat is not None and args.lon is not None:
            request = LocationRequest(latitude=args.lat, longitude=args.lon)
        elif args.auto:
            request = LocationRequest(use_auto_location=True)
        else:
            parser.error("No location specified")
        
        # Show loading message
        if not args.quiet:
            print("🔍 Fetching air quality data...")
        
        # Get AQI data
        aqi_response = await aqi_agent.get_aqi_for_location(request)
        
        # Output results
        if args.json:
            # JSON output
            print(json.dumps(aqi_response.dict(), indent=2, default=str))
        elif args.summary:
            # Summary format
            summary = aqi_agent.format_aqi_summary(aqi_response)
            print(summary)
        else:
            # Default formatted output
            print_aqi_report(aqi_response)
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def run_cli():
    """Entry point for CLI application."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    run_cli()