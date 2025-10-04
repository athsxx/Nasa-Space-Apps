"""Core AQI monitoring agent functionality."""

from typing import List, Optional

from .models import AQIResponse, LocationRequest, Location
from .location import location_service
from .data_sources import data_manager
from .aqi_calculator import AQICalculator


class AQIAgent:
    """Main AQI monitoring agent."""
    
    def __init__(self):
        self.location_service = location_service
        self.data_manager = data_manager
        self.aqi_calculator = AQICalculator()
    
    async def get_aqi_for_location(self, location_request: LocationRequest) -> AQIResponse:
        """Get AQI data for a specified location."""
        try:
            # Resolve location
            location = self._resolve_location(location_request)
            
            # Get pollutant data from all sources
            readings = await self.data_manager.get_all_pollutant_data(location)
            
            if not readings:
                raise ValueError("No pollutant data available for this location")
            
            # Calculate AQI
            aqi_response = self.aqi_calculator.create_aqi_response(location, readings)
            
            return aqi_response
            
        except Exception as e:
            raise ValueError(f"Failed to get AQI data: {str(e)}")
    
    def get_aqi_for_coordinates(self, latitude: float, longitude: float) -> AQIResponse:
        """Get AQI data for specific coordinates."""
        request = LocationRequest(
            latitude=latitude,
            longitude=longitude
        )
        return self.get_aqi_for_location(request)
    
    def get_aqi_for_address(self, address: str) -> AQIResponse:
        """Get AQI data for an address."""
        request = LocationRequest(location=address)
        return self.get_aqi_for_location(request)
    
    def get_aqi_auto_location(self) -> AQIResponse:
        """Get AQI data for auto-detected location."""
        request = LocationRequest(use_auto_location=True)
        return self.get_aqi_for_location(request)
    
    def _resolve_location(self, request: LocationRequest) -> Location:
        """Resolve location from request parameters."""
        return self.location_service.resolve_location(
            location_str=request.location,
            latitude=request.latitude,
            longitude=request.longitude,
            use_auto_location=request.use_auto_location
        )
    
    def get_health_recommendations(self, aqi_value: int, 
                                 sensitive_groups: Optional[List[str]] = None) -> List[str]:
        """Get detailed health recommendations based on AQI level."""
        recommendations = []
        
        if aqi_value <= 50:  # Good
            recommendations = [
                "Enjoy outdoor activities",
                "Air quality is ideal for outdoor exercise",
                "No health precautions needed"
            ]
        elif aqi_value <= 100:  # Moderate
            recommendations = [
                "Outdoor activities are generally safe",
                "Sensitive individuals should watch for symptoms",
                "Consider reducing prolonged outdoor exertion"
            ]
        elif aqi_value <= 150:  # Unhealthy for Sensitive Groups
            recommendations = [
                "Sensitive groups should reduce outdoor activities",
                "Children and adults with heart/lung disease should avoid prolonged outdoor exertion",
                "Everyone else can enjoy normal outdoor activities"
            ]
        elif aqi_value <= 200:  # Unhealthy
            recommendations = [
                "Everyone should reduce outdoor activities",
                "Avoid prolonged outdoor exertion",
                "Sensitive groups should avoid all outdoor activities",
                "Consider wearing a mask when outdoors"
            ]
        elif aqi_value <= 300:  # Very Unhealthy
            recommendations = [
                "Everyone should avoid outdoor activities",
                "Stay indoors and keep windows closed",
                "Use air purifiers if available",
                "Wear N95 masks if you must go outside"
            ]
        else:  # Hazardous
            recommendations = [
                "Emergency conditions - stay indoors",
                "Avoid all outdoor activities",
                "Seek medical attention if experiencing symptoms",
                "Use air purifiers and keep all windows closed"
            ]
        
        # Add specific recommendations for sensitive groups
        if sensitive_groups:
            if "children" in sensitive_groups:
                recommendations.append("Keep children indoors during high pollution periods")
            if "elderly" in sensitive_groups:
                recommendations.append("Elderly individuals should monitor their health closely")
            if "asthma" in sensitive_groups:
                recommendations.append("Have rescue inhalers readily available")
        
        return recommendations
    
    def format_aqi_summary(self, aqi_response: AQIResponse) -> str:
        """Format AQI response as human-readable summary."""
        location_str = aqi_response.location.address or f"{aqi_response.location.latitude:.4f}, {aqi_response.location.longitude:.4f}"
        
        summary = f"""
Air Quality Report for {location_str}
{'=' * 50}

Overall AQI: {aqi_response.overall_aqi} ({aqi_response.overall_category})
Dominant Pollutant: {aqi_response.dominant_pollutant.upper()}
Health Advisory: {aqi_response.health_message}

Pollutant Details:
"""
        
        for detail in aqi_response.pollutant_details:
            summary += f"  • {detail.pollutant.upper()}: {detail.aqi_value} AQI ({detail.category})\n"
            summary += f"    Concentration: {detail.concentration:.2f} {detail.unit}\n"
        
        summary += f"\nData Sources: {', '.join(aqi_response.data_sources)}"
        summary += f"\nLast Updated: {aqi_response.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return summary
    
    def get_pollutant_trends(self, location: Location, hours: int = 24) -> dict:
        """Get pollutant trends over time (placeholder for future implementation)."""
        # This would require historical data storage and analysis
        return {
            "message": "Trend analysis not yet implemented",
            "suggestion": "Historical data collection needed for trend analysis"
        }


# Global AQI agent instance
aqi_agent = AQIAgent()