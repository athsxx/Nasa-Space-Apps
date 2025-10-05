"""Core AQI monitoring agent functionality."""

from typing import List, Optional
import logging

from .models import AQIResponse, LocationRequest, Location
from .location import location_service
from .data_sources import data_manager
from .aqi_calculator import AQICalculator

# Try to import ML Fusion components
try:
    from ml_fusion import SurfaceEstimator, WeatherIntegration
    ML_FUSION_AVAILABLE = True
except ImportError:
    ML_FUSION_AVAILABLE = False
    SurfaceEstimator = None
    WeatherIntegration = None

logger = logging.getLogger(__name__)


class AQIAgent:
    """Main AQI monitoring agent with ML fusion capabilities."""
    
    def __init__(self, use_ml_fusion: bool = True):
        self.location_service = location_service
        self.data_manager = data_manager
        self.aqi_calculator = AQICalculator()
        
        # Initialize ML Fusion components if available
        self.use_ml_fusion = use_ml_fusion and ML_FUSION_AVAILABLE
        self.surface_estimator = None
        self.weather_integration = None
        
        if self.use_ml_fusion:
            try:
                self.surface_estimator = SurfaceEstimator()
                self.weather_integration = WeatherIntegration()
                logger.info("ML Fusion components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ML Fusion: {e}")
                self.use_ml_fusion = False
    
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
    
    async def get_ml_predicted_aqi(self, location_request: LocationRequest) -> dict:
        """Get AQI prediction using ML fusion model."""
        if not self.use_ml_fusion or not self.surface_estimator or not self.surface_estimator.is_trained:
            return {
                "error": "ML Fusion not available or models not trained",
                "fallback_message": "Using traditional data sources instead"
            }
        
        try:
            # Resolve location
            location = self._resolve_location(location_request)
            
            # Get TEMPO satellite features (mock for now)
            tempo_features = await self._get_tempo_features(location)
            
            # Get weather features
            weather_features = None
            if self.weather_integration:
                weather_features = self.weather_integration.create_weather_features_for_location(
                    location.latitude, location.longitude
                )
            
            # Make ML prediction
            prediction_result = self.surface_estimator.predict_surface_concentrations(
                latitude=location.latitude,
                longitude=location.longitude,
                tempo_features=tempo_features,
                weather_features=weather_features
            )
            
            if 'error' in prediction_result:
                return prediction_result
            
            # Add traditional AQI calculation for comparison
            try:
                traditional_aqi = await self.get_aqi_for_location(location_request)
                prediction_result['traditional_comparison'] = {
                    'aqi_value': traditional_aqi.overall_aqi,
                    'category': traditional_aqi.overall_category,
                    'dominant_pollutant': traditional_aqi.dominant_pollutant
                }
            except Exception as e:
                logger.warning(f"Failed to get traditional AQI for comparison: {e}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {
                "error": f"ML prediction failed: {str(e)}",
                "fallback_message": "Try using traditional data sources"
            }
    
    async def _get_tempo_features(self, location: Location) -> Optional[dict]:
        """Get TEMPO satellite features for a location (mock implementation)."""
        # This would integrate with actual TEMPO data processing
        # For now, return mock features based on location
        
        # Mock TEMPO features based on location characteristics
        mock_features = {
            'no2_column': 5e15,  # molecules/cm²
            'o3_column': 8e17,   # molecules/cm²
            'co_column': 4e17,   # molecules/cm²
            'so2_column': 1e14,  # molecules/cm²
            'hcho_column': 3e15, # molecules/cm²
            'aerosol_optical_depth': 0.2,
            'cloud_fraction': 0.1
        }
        
        # Add some location-based variation
        import random
        if location.latitude > 40:  # Northern cities - potentially higher pollution
            mock_features['no2_column'] *= random.uniform(1.2, 2.0)
        
        if abs(location.longitude) > 100:  # Western US - different pollution profile
            mock_features['co_column'] *= random.uniform(0.8, 1.5)
        
        return mock_features
    
    def get_ml_model_info(self) -> dict:
        """Get information about ML fusion models."""
        if not self.use_ml_fusion:
            return {"status": "ML Fusion not available"}
        
        if not self.surface_estimator:
            return {"status": "Surface estimator not initialized"}
        
        if not self.surface_estimator.is_trained:
            return {
                "status": "Models not trained",
                "suggestion": "Run 'python train_ml_fusion.py' to train models"
            }
        
        # Get model performance info
        performance = self.surface_estimator.get_model_performance()
        feature_importance = self.surface_estimator.get_feature_importance()
        
        return {
            "status": "ML Fusion available",
            "models_trained": True,
            "performance": performance,
            "top_features": feature_importance.head(5).to_dict('records') if not feature_importance.empty else []
        }


# Global AQI agent instance
aqi_agent = AQIAgent()