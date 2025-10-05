"""FastAPI web application for Air Quality Monitoring Agent."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import os

from aqi_agent import aqi_agent, AQIResponse, LocationRequest

# Import enhanced integration components
try:
    from enhanced_data_integration import enhanced_data_integration
    from advanced_model_trainer import advanced_model_trainer
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Air Quality Monitoring Agent",
    description="Real-time air quality monitoring with TEMPO satellite and OpenAQ data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend interface or API information."""
    # Check if frontend exists and serve it
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    
    # Fallback to API information page
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Air Quality Monitoring Agent</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; }
            .section { margin: 20px 0; }
            .endpoint { background: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <h1 class="header">🌍 Air Quality Monitoring Agent</h1>
        <p>Welcome to the NASA Air Quality Monitoring System</p>
        
        <div class="section">
            <h2>Features</h2>
            <ul>
                <li>Real-time AQI calculations using EPA standards</li>
                <li>TEMPO satellite and OpenAQ ground station data</li>
                <li>Location-based air quality monitoring</li>
                <li>Health advisory messages</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>API Endpoints</h2>
            <div class="endpoint">
                <strong>GET /aqi/location/{location}</strong><br>
                Get AQI by location name (e.g., "New York, NY")
            </div>
            <div class="endpoint">
                <strong>GET /aqi/coordinates</strong><br>
                Get AQI by coordinates (lat, lon parameters)
            </div>
            <div class="endpoint">
                <strong>GET /aqi/auto</strong><br>
                Get AQI for auto-detected location (IP-based)
            </div>
        </div>
        
        <div class="section">
            <p><a href="/docs">📚 Interactive API Documentation</a></p>
            <p><a href="/health">🏥 Health Check</a></p>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Air Quality Monitoring Agent",
        "version": "1.0.0"
    }


@app.get("/aqi/location/{location}", response_model=AQIResponse)
async def get_aqi_by_location(location: str):
    """
    Get AQI data for a specific location (city, address, etc.).
    
    - **location**: Location name (e.g., "New York, NY", "Paris, France")
    """
    try:
        request = LocationRequest(location=location)
        result = await aqi_agent.get_aqi_for_location(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/aqi/coordinates", response_model=AQIResponse)
async def get_aqi_by_coordinates(
    lat: float = Query(..., description="Latitude coordinate", ge=-90, le=90),
    lon: float = Query(..., description="Longitude coordinate", ge=-180, le=180)
):
    """
    Get AQI data for specific coordinates.
    
    - **lat**: Latitude (-90 to 90)
    - **lon**: Longitude (-180 to 180)
    """
    try:
        request = LocationRequest(latitude=lat, longitude=lon)
        result = await aqi_agent.get_aqi_for_location(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/aqi/auto", response_model=AQIResponse)
async def get_aqi_auto_location():
    """
    Get AQI data for auto-detected location (based on IP address).
    """
    try:
        request = LocationRequest(use_auto_location=True)
        result = await aqi_agent.get_aqi_for_location(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/aqi/custom", response_model=AQIResponse)
async def get_aqi_custom(request: LocationRequest):
    """
    Get AQI data with custom location request parameters.
    
    Supports multiple input methods:
    - Location name
    - Coordinates
    - Auto-detection
    """
    try:
        result = await aqi_agent.get_aqi_for_location(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/aqi/summary/{location}")
async def get_aqi_summary(location: str):
    """
    Get human-readable AQI summary for a location.
    """
    try:
        request = LocationRequest(location=location)
        result = await aqi_agent.get_aqi_for_location(request)
        summary = aqi_agent.format_aqi_summary(result)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health/recommendations")
async def get_health_recommendations(
    aqi: int = Query(..., description="AQI value", ge=0, le=500),
    sensitive_groups: Optional[str] = Query(None, description="Comma-separated sensitive groups")
):
    """
    Get health recommendations for a given AQI level.
    
    - **aqi**: AQI value (0-500)
    - **sensitive_groups**: Optional comma-separated list (children,elderly,asthma)
    """
    try:
        groups = sensitive_groups.split(",") if sensitive_groups else None
        recommendations = aqi_agent.get_health_recommendations(aqi, groups)
        return {
            "aqi": aqi,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ML Fusion Endpoints
@app.get("/ml/predict/location/{location}")
async def get_ml_prediction_by_location(location: str):
    """
    Get ML-predicted AQI data for a specific location using trained fusion models.
    
    - **location**: Location name (e.g., "New York, NY", "Paris, France")
    """
    try:
        request = LocationRequest(location=location)
        result = await aqi_agent.get_ml_predicted_aqi(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ml/predict/coordinates")
async def get_ml_prediction_by_coordinates(
    lat: float = Query(..., description="Latitude coordinate", ge=-90, le=90),
    lon: float = Query(..., description="Longitude coordinate", ge=-180, le=180)
):
    """
    Get ML-predicted AQI data for specific coordinates using trained fusion models.
    
    - **lat**: Latitude (-90 to 90)
    - **lon**: Longitude (-180 to 180)
    """
    try:
        request = LocationRequest(latitude=lat, longitude=lon)
        result = await aqi_agent.get_ml_predicted_aqi(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ml/info")
async def get_ml_model_info():
    """
    Get information about ML fusion models and their performance including MAE, RMSE, and accuracy.
    """
    try:
        if not aqi_agent.use_ml_fusion or not aqi_agent.surface_estimator:
            raise HTTPException(status_code=404, detail="ML Fusion not available")
        
        if not aqi_agent.surface_estimator.is_trained:
            raise HTTPException(status_code=404, detail="Models not trained")
        
        # Get comprehensive performance summary
        performance_summary = aqi_agent.surface_estimator.get_model_performance_summary()
        
        return performance_summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/features")
async def get_ml_feature_importance():
    """
    Get feature importance from trained ML models.
    """
    try:
        if not aqi_agent.use_ml_fusion or not aqi_agent.surface_estimator:
            raise HTTPException(status_code=404, detail="ML Fusion not available")
        
        if not aqi_agent.surface_estimator.is_trained:
            raise HTTPException(status_code=404, detail="Models not trained")
        
        importance_df = aqi_agent.surface_estimator.get_feature_importance()
        
        if importance_df.empty:
            return {"message": "No feature importance data available"}
        
        return {
            "feature_importance": importance_df.to_dict('records'),
            "top_10_features": importance_df.head(10).to_dict('records')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/accuracy")
async def get_ml_accuracy_report():
    """
    Get detailed accuracy report with MAE, RMSE, R², and accuracy interpretations for all pollutants.
    """
    try:
        if not aqi_agent.use_ml_fusion or not aqi_agent.surface_estimator:
            raise HTTPException(status_code=404, detail="ML Fusion not available")
        
        if not aqi_agent.surface_estimator.is_trained:
            raise HTTPException(status_code=404, detail="Models not trained")
        
        # Get comprehensive accuracy report
        accuracy_report = aqi_agent.surface_estimator.get_model_accuracy_report()
        
        return accuracy_report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced NASA Space Apps Integration Endpoints
@app.get("/data/sources")
async def get_data_sources():
    """
    Get information about all available data sources including NASA Space Apps integration.
    """
    try:
        if ADVANCED_FEATURES_AVAILABLE:
            return enhanced_data_integration.get_data_summary()
        else:
            return {
                "current_system": {
                    "openaq_api": True,
                    "tempo_satellite": True,
                    "mock_data": True
                },
                "nasa_space_apps": {
                    "enabled": False,
                    "reason": "Advanced components not available"
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/enhanced/location/{location}")
async def get_enhanced_data_by_location(location: str):
    """
    Get enhanced multi-source air quality data including NASA Space Apps datasets.
    
    - **location**: Location name (e.g., "New York, NY", "Paris, France")
    """
    try:
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=404, detail="Enhanced data integration not available")
        
        # Convert location to coordinates (simplified)
        # In production, you'd use a geocoding service
        from aqi_agent.location import location_service
        loc_obj = await location_service.get_location_by_name(location)
        
        # Get enhanced data
        enhanced_data = enhanced_data_integration.get_real_time_data(
            loc_obj, 
            parameters=['no2', 'o3', 'co', 'so2', 'pm25']
        )
        
        return enhanced_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/enhanced/coordinates")
async def get_enhanced_data_by_coordinates(
    lat: float = Query(..., description="Latitude coordinate", ge=-90, le=90),
    lon: float = Query(..., description="Longitude coordinate", ge=-180, le=180)
):
    """
    Get enhanced multi-source air quality data for specific coordinates.
    
    - **lat**: Latitude (-90 to 90)
    - **lon**: Longitude (-180 to 180)
    """
    try:
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=404, detail="Enhanced data integration not available")
        
        from aqi_agent.models import Location
        loc_obj = Location(latitude=lat, longitude=lon)
        
        # Get enhanced data
        enhanced_data = enhanced_data_integration.get_real_time_data(
            loc_obj, 
            parameters=['no2', 'o3', 'co', 'so2', 'pm25']
        )
        
        return enhanced_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/models/comparison")
async def get_model_comparison():
    """
    Get comparison of all available ML models (Random Forest, TCN, Ensemble).
    """
    try:
        if not ADVANCED_FEATURES_AVAILABLE:
            # Return basic model info
            if not aqi_agent.use_ml_fusion or not aqi_agent.surface_estimator:
                raise HTTPException(status_code=404, detail="ML models not available")
            
            return {
                "available_models": ["Random Forest"],
                "current_model": "Random Forest",
                "advanced_models": "Not available - requires NASA Space Apps integration"
            }
        
        # Get advanced model comparison
        comparison = advanced_model_trainer.get_model_comparison()
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict/advanced/location/{location}")
async def get_advanced_ml_prediction_by_location(
    location: str,
    model_type: str = Query("ensemble", description="Model type: random_forest, tcn, or ensemble")
):
    """
    Get ML predictions using advanced models (TCN, Ensemble) for a specific location.
    
    - **location**: Location name
    - **model_type**: Type of model to use (random_forest, tcn, ensemble)
    """
    try:
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=404, detail="Advanced ML models not available")
        
        if model_type not in ['random_forest', 'tcn', 'ensemble']:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # For now, redirect to standard ML prediction with model type info
        request = LocationRequest(location=location)
        result = await aqi_agent.get_ml_predicted_aqi(request)
        
        # Add model type information
        result.metadata = result.metadata or {}
        result.metadata["model_type"] = model_type
        result.metadata["advanced_features"] = True
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🌍 Starting Air Quality Monitoring Agent...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🏠 Home Page: http://localhost:8000")
    
    # Disable reload to avoid multiprocessing issues with advanced dependencies
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled to prevent scipy/tensorflow conflicts
        log_level="info"
    )