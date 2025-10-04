"""FastAPI web application for Air Quality Monitoring Agent."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import os

from aqi_agent import aqi_agent, AQIResponse, LocationRequest

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


if __name__ == "__main__":
    print("🌍 Starting Air Quality Monitoring Agent...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🏠 Home Page: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )