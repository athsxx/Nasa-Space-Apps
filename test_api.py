"""
Test script to demonstrate the Air Quality Monitoring Agent API functionality.
"""

import requests
import json
import time


def test_api_endpoint(url, method="GET", data=None, description=""):
    """Test an API endpoint and display results."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {description}")
    print(f"📡 {method} {url}")
    print('='*60)
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ Response (JSON):")
                print(json.dumps(result, indent=2, default=str))
            except:
                print("✅ Response (Text):")
                print(response.text)
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Server might not be running")
        print("   Start the server with: python main.py")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run API tests with various sample data."""
    print("🌍 AIR QUALITY MONITORING AGENT - API TESTING")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test 1: Health Check
    test_api_endpoint(
        f"{base_url}/health",
        description="Health Check Endpoint"
    )
    
    # Test 2: Get AQI for New York City
    test_api_endpoint(
        f"{base_url}/aqi/location/New York, NY",
        description="AQI for New York City"
    )
    
    # Test 3: Get AQI by coordinates (Los Angeles)
    test_api_endpoint(
        f"{base_url}/aqi/coordinates?lat=34.0522&lon=-118.2437",
        description="AQI for Los Angeles (by coordinates)"
    )
    
    # Test 4: Auto-detect location
    test_api_endpoint(
        f"{base_url}/aqi/auto",
        description="AQI for Auto-detected Location"
    )
    
    # Test 5: Health recommendations
    test_api_endpoint(
        f"{base_url}/health/recommendations?aqi=150&sensitive_groups=children,elderly",
        description="Health Recommendations for AQI 150"
    )
    
    # Test 6: Custom location request (POST)
    custom_request = {
        "location": "Paris, France",
        "use_auto_location": False
    }
    test_api_endpoint(
        f"{base_url}/aqi/custom",
        method="POST",
        data=custom_request,
        description="Custom Location Request (Paris)"
    )
    
    # Test 7: AQI Summary
    test_api_endpoint(
        f"{base_url}/aqi/summary/Tokyo, Japan",
        description="AQI Summary for Tokyo"
    )
    
    print("\n" + "="*60)
    print("🎉 API TESTING COMPLETED!")
    print("="*60)
    print("\n📚 Additional Features:")
    print("• Interactive API docs: http://localhost:8000/docs")
    print("• Home page: http://localhost:8000")
    print("• CLI tool: python cli.py --help")
    print("\n💡 The system uses mock data when real APIs are unavailable")
    print("   This ensures the system always works for demonstration purposes!")


if __name__ == "__main__":
    main()