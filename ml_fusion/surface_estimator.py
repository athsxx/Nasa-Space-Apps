"""
Surface Estimator for Real-time Pollutant Concentration Prediction

This module provides the main interface for predicting surface pollutant
concentrations using trained machine learning models with live data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

from .model_trainer import ModelTrainer
from .data_preprocessor import DataPreprocessor

# Try to import AQI calculator, fall back if not available
try:
    from aqi_agent.aqi_calculator import AQICalculator
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from aqi_agent.aqi_calculator import AQICalculator
    except ImportError:
        AQICalculator = None

logger = logging.getLogger(__name__)

class SurfaceEstimator:
    """Main interface for surface pollutant concentration prediction."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Surface Estimator.
        
        Args:
            model_path: Path to trained model file. If None, will look for default.
        """
        self.model_trainer = ModelTrainer()
        self.preprocessor = DataPreprocessor()
        self.aqi_calculator = AQICalculator() if AQICalculator else None
        
        self.is_trained = False
        self.model_path = model_path or "models/surface_estimator.pkl"
        
        # Load models if available
        if Path(self.model_path).exists():
            self.load_models()
    
    def load_models(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained models from file.
        
        Args:
            model_path: Path to model file. Uses default if None.
            
        Returns:
            True if successful, False otherwise
        """
        path_to_use = model_path or self.model_path
        
        if self.model_trainer.load_models(path_to_use):
            self.is_trained = True
            logger.info("Surface estimator models loaded successfully")
            return True
        else:
            logger.warning("Failed to load surface estimator models")
            return False
    
    def predict_surface_concentrations(self,
                                     latitude: float,
                                     longitude: float,
                                     tempo_features: Optional[Dict[str, float]] = None,
                                     weather_features: Optional[Dict[str, float]] = None,
                                     timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Predict surface pollutant concentrations for a specific location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            tempo_features: TEMPO satellite measurements
            weather_features: Weather/meteorological data
            timestamp: Observation timestamp
            
        Returns:
            Dictionary with predictions, confidence, and AQI information
        """
        if not self.is_trained:
            logger.error("Models not loaded. Cannot make predictions.")
            return self._create_error_response("Models not trained")
        
        # Prepare feature vector
        feature_vector = self._prepare_prediction_features(
            latitude, longitude, tempo_features, weather_features, timestamp
        )
        
        if feature_vector is None:
            return self._create_error_response("Failed to prepare features")
        
        # Make predictions
        try:
            predictions = self.model_trainer.predict_surface_concentrations(
                feature_vector.reshape(1, -1),
                return_confidence=True
            )
            
            # Convert predictions to response format
            response = self._format_prediction_response(
                predictions, latitude, longitude, timestamp
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._create_error_response(f"Prediction failed: {str(e)}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained models."""
        if not self.is_trained:
            return pd.DataFrame()
        
        return self.model_trainer.get_feature_importance_summary()
    
    def get_model_accuracy_report(self) -> Dict[str, Any]:
        """Get comprehensive accuracy report with MAE, RMSE, and interpretations."""
        if not self.is_trained:
            return {"error": "Models not trained"}
        
        return self.model_trainer.get_accuracy_report()
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summarized model performance for API responses."""
        if not self.is_trained:
            return {"error": "Models not trained"}
        
        accuracy_report = self.model_trainer.get_accuracy_report()
        
        if "error" in accuracy_report:
            return accuracy_report
        
        # Create simplified summary for API responses
        summary = {
            "model_status": "trained" if self.is_trained else "not_trained",
            "available_pollutants": list(self.model_trainer.models.keys()),
            "overall_performance": accuracy_report.get("overall_summary", {}),
            "accuracy_level": accuracy_report.get("accuracy_interpretation", {}).get("overall_level", "Unknown"),
            "metrics_explanation": accuracy_report.get("accuracy_interpretation", {}).get("metrics_explanation", {})
        }
        
        return summary
    
    def _prepare_prediction_features(self,
                                   latitude: float,
                                   longitude: float,
                                   tempo_features: Optional[Dict[str, float]],
                                   weather_features: Optional[Dict[str, float]],
                                   timestamp: Optional[datetime]) -> Optional[np.ndarray]:
        """Prepare feature vector for prediction."""
        try:
            features = []
            
            # Get expected feature names from trained model
            feature_names = self.model_trainer.feature_names
            
            # Initialize feature dictionary with defaults
            feature_dict = {}
            
            # TEMPO satellite features (normalized)
            tempo_defaults = {
                'no2_column': 3e15,
                'o3_column': 1e18,
                'co_column': 2e17,
                'so2_column': 1e14,
                'hcho_column': 2e15,
                'aerosol_optical_depth': 0.15,
                'cloud_fraction': 0.2
            }
            
            # Update with provided TEMPO features
            if tempo_features:
                for key, value in tempo_features.items():
                    if key in tempo_defaults:
                        feature_dict[key] = value
            
            # Fill missing TEMPO features with defaults
            for key, default_val in tempo_defaults.items():
                if key not in feature_dict:
                    feature_dict[key] = default_val
            
            # Location features
            feature_dict['latitude'] = latitude
            feature_dict['longitude'] = longitude
            feature_dict['elevation'] = 0.0  # Default elevation
            
            # Weather features
            weather_defaults = {
                'temperature': 20.0,  # Default temperature (°C)
                'humidity': 50.0,     # Default humidity (%)
                'wind_speed': 3.0,    # Default wind speed (m/s)
                'wind_direction': 180.0  # Default wind direction (degrees)
            }
            
            if weather_features:
                for key, value in weather_features.items():
                    if key in weather_defaults:
                        feature_dict[key] = value
            
            # Fill missing weather features with defaults
            for key, default_val in weather_defaults.items():
                if key not in feature_dict:
                    feature_dict[key] = default_val
            
            # Temporal features
            if timestamp:
                feature_dict['hour'] = timestamp.hour
                feature_dict['day_of_year'] = timestamp.timetuple().tm_yday
                feature_dict['month'] = timestamp.month
                feature_dict['season'] = (timestamp.month - 1) // 3  # 0=Winter, 1=Spring, 2=Summer, 3=Fall
            else:
                # Current time defaults
                now = datetime.now()
                feature_dict['hour'] = now.hour
                feature_dict['day_of_year'] = now.timetuple().tm_yday
                feature_dict['month'] = now.month
                feature_dict['season'] = (now.month - 1) // 3
            
            # Add log-transformed features
            feature_dict['no2_column_log'] = np.log1p(feature_dict['no2_column'])
            feature_dict['o3_column_log'] = np.log1p(feature_dict['o3_column'])
            feature_dict['co_column_log'] = np.log1p(feature_dict['co_column'])
            
            # Build feature vector in correct order
            for feature_name in feature_names:
                if feature_name in feature_dict:
                    features.append(feature_dict[feature_name])
                else:
                    features.append(0.0)  # Default for missing features
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _format_prediction_response(self,
                                  predictions: Dict[str, Any],
                                  latitude: float,
                                  longitude: float,
                                  timestamp: Optional[datetime]) -> Dict[str, Any]:
        """Format prediction results into response format."""
        response = {
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'timestamp': timestamp.isoformat() if timestamp else datetime.now().isoformat(),
            'pollutant_concentrations': {},
            'confidence_scores': {},
            'aqi_analysis': {},
            'model_info': {
                'method': 'Random Forest Regressor',
                'data_sources': 'TEMPO + OpenAQ + Weather',
                'model_version': '1.0.0'
            }
        }
        
        # Extract predictions and confidence scores
        pred_dict = predictions['predictions']
        conf_dict = predictions.get('confidence', {})
        
        # Format pollutant concentrations
        pollutant_readings = []
        
        for pollutant in pred_dict:
            concentration = float(pred_dict[pollutant][0])  # First (and only) sample
            confidence = float(conf_dict.get(pollutant, [0.0])[0]) if conf_dict else 0.0
            
            # Store concentration and confidence
            response['pollutant_concentrations'][pollutant] = {
                'value': max(0.0, concentration),  # Ensure non-negative
                'unit': 'µg/m³',
                'confidence_std': confidence
            }
            
            response['confidence_scores'][pollutant] = min(100.0, max(0.0, 100 - confidence * 10))
            
            # Create pollutant reading for AQI calculation
            if pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
                pollutant_readings.append({
                    'pollutant': pollutant.upper().replace('25', '2.5').replace('10', '10'),
                    'concentration': max(0.0, concentration),
                    'unit': 'µg/m³'
                })
        
        # Calculate AQI if we have pollutant readings and AQI calculator
        if pollutant_readings and self.aqi_calculator:
            try:
                # Convert to format expected by AQI calculator
                try:
                    from aqi_agent.models import PollutantReading
                except ImportError:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from aqi_agent.models import PollutantReading
                
                readings = []
                for reading in pollutant_readings:
                    readings.append(PollutantReading(
                        pollutant=reading['pollutant'],
                        concentration=reading['concentration'],
                        unit=reading['unit']
                    ))
                
                # Calculate AQI
                aqi_result = self.aqi_calculator.process_pollutant_readings(readings)
                
                if aqi_result:
                    response['aqi_analysis'] = {
                        'aqi_value': aqi_result.aqi_value,
                        'aqi_category': aqi_result.aqi_category,
                        'dominant_pollutant': aqi_result.dominant_pollutant,
                        'health_message': aqi_result.health_message,
                        'color_code': aqi_result.color_code
                    }
                else:
                    response['aqi_analysis'] = {'error': 'AQI calculation failed'}
                    
            except Exception as e:
                logger.warning(f"AQI calculation error: {e}")
                response['aqi_analysis'] = {'error': f'AQI calculation failed: {str(e)}'}
        
        return response
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response format."""
        return {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'method': 'Random Forest Regressor',
                'data_sources': 'TEMPO + OpenAQ + Weather',
                'model_version': '1.0.0'
            }
        }