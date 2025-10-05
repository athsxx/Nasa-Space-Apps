"""
Data Fusion and Surface Estimation Module

This module provides machine learning capabilities to predict surface pollutant
concentrations by fusing TEMPO satellite data with OpenAQ ground station data.
"""

from .data_preprocessor import DataPreprocessor
from .spatial_temporal_fusion import SpatialTemporalFusion
from .surface_estimator import SurfaceEstimator
from .model_trainer import ModelTrainer
from .weather_integration import WeatherIntegration

__version__ = "1.0.0"
__all__ = [
    "DataPreprocessor",
    "SpatialTemporalFusion", 
    "SurfaceEstimator",
    "ModelTrainer",
    "WeatherIntegration"
]