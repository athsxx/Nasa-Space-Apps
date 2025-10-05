"""
ML Fusion Training Pipeline

This script demonstrates the complete data fusion and surface estimation 
training pipeline using TEMPO satellite and OpenAQ ground station data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse

from ml_fusion import (
    DataPreprocessor,
    SpatialTemporalFusion,
    ModelTrainer,
    SurfaceEstimator,
    WeatherIntegration
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_fusion_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MLFusionTrainingPipeline:
    """Complete training pipeline for surface concentration prediction."""
    
    def __init__(self, config: dict):
        """Initialize the training pipeline."""
        self.config = config
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.fusion = SpatialTemporalFusion(
            spatial_threshold_km=config.get('spatial_threshold_km', 25.0),
            temporal_threshold_hours=config.get('temporal_threshold_hours', 1.0)
        )
        self.trainer = ModelTrainer(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 20),
            random_state=config.get('random_state', 42)
        )
        self.weather = WeatherIntegration()
        
        # Paths
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
    
    def generate_synthetic_training_data(self, n_tempo_samples: int = 1000, 
                                       n_openaq_samples: int = 500) -> tuple:
        """
        Generate synthetic training data for demonstration.
        
        Args:
            n_tempo_samples: Number of TEMPO satellite observations
            n_openaq_samples: Number of OpenAQ ground station measurements
            
        Returns:
            Tuple of (tempo_df, openaq_df)
        """
        logger.info("Generating synthetic training data...")
        
        # Generate TEMPO satellite data
        tempo_data = self._generate_synthetic_tempo_data(n_tempo_samples)
        
        # Generate OpenAQ ground station data
        openaq_data = self._generate_synthetic_openaq_data(n_openaq_samples)
        
        logger.info(f"Generated {len(tempo_data)} TEMPO and {len(openaq_data)} OpenAQ samples")
        return tempo_data, openaq_data
    
    def _generate_synthetic_tempo_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic TEMPO satellite observations."""
        np.random.seed(42)
        
        # Generate realistic locations (focus on populated areas)
        major_cities = [
            (40.7128, -74.0060),  # New York
            (34.0522, -118.2437), # Los Angeles  
            (41.8781, -87.6298),  # Chicago
            (29.7604, -95.3698),  # Houston
            (33.4484, -112.0740), # Phoenix
            (39.7392, -104.9903), # Denver
            (47.6062, -122.3321), # Seattle
            (25.7617, -80.1918),  # Miami
        ]
        
        data = []
        for i in range(n_samples):
            # Choose random city and add noise
            city_lat, city_lon = major_cities[np.random.randint(len(major_cities))]
            lat = city_lat + np.random.normal(0, 0.5)  # ~50km radius
            lon = city_lon + np.random.normal(0, 0.5)
            
            # Generate observation time
            base_date = datetime(2023, 1, 1)
            days_offset = np.random.randint(0, 365)
            hours_offset = np.random.randint(0, 24)
            obs_time = base_date + timedelta(days=days_offset, hours=hours_offset)
            
            # Generate realistic satellite measurements
            # NO2 column density (molecules/cm²)
            no2_base = np.random.lognormal(15.5, 0.5)  # ~5e15 molecules/cm²
            
            # O3 column density  
            o3_base = np.random.lognormal(18.0, 0.3)   # ~8e17 molecules/cm²
            
            # CO column density
            co_base = np.random.lognormal(17.5, 0.4)   # ~4e17 molecules/cm²
            
            # SO2 column density
            so2_base = np.random.lognormal(14.0, 0.6)  # ~1e14 molecules/cm²
            
            # HCHO column density
            hcho_base = np.random.lognormal(15.0, 0.5) # ~3e15 molecules/cm²
            
            # Aerosol optical depth
            aod = np.random.beta(2, 5) * 0.5  # 0-0.5 range
            
            # Cloud fraction
            cloud_frac = np.random.beta(2, 3)  # 0-1 range
            
            data.append({
                'latitude': lat,
                'longitude': lon,
                'observation_time': obs_time,
                'no2_column': no2_base,
                'o3_column': o3_base,
                'co_column': co_base,
                'so2_column': so2_base,
                'hcho_column': hcho_base,
                'aerosol_optical_depth': aod,
                'cloud_fraction': cloud_frac
            })
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_openaq_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic OpenAQ ground station measurements."""
        np.random.seed(43)
        
        # Ground station locations (subset of cities with monitors)
        stations = [
            (40.7128, -74.0060, 10),   # New York, 10m elevation
            (34.0522, -118.2437, 71), # Los Angeles, 71m
            (41.8781, -87.6298, 179), # Chicago, 179m
            (29.7604, -95.3698, 13),  # Houston, 13m
            (33.4484, -112.0740, 331), # Phoenix, 331m
        ]
        
        data = []
        for i in range(n_samples):
            # Choose random station
            station_lat, station_lon, elevation = stations[np.random.randint(len(stations))]
            
            # Add small noise to station location
            lat = station_lat + np.random.normal(0, 0.01)  # ~1km radius
            lon = station_lon + np.random.normal(0, 0.01)
            
            # Generate timestamp
            base_date = datetime(2023, 1, 1)
            days_offset = np.random.randint(0, 365)
            hours_offset = np.random.randint(0, 24)
            timestamp = base_date + timedelta(days=days_offset, hours=hours_offset)
            
            # Generate realistic pollutant concentrations (µg/m³)
            # PM2.5: typically 5-50 µg/m³
            pm25 = np.random.lognormal(2.5, 0.6)
            
            # PM10: typically 10-100 µg/m³
            pm10 = pm25 * np.random.uniform(1.5, 3.0)
            
            # O3: typically 20-150 µg/m³
            o3 = np.random.lognormal(3.8, 0.5)
            
            # NO2: typically 10-100 µg/m³
            no2 = np.random.lognormal(3.0, 0.6)
            
            # SO2: typically 1-50 µg/m³
            so2 = np.random.lognormal(1.5, 0.8)
            
            # CO: typically 1-10 mg/m³ = 1000-10000 µg/m³
            co = np.random.lognormal(7.0, 0.5)
            
            # Weather data
            temp = 20 + 10 * np.sin(2 * np.pi * timestamp.timetuple().tm_yday / 365) + np.random.normal(0, 5)
            humidity = np.random.uniform(30, 80)
            wind_speed = np.random.exponential(3)
            wind_direction = np.random.uniform(0, 360)
            
            data.append({
                'latitude': lat,
                'longitude': lon,
                'elevation': elevation,
                'timestamp': timestamp,
                'pm25': pm25,
                'pm10': pm10,
                'o3': o3,
                'no2': no2,
                'so2': so2,
                'co': co,
                'temperature': temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction
            })
        
        return pd.DataFrame(data)
    
    def run_training_pipeline(self):
        """Execute the complete training pipeline."""
        logger.info("Starting ML Fusion Training Pipeline")
        
        # Step 1: Generate or load data
        logger.info("Step 1: Loading/generating training data")
        tempo_df, openaq_df = self.generate_synthetic_training_data(
            n_tempo_samples=self.config.get('n_tempo_samples', 2000),
            n_openaq_samples=self.config.get('n_openaq_samples', 1000)
        )
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data")
        tempo_processed = self.preprocessor.preprocess_tempo_data(tempo_df)
        openaq_processed = self.preprocessor.preprocess_openaq_data(openaq_df)
        
        # Step 3: Spatial-temporal fusion
        logger.info("Step 3: Finding collocated data pairs")
        
        # Create hourly averages for better matching
        tempo_hourly, openaq_hourly = self.fusion.create_hourly_averages(
            tempo_processed, openaq_processed
        )
        
        # Find collocated pairs
        matches = self.fusion.find_collocated_pairs(tempo_hourly, openaq_hourly)
        
        if len(matches) < 50:
            logger.warning(f"Only {len(matches)} matches found. Consider relaxing matching criteria.")
            return None
        
        # Validate matches
        validated_matches = self.fusion.validate_matches(matches)
        
        # Step 4: Create feature matrix
        logger.info("Step 4: Creating feature matrix")
        
        # Extract matched pairs
        matched_pairs = [(tempo_idx, openaq_idx) for tempo_idx, openaq_idx, _, _ in validated_matches]
        
        # Create features and targets
        X, y = self.preprocessor.create_feature_matrix(
            tempo_hourly, openaq_hourly, matched_pairs
        )
        
        if X.shape[0] < 50:
            logger.error("Insufficient training samples. Pipeline terminated.")
            return None
        
        # Step 5: Train models
        logger.info("Step 5: Training Random Forest models")
        
        # Prepare data splits
        feature_names = self.preprocessor.get_feature_names()
        target_names = self.preprocessor.get_target_names()
        
        data_splits = self.trainer.prepare_data(X, y, feature_names, target_names)
        
        # Train models
        training_results = self.trainer.train_models(
            data_splits, 
            optimize_hyperparameters=self.config.get('optimize_hyperparameters', False)
        )
        
        # Step 6: Evaluate models
        logger.info("Step 6: Evaluating models")
        evaluation_results = self.trainer.evaluate_models(data_splits)
        
        # Step 7: Save models
        logger.info("Step 7: Saving trained models")
        model_path = self.trainer.save_models(str(self.model_dir))
        
        # Step 8: Generate report
        self._generate_training_report(training_results, evaluation_results, validated_matches)
        
        logger.info(f"Training pipeline completed. Models saved to: {model_path}")
        return model_path
    
    def _generate_training_report(self, training_results, evaluation_results, matches):
        """Generate training report."""
        report_path = self.model_dir / "training_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# ML Fusion Training Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Training Configuration\n")
            for key, value in self.config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            f.write("## Data Fusion Summary\n")
            f.write(f"- Collocated pairs found: {len(matches)}\n")
            f.write(f"- Training samples: {len(matches)}\n\n")
            
            f.write("## Model Performance\n")
            for pollutant, results in evaluation_results.items():
                metrics = results['test_metrics']
                f.write(f"### {pollutant.upper()}\n")
                f.write(f"- R² Score: {metrics['r2']:.3f}\n")
                f.write(f"- MAE: {metrics['mae']:.3f} µg/m³\n")
                f.write(f"- RMSE: {metrics['rmse']:.3f} µg/m³\n")
                f.write(f"- Test samples: {results['n_test_samples']}\n\n")
            
            f.write("## Feature Importance\n")
            importance_df = self.trainer.get_feature_importance_summary()
            if not importance_df.empty:
                f.write("Top 10 most important features:\n")
                for idx, row in importance_df.head(10).iterrows():
                    f.write(f"- {row['feature']}: {row['importance']:.3f}\n")
        
        logger.info(f"Training report saved to: {report_path}")

def main():
    """Main function to run training pipeline."""
    parser = argparse.ArgumentParser(description='ML Fusion Training Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--n-tempo', type=int, default=2000, help='Number of TEMPO samples')
    parser.add_argument('--n-openaq', type=int, default=1000, help='Number of OpenAQ samples')
    parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'n_tempo_samples': args.n_tempo,
        'n_openaq_samples': args.n_openaq,
        'spatial_threshold_km': 25.0,
        'temporal_threshold_hours': 1.0,
        'n_estimators': 200,
        'max_depth': 20,
        'random_state': 42,
        'optimize_hyperparameters': args.optimize,
        'data_dir': 'data',
        'model_dir': 'models'
    }
    
    # Run pipeline
    pipeline = MLFusionTrainingPipeline(config)
    model_path = pipeline.run_training_pipeline()
    
    if model_path:
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Models saved to: {model_path}")
        print(f"📊 Check training_report.md for detailed results")
        
        # Test the trained model
        estimator = SurfaceEstimator(model_path)
        if estimator.is_trained:
            print("\n🔬 Testing trained model...")
            
            # Test prediction for New York
            test_result = estimator.predict_surface_concentrations(
                latitude=40.7128,
                longitude=-74.0060,
                tempo_features={
                    'no2_column': 5e15,
                    'o3_column': 8e17,
                    'co_column': 4e17,
                    'aerosol_optical_depth': 0.2,
                    'cloud_fraction': 0.1
                },
                weather_features={
                    'temperature': 22.0,
                    'humidity': 60.0,
                    'wind_speed': 3.5,
                    'wind_direction': 180.0
                }
            )
            
            if 'error' not in test_result:
                print("✅ Test prediction successful!")
                print(f"🌍 Location: New York ({test_result['location']['latitude']:.4f}, {test_result['location']['longitude']:.4f})")
                
                if 'aqi_analysis' in test_result and 'aqi_value' in test_result['aqi_analysis']:
                    aqi_info = test_result['aqi_analysis']
                    print(f"🏭 AQI: {aqi_info['aqi_value']} ({aqi_info['aqi_category']})")
                    print(f"🎯 Dominant pollutant: {aqi_info['dominant_pollutant']}")
                
                print("💨 Predicted concentrations:")
                for pollutant, conc_info in test_result['pollutant_concentrations'].items():
                    confidence = test_result['confidence_scores'].get(pollutant, 0)
                    print(f"  - {pollutant.upper()}: {conc_info['value']:.2f} {conc_info['unit']} (confidence: {confidence:.0f}%)")
            else:
                print(f"❌ Test prediction failed: {test_result['error']}")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()