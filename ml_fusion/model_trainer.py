"""
Model Trainer for Surface Pollutant Concentration Prediction

This module implements the Random Forest Regressor training pipeline
for predicting surface pollutant concentrations from satellite and ground data.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from typing import Tuple, Dict, List, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and evaluates Random Forest models for surface concentration prediction."""
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: int = 20,
                 random_state: int = 42,
                 test_size: float = 0.2,
                 validation_size: float = 0.1):
        """
        Initialize the model trainer.
        
        Args:
            n_estimators: Number of trees in random forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Models for each pollutant
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_names = []
        
        # Performance metrics
        self.training_metrics = {}
        self.feature_importance = {}
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    feature_names: List[str], target_names: List[str]) -> Dict[str, Any]:
        """
        Prepare data for training by splitting and scaling.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target matrix (n_samples, n_targets)
            feature_names: Names of features
            target_names: Names of target pollutants
            
        Returns:
            Dictionary containing train/validation/test splits
        """
        logger.info(f"Preparing data: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
        
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Remove samples with missing targets
        complete_samples = ~np.isnan(y).any(axis=1)
        X_clean = X[complete_samples]
        y_clean = y[complete_samples]
        
        logger.info(f"After removing incomplete samples: {X_clean.shape[0]} samples")
        
        # Split into train/validation/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_clean, y_clean, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['feature_scaler'] = scaler
        
        data_splits = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_raw': X_train,
            'X_val_raw': X_val,
            'X_test_raw': X_test
        }
        
        logger.info(f"Data splits - Train: {X_train_scaled.shape[0]}, "
                   f"Val: {X_val_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}")
        
        return data_splits
    
    def train_models(self, data_splits: Dict[str, Any], 
                    optimize_hyperparameters: bool = False) -> Dict[str, Dict]:
        """
        Train separate Random Forest models for each pollutant.
        
        Args:
            data_splits: Data splits from prepare_data
            optimize_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Training results and metrics
        """
        logger.info("Training Random Forest models for each pollutant...")
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        results = {}
        
        # Train a model for each pollutant
        for i, pollutant in enumerate(self.target_names):
            logger.info(f"Training model for {pollutant}...")
            
            # Extract target for this pollutant
            y_train_pollutant = y_train[:, i]
            y_val_pollutant = y_val[:, i]
            
            # Remove samples with missing target values
            valid_train = ~np.isnan(y_train_pollutant)
            valid_val = ~np.isnan(y_val_pollutant)
            
            if np.sum(valid_train) < 10:
                logger.warning(f"Not enough training samples for {pollutant}: {np.sum(valid_train)}")
                continue
            
            X_train_valid = X_train[valid_train]
            y_train_valid = y_train_pollutant[valid_train]
            X_val_valid = X_val[valid_val]
            y_val_valid = y_val_pollutant[valid_val]
            
            # Initialize model
            if optimize_hyperparameters:
                model = self._optimize_hyperparameters(X_train_valid, y_train_valid)
            else:
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            
            # Train model
            model.fit(X_train_valid, y_train_valid)
            
            # Store model
            self.models[pollutant] = model
            
            # Evaluate on validation set
            if len(X_val_valid) > 0:
                val_predictions = model.predict(X_val_valid)
                val_metrics = self._calculate_metrics(y_val_valid, val_predictions)
            else:
                val_metrics = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan}
            
            # Feature importance
            self.feature_importance[pollutant] = dict(zip(
                self.feature_names, 
                model.feature_importances_
            ))
            
            results[pollutant] = {
                'model': model,
                'validation_metrics': val_metrics,
                'n_train_samples': len(X_train_valid),
                'n_val_samples': len(X_val_valid)
            }
            
            logger.info(f"{pollutant} - Train samples: {len(X_train_valid)}, "
                       f"Validation R²: {val_metrics['r2']:.3f}, "
                       f"RMSE: {val_metrics['rmse']:.3f}")
        
        return results
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Optimize hyperparameters using GridSearchCV."""
        logger.info("Optimizing hyperparameters...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=3, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def evaluate_models(self, data_splits: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Evaluate trained models on test set.
        
        Args:
            data_splits: Data splits from prepare_data
            
        Returns:
            Test evaluation results
        """
        logger.info("Evaluating models on test set...")
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        evaluation_results = {}
        
        for i, pollutant in enumerate(self.target_names):
            if pollutant not in self.models:
                continue
                
            model = self.models[pollutant]
            y_test_pollutant = y_test[:, i]
            
            # Remove samples with missing target values
            valid_test = ~np.isnan(y_test_pollutant)
            
            if np.sum(valid_test) == 0:
                logger.warning(f"No valid test samples for {pollutant}")
                continue
            
            X_test_valid = X_test[valid_test]
            y_test_valid = y_test_pollutant[valid_test]
            
            # Make predictions
            predictions = model.predict(X_test_valid)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test_valid, predictions)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_test_valid, y_test_valid, 
                cv=5, scoring='r2'
            )
            
            evaluation_results[pollutant] = {
                'test_metrics': metrics,
                'cv_mean_r2': cv_scores.mean(),
                'cv_std_r2': cv_scores.std(),
                'n_test_samples': len(X_test_valid),
                'predictions': predictions,
                'actual': y_test_valid
            }
            
            logger.info(f"{pollutant} Test Results - R²: {metrics['r2']:.3f}, "
                       f"MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}")
        
        self.training_metrics = evaluation_results
        return evaluation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate additional accuracy metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Mean Absolute Percentage Error
        
        # Calculate accuracy based on R²
        accuracy_percentage = max(0, min(100, r2 * 100))  # Convert R² to percentage (0-100%)
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy_percentage': accuracy_percentage
        }
    
    def predict_surface_concentrations(self, 
                                     features: np.ndarray,
                                     return_confidence: bool = True) -> Dict[str, Any]:
        """
        Predict surface pollutant concentrations for new data.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Scale features
        if 'feature_scaler' in self.scalers:
            features_scaled = self.scalers['feature_scaler'].transform(features)
        else:
            features_scaled = features
        
        predictions = {}
        confidences = {}
        
        for pollutant, model in self.models.items():
            # Make predictions
            pred = model.predict(features_scaled)
            predictions[pollutant] = pred
            
            if return_confidence:
                # Calculate prediction intervals using ensemble variance
                tree_predictions = np.array([
                    tree.predict(features_scaled) 
                    for tree in model.estimators_
                ])
                
                # Standard deviation across trees as confidence measure
                confidence = np.std(tree_predictions, axis=0)
                confidences[pollutant] = confidence
        
        result = {
            'predictions': predictions,
            'pollutants': list(predictions.keys()),
            'n_samples': features.shape[0]
        }
        
        if return_confidence:
            result['confidence'] = confidences
        
        return result
    
    def save_models(self, model_dir: str = "models") -> str:
        """
        Save trained models, scalers, and metadata.
        
        Args:
            model_dir: Directory to save models
            
        Returns:
            Path to saved model file
        """
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Create model package
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'model_config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
        }
        
        # Save main model file
        model_file = model_path / "surface_estimator.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save individual models for backup
        for pollutant, model in self.models.items():
            individual_file = model_path / f"model_{pollutant}.joblib"
            joblib.dump(model, individual_file)
        
        logger.info(f"Models saved to {model_file}")
        return str(model_file)
    
    def load_models(self, model_file: str) -> bool:
        """
        Load trained models from file.
        
        Args:
            model_file: Path to saved model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(model_file, 'rb') as f:
                model_package = pickle.load(f)
            
            self.models = model_package['models']
            self.scalers = model_package['scalers']
            self.feature_names = model_package['feature_names']
            self.target_names = model_package['target_names']
            self.feature_importance = model_package.get('feature_importance', {})
            self.training_metrics = model_package.get('training_metrics', {})
            
            # Load model configuration
            config = model_package.get('model_config', {})
            self.n_estimators = config.get('n_estimators', 200)
            self.max_depth = config.get('max_depth', 20)
            self.random_state = config.get('random_state', 42)
            
            logger.info(f"Models loaded from {model_file}")
            logger.info(f"Available pollutants: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importance summary across all pollutants."""
        if not self.feature_importance:
            return pd.DataFrame()
        
        importance_data = []
        
        for pollutant, importance_dict in self.feature_importance.items():
            for feature, importance in importance_dict.items():
                importance_data.append({
                    'pollutant': pollutant,
                    'feature': feature,
                    'importance': importance
                })
        
        importance_df = pd.DataFrame(importance_data)
        
        # Calculate average importance across pollutants
        avg_importance = importance_df.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        return avg_importance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models and performance."""
        summary = {
            'n_models': len(self.models),
            'pollutants': list(self.models.keys()),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
        }
        
        if self.training_metrics:
            performance_summary = {}
            for pollutant, metrics in self.training_metrics.items():
                if 'test_metrics' in metrics:
                    performance_summary[pollutant] = metrics['test_metrics']
            summary['performance'] = performance_summary
        
        return summary
    
    def get_accuracy_report(self) -> Dict[str, Any]:
        """Get detailed accuracy report with interpretation."""
        if not self.training_metrics:
            return {"error": "No training metrics available"}
        
        report = {
            "overall_summary": {},
            "pollutant_details": {},
            "accuracy_interpretation": {}
        }
        
        # Collect all metrics
        all_r2_scores = []
        all_mae_scores = []
        all_rmse_scores = []
        all_accuracy_scores = []
        
        for pollutant, metrics in self.training_metrics.items():
            if 'test_metrics' in metrics:
                test_metrics = metrics['test_metrics']
                r2 = test_metrics.get('r2', 0)
                mae = test_metrics.get('mae', 0)
                rmse = test_metrics.get('rmse', 0)
                accuracy = test_metrics.get('accuracy_percentage', 0)
                mape = test_metrics.get('mape', 0)
                
                all_r2_scores.append(r2)
                all_mae_scores.append(mae)
                all_rmse_scores.append(rmse)
                all_accuracy_scores.append(accuracy)
                
                # Interpret accuracy level
                if r2 >= 0.9:
                    accuracy_level = "Excellent"
                elif r2 >= 0.8:
                    accuracy_level = "Very Good"
                elif r2 >= 0.7:
                    accuracy_level = "Good"
                elif r2 >= 0.5:
                    accuracy_level = "Moderate"
                elif r2 >= 0.3:
                    accuracy_level = "Fair"
                else:
                    accuracy_level = "Poor"
                
                report["pollutant_details"][pollutant] = {
                    "r2_score": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "accuracy_percentage": accuracy,
                    "accuracy_level": accuracy_level,
                    "sample_count": metrics.get('n_test_samples', 0)
                }
        
        # Calculate overall metrics
        if all_r2_scores:
            avg_r2 = np.mean(all_r2_scores)
            avg_mae = np.mean(all_mae_scores)
            avg_rmse = np.mean(all_rmse_scores)
            avg_accuracy = np.mean(all_accuracy_scores)
            
            report["overall_summary"] = {
                "average_r2": avg_r2,
                "average_mae": avg_mae,
                "average_rmse": avg_rmse,
                "average_accuracy_percentage": avg_accuracy,
                "best_pollutant": max(report["pollutant_details"].items(), 
                                    key=lambda x: x[1]["r2_score"])[0],
                "worst_pollutant": min(report["pollutant_details"].items(), 
                                     key=lambda x: x[1]["r2_score"])[0]
            }
            
            # Overall accuracy interpretation
            if avg_r2 >= 0.8:
                overall_level = "Very Good - Model predictions are highly reliable"
            elif avg_r2 >= 0.6:
                overall_level = "Good - Model predictions are reasonably accurate"
            elif avg_r2 >= 0.4:
                overall_level = "Moderate - Model shows some predictive ability"
            else:
                overall_level = "Poor - Model predictions may not be reliable"
            
            report["accuracy_interpretation"] = {
                "overall_level": overall_level,
                "metrics_explanation": {
                    "r2_score": "Coefficient of determination (0-1, higher is better). Explains how much variance is captured.",
                    "mae": "Mean Absolute Error - average absolute difference between predicted and actual values.",
                    "rmse": "Root Mean Square Error - penalizes larger errors more than MAE.",
                    "mape": "Mean Absolute Percentage Error - error as percentage of actual values.",
                    "accuracy_percentage": "Model accuracy expressed as percentage (based on R² score)."
                }
            }
        
        return report