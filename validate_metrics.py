#!/usr/bin/env python3
"""
Validate TCN model metrics calculations
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def validate_accuracy_calculations():
    """Test accuracy metric calculations with known data"""
    
    print("=== VALIDATING ACCURACY CALCULATIONS ===\n")
    
    # Test Case 1: Perfect predictions (should be 100% accurate)
    print("Test Case 1: Perfect Predictions")
    y_true_1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    # Calculate accuracy manually
    abs_true = np.abs(y_true_1)
    denom_floor = max(np.percentile(abs_true, 10), 1e-3)
    denom = np.maximum(abs_true, denom_floor)
    
    tolerance_10 = np.mean(np.abs(y_true_1 - y_pred_1) <= 0.1 * denom)
    tolerance_20 = np.mean(np.abs(y_true_1 - y_pred_1) <= 0.2 * denom)
    
    print(f"  y_true: {y_true_1}")
    print(f"  y_pred: {y_pred_1}")
    print(f"  Accuracy(±10%): {tolerance_10:.3f} (should be 1.000)")
    print(f"  Accuracy(±20%): {tolerance_20:.3f} (should be 1.000)")
    
    # Test Case 2: 10% off predictions
    print("\nTest Case 2: 10% Off Predictions")
    y_true_2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_2 = np.array([11.0, 22.0, 33.0, 44.0, 55.0])  # 10% higher
    
    abs_true = np.abs(y_true_2)
    denom_floor = max(np.percentile(abs_true, 10), 1e-3)
    denom = np.maximum(abs_true, denom_floor)
    
    tolerance_10 = np.mean(np.abs(y_true_2 - y_pred_2) <= 0.1 * denom)
    tolerance_20 = np.mean(np.abs(y_true_2 - y_pred_2) <= 0.2 * denom)
    
    print(f"  y_true: {y_true_2}")
    print(f"  y_pred: {y_pred_2}")
    print(f"  Errors: {np.abs(y_true_2 - y_pred_2)}")
    print(f"  Tolerance 10%: {0.1 * denom}")
    print(f"  Accuracy(±10%): {tolerance_10:.3f} (should be 1.000)")
    print(f"  Accuracy(±20%): {tolerance_20:.3f} (should be 1.000)")
    
    # Test Case 3: Mixed accuracy
    print("\nTest Case 3: Mixed Accuracy")
    y_true_3 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_3 = np.array([10.5, 18.0, 35.0, 48.0, 45.0])  # Mixed errors
    
    abs_true = np.abs(y_true_3)
    denom_floor = max(np.percentile(abs_true, 10), 1e-3)
    denom = np.maximum(abs_true, denom_floor)
    
    errors = np.abs(y_true_3 - y_pred_3)
    tolerance_10_thresh = 0.1 * denom
    tolerance_20_thresh = 0.2 * denom
    
    within_10 = errors <= tolerance_10_thresh
    within_20 = errors <= tolerance_20_thresh
    
    tolerance_10 = np.mean(within_10)
    tolerance_20 = np.mean(within_20)
    
    print(f"  y_true: {y_true_3}")
    print(f"  y_pred: {y_pred_3}")
    print(f"  Errors: {errors}")
    print(f"  Tolerance 10%: {tolerance_10_thresh}")
    print(f"  Within 10%: {within_10}")
    print(f"  Tolerance 20%: {tolerance_20_thresh}")
    print(f"  Within 20%: {within_20}")
    print(f"  Accuracy(±10%): {tolerance_10:.3f}")
    print(f"  Accuracy(±20%): {tolerance_20:.3f}")
    
    # Test Case 4: Near-zero values (stabilization test)
    print("\nTest Case 4: Near-Zero Values")
    y_true_4 = np.array([0.01, 0.02, 0.001, 0.05, 0.1])
    y_pred_4 = np.array([0.011, 0.018, 0.0015, 0.055, 0.09])
    
    abs_true = np.abs(y_true_4)
    denom_floor = max(np.percentile(abs_true, 10), 1e-3)
    denom = np.maximum(abs_true, denom_floor)
    
    errors = np.abs(y_true_4 - y_pred_4)
    tolerance_10_thresh = 0.1 * denom
    tolerance_20_thresh = 0.2 * denom
    
    tolerance_10 = np.mean(errors <= tolerance_10_thresh)
    tolerance_20 = np.mean(errors <= tolerance_20_thresh)
    
    print(f"  y_true: {y_true_4}")
    print(f"  y_pred: {y_pred_4}")
    print(f"  Denom floor: {denom_floor}")
    print(f"  Stabilized denom: {denom}")
    print(f"  Errors: {errors}")
    print(f"  Tolerance 10%: {tolerance_10_thresh}")
    print(f"  Tolerance 20%: {tolerance_20_thresh}")
    print(f"  Accuracy(±10%): {tolerance_10:.3f}")
    print(f"  Accuracy(±20%): {tolerance_20:.3f}")

def load_and_validate_model_results():
    """Load actual model results and validate metrics"""
    
    print("\n\n=== VALIDATING ACTUAL MODEL RESULTS ===\n")
    
    # Load the trained model to get actual predictions
    try:
        import tensorflow as tf
        from ensemble_air_quality_model import TCNAirQualityModel
        
        # Initialize model class
        model_class = TCNAirQualityModel()
        
        # Load the saved model
        model_class.model = tf.keras.models.load_model('tcn_air_quality_model.keras')
        
        # Load test data
        print("Loading test data...")
        X_test = np.load('integrated_data/X_test.npy')
        y_test = np.load('integrated_data/y_test.npy') 
        
        # Load target scaler if it exists
        import joblib
        try:
            model_class.target_scaler = joblib.load('tcn_models/target_scaler.pkl')
            print("Loaded target scaler")
        except:
            print("No target scaler found")
            model_class.target_scaler = None
            
        target_names = ['NO2', 'O3']
        
        print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        
        # Get predictions
        y_pred = model_class.predict_tcn(X_test)
        
        print(f"Predictions shape: {y_pred.shape}")
        
        # Manual validation of metrics for each target
        for i, target_name in enumerate(target_names):
            print(f"\n--- Validating {target_name} metrics ---")
            
            y_true_target = y_test[:, i]
            y_pred_target = y_pred[:, i]
            
            # Basic stats
            print(f"True values - Min: {y_true_target.min():.4f}, Max: {y_true_target.max():.4f}, Mean: {y_true_target.mean():.4f}")
            print(f"Pred values - Min: {y_pred_target.min():.4f}, Max: {y_pred_target.max():.4f}, Mean: {y_pred_target.mean():.4f}")
            
            # Standard metrics
            mse = mean_squared_error(y_true_target, y_pred_target)
            mae = mean_absolute_error(y_true_target, y_pred_target)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_target, y_pred_target)
            
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"R²: {r2:.6f}")
            
            # Accuracy calculations (matching model code)
            abs_true = np.abs(y_true_target)
            denom_floor = max(np.percentile(abs_true, 10), 1e-3)
            denom = np.maximum(abs_true, denom_floor)
            
            errors = np.abs(y_true_target - y_pred_target)
            tolerance_10 = np.mean(errors <= 0.1 * denom)
            tolerance_20 = np.mean(errors <= 0.2 * denom)
            
            print(f"Denom floor (10th percentile or 1e-3): {denom_floor:.6f}")
            print(f"Mean error: {errors.mean():.6f}")
            print(f"Mean tolerance 10%: {(0.1 * denom).mean():.6f}")
            print(f"Mean tolerance 20%: {(0.2 * denom).mean():.6f}")
            print(f"Accuracy(±10%): {tolerance_10:.3f}")
            print(f"Accuracy(±20%): {tolerance_20:.3f}")
            
            # MAPE calculation
            mape = np.mean(2.0 * np.abs(y_true_target - y_pred_target) / 
                          (np.abs(y_true_target) + np.abs(y_pred_target) + 1e-7)) * 100
            print(f"MAPE: {mape:.2f}%")
            
            # Sample some individual predictions for spot checking
            print(f"\nSample predictions (first 10):")
            for j in range(min(10, len(y_true_target))):
                error = errors[j]
                tol_10 = 0.1 * denom[j]
                tol_20 = 0.2 * denom[j]
                within_10 = error <= tol_10
                within_20 = error <= tol_20
                print(f"  True: {y_true_target[j]:.4f}, Pred: {y_pred_target[j]:.4f}, "
                      f"Error: {error:.4f}, Tol10%: {tol_10:.4f}, Tol20%: {tol_20:.4f}, "
                      f"Within10%: {within_10}, Within20%: {within_20}")
                
    except Exception as e:
        print(f"Could not validate actual model results: {e}")

if __name__ == "__main__":
    validate_accuracy_calculations()
    load_and_validate_model_results()
