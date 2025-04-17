import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor
import functools
import joblib
from scipy.interpolate import UnivariateSpline
from models.stacking import StackingRegressor
from models.hyperparameter_tuning import optimize_hyperparameters
from config import (SEED, DEFAULT_XGB_PARAMS, DEFAULT_LGB_PARAMS, DEFAULT_GBR_PARAMS, 
                   DEFAULT_RF_PARAMS, DEFAULT_RIDGE_PARAMS, DEFAULT_ELASTIC_PARAMS,
                   XGB_PARAM_SPACE, LGB_PARAM_SPACE, GBR_PARAM_SPACE, RF_PARAM_SPACE)

def train_model_for_position(pos_idx, X_train_processed, y_train, n_outputs, positions_params=None):
    """
    Train models for a specific position with hyperparameter optimization
    
    Args:
        pos_idx: Position index
        X_train_processed: Processed training features
        y_train: Training targets
        n_outputs: Total number of outputs
        positions_params: Pre-optimized parameters for each position type
    
    Returns:
        Dictionary with trained models and metadata
    """
    pos = pos_idx + 1  # Convert to 1-indexed
    print(f"Training models for position {pos} of {n_outputs}...")
    
    # Extract target for this position
    y_pos = y_train[:, pos_idx]
    
    # Dictionary to store model performance
    model_performances = {}
    
    # Position groups to handle optimization efficiently
    if pos <= 20:
        pos_group = 'early'  # First 20 positions
    elif pos <= 100:
        pos_group = 'mid_early'  # Positions 21-100
    elif pos <= 200:
        pos_group = 'mid_late'  # Positions 101-200
    else:
        pos_group = 'late'  # Positions 201-300
    
    # Use pre-optimized parameters if available
    if positions_params and pos_group in positions_params:
        print(f"  Using pre-optimized parameters for position group: {pos_group}")
        param_set = positions_params[pos_group]
        xgb_params = param_set['xgb']
        lgb_params = param_set['lgb']
        gbr_params = param_set['gbr']
        rf_params = param_set['rf']
    else:
        # Default parameters if optimization wasn't done
        print("  Using default parameters (no optimization data available)")
        xgb_params = DEFAULT_XGB_PARAMS
        lgb_params = DEFAULT_LGB_PARAMS
        gbr_params = DEFAULT_GBR_PARAMS
        rf_params = DEFAULT_RF_PARAMS
    
    # Ridge and ElasticNet params (less sensitive to tuning)
    ridge_params = DEFAULT_RIDGE_PARAMS
    elastic_params = DEFAULT_ELASTIC_PARAMS
    
    # Train XGBoost
    try:
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_processed, y_pos)
        xgb_pred = xgb_model.predict(X_train_processed)
        xgb_mse = mean_squared_error(y_pos, xgb_pred)
        model_performances['xgb'] = {'model': xgb_model, 'mse': xgb_mse}
    except Exception as e:
        print(f"  Error training XGBoost: {e}")
    
    # Train LightGBM com supressão de avisos
    try:
        # Configuração para suprimir logs do LightGBM
        lgb_params_copy = lgb_params.copy()
        lgb_params_copy['verbose'] = -1
        
        # Create feature names to avoid the warning
        feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
        
        lgb_model = lgb.LGBMRegressor(**lgb_params_copy)
        # Pass feature_names to avoid the warning during prediction
        lgb_model.fit(X_train_processed, y_pos, feature_name=feature_names)
        lgb_pred = lgb_model.predict(X_train_processed)
        lgb_mse = mean_squared_error(y_pos, lgb_pred)
        model_performances['lgb'] = {'model': lgb_model, 'mse': lgb_mse, 'feature_names': feature_names}
    except Exception as e:
        print(f"  Error training LightGBM: {e}")
    
    # Train GBR
    try:
        gbr_model = GradientBoostingRegressor(**gbr_params)
        gbr_model.fit(X_train_processed, y_pos)
        gbr_pred = gbr_model.predict(X_train_processed)
        gbr_mse = mean_squared_error(y_pos, gbr_pred)
        model_performances['gbr'] = {'model': gbr_model, 'mse': gbr_mse}
    except Exception as e:
        print(f"  Error training GBR: {e}")
    
    # Train RF
    try:
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train_processed, y_pos)
        rf_pred = rf_model.predict(X_train_processed)
        rf_mse = mean_squared_error(y_pos, rf_pred)
        model_performances['rf'] = {'model': rf_model, 'mse': rf_mse}
    except Exception as e:
        print(f"  Error training RF: {e}")
    
    # Train Ridge Regression
    try:
        ridge_model = Ridge(**ridge_params)
        ridge_model.fit(X_train_processed, y_pos)
        ridge_pred = ridge_model.predict(X_train_processed)
        ridge_mse = mean_squared_error(y_pos, ridge_pred)
        model_performances['ridge'] = {'model': ridge_model, 'mse': ridge_mse}
    except Exception as e:
        print(f"  Error training Ridge: {e}")
    
    # Train ElasticNet
    try:
        elastic_model = ElasticNet(**elastic_params)
        elastic_model.fit(X_train_processed, y_pos)
        elastic_pred = elastic_model.predict(X_train_processed)
        elastic_mse = mean_squared_error(y_pos, elastic_pred)
        model_performances['elastic'] = {'model': elastic_model, 'mse': elastic_mse}
    except Exception as e:
        print(f"  Error training ElasticNet: {e}")
    
    # Train Stacking model if we have enough base models
    if len(model_performances) >= 3:
        try:
            # Get available models for stacking
            base_models = [info['model'] for info in model_performances.values()]
            
            # Use RidgeCV as meta-learner (more stable than OLS)
            meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
            
            # Create stacking regressor
            stack = StackingRegressor(
                base_models=base_models[:3],  # Use top 3 models to avoid overfitting
                meta_model=meta_learner,
                n_folds=3,
                use_features_in_meta=False
            )
            
            # Train stacking model
            stack.fit(X_train_processed, y_pos)
            stack_pred = stack.predict(X_train_processed)
            stack_mse = mean_squared_error(y_pos, stack_pred)
            model_performances['stack'] = {'model': stack, 'mse': stack_mse}
        except Exception as e:
            print(f"  Error training Stacking model: {e}")
    
    # Calculate weights using more sophisticated weighting scheme
    weights = {}
    # Base weight using inverse MSE
    total_inv_mse = 0
    inv_mse_weights = {}
    
    for model_name, info in model_performances.items():
        inv_mse = 1.0 / (info['mse'] + 1e-6)
        inv_mse_weights[model_name] = inv_mse
        total_inv_mse += inv_mse
    
    # Normalize inverse MSE weights
    for model_name in inv_mse_weights:
        inv_mse_weights[model_name] /= total_inv_mse
    
    # Apply weight adjustments based on model characteristics
    for model_name in model_performances:
        # Start with inverse MSE weight
        weights[model_name] = inv_mse_weights[model_name]
        
        # Boost stacking model weight if it's good
        if model_name == 'stack' and model_performances['stack']['mse'] < min(
            info['mse'] for name, info in model_performances.items() if name != 'stack'
        ):
            weights[model_name] *= 1.5
        
        # Boost tree-based models at the beginning (positions 1-60)
        # Tree models better capture the initial trajectory
        if pos <= 60 and model_name in ['xgb', 'lgb', 'gbr', 'rf']:
            weights[model_name] *= 1.2
        
        # Boost linear models for far positions (positions 200-300)
        # Linear models tend to extrapolate better for far positions
        if pos >= 200 and model_name in ['ridge', 'elastic']:
            weights[model_name] *= 1.2
    
    # Renormalize weights
    total_weight = sum(weights.values())
    for model_name in weights:
        weights[model_name] /= total_weight
    
    # Store models and their weights
    model_data = {
        'models': {name: info['model'] for name, info in model_performances.items()},
        'weights': weights,
        'performances': {name: info['mse'] for name, info in model_performances.items()}
    }
    
    # Also store feature names for LightGBM if available
    if 'lgb' in model_performances and 'feature_names' in model_performances['lgb']:
        model_data['lgb_feature_names'] = model_performances['lgb']['feature_names']
    
    # Print performance summary
    print(f"  Model performances (MSE) and weights:")
    for model_name, info in model_performances.items():
        print(f"    {model_name}: MSE={info['mse']:.4f}, Weight={weights[model_name]:.2f}")
    
    return pos, model_data

def train_advanced_ensemble_models(X_train, y_train, n_derived_features, n_jobs=-1):
    """
    Train an advanced ensemble of models including XGBoost, LightGBM, and stacking
    with parallel training and hyperparameter optimization
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_derived_features: Number of derived features
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary of models, scaler, key positions, number of derived features
    """
    print("Training advanced ensemble with XGBoost, LightGBM, and stacking...")
    
    # Determine number of output models (one for each future position)
    n_outputs = y_train.shape[1]
    
    # Create models for key positions and interpolate intermediates
    key_positions = []
    
    # More granularity at the beginning (where the metric is more sensitive)
    key_positions.extend(list(range(0, 60, 10)))
    key_positions.extend(list(range(60, 240, 30)))
    key_positions.extend(list(range(240, n_outputs, 20)))
    
    if key_positions[-1] != n_outputs - 1:
        key_positions.append(n_outputs - 1)  # Ensure we have the last point
    
    # Dictionary to store models
    models = {}
    
    # Scale the data
    scaler = StandardScaler()
    X_original = X_train[:, :-n_derived_features]  # Separate original features
    X_derived = X_train[:, -n_derived_features:]   # Separate derived features
    
    # Scale only the original features
    X_original_scaled = scaler.fit_transform(X_original)
    
    # Recombine
    X_train_processed = np.hstack((X_original_scaled, X_derived))
    
    # Perform hyperparameter optimization for representative positions
    print("Performing hyperparameter optimization for representative positions...")
    positions_params = {}
    
    # Representative positions for different parts of the sequence
    rep_positions = [10, 60, 150, 250]  # Early, mid-early, mid-late, late
    position_groups = ['early', 'mid_early', 'mid_late', 'late']
    
    # For each representative position, find optimal hyperparameters
    for rep_pos, pos_group in zip(rep_positions, position_groups):
        print(f"Optimizing hyperparameters for position group: {pos_group} (position {rep_pos})")
        # Get target for this position
        y_pos = y_train[:, rep_pos]
        
        # Optimize XGBoost
        print("  Optimizing XGBoost...")
        xgb_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'xgb', XGB_PARAM_SPACE, n_iter=15)
        
        # Optimize LightGBM
        print("  Optimizing LightGBM...")
        lgb_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'lgb', LGB_PARAM_SPACE, n_iter=15)
        
        # Optimize GradientBoosting
        print("  Optimizing GradientBoosting...")
        gbr_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'gbr', GBR_PARAM_SPACE, n_iter=15)
        
        # Optimize RandomForest
        print("  Optimizing RandomForest...")
        rf_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'rf', RF_PARAM_SPACE, n_iter=15)
        
        # Store optimized params
        positions_params[pos_group] = {
            'xgb': xgb_best_params,
            'lgb': lgb_best_params,
            'gbr': gbr_best_params,
            'rf': rf_best_params
        }
        
        print(f"  Optimization completed for position {rep_pos}")
    
    # Save optimized parameters
    joblib.dump(positions_params, 'position_params.pkl')
    print("Hyperparameter optimization completed and saved")
    
    # Train models for all key positions (potentially in parallel)
    train_func = functools.partial(
        train_model_for_position,
        X_train_processed=X_train_processed,
        y_train=y_train,
        n_outputs=n_outputs,
        positions_params=positions_params
    )
    
    # Determine parallelization approach
    if n_jobs != 1:
        print(f"Training models in parallel with {n_jobs} workers...")
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            results = list(executor.map(train_func, key_positions))
        
        # Process results
        for pos, model_data in results:
            models[pos] = model_data
    else:
        print("Training models sequentially...")
        for pos_idx in key_positions:
            pos, model_data = train_func(pos_idx)
            models[pos] = model_data
    
    return models, scaler, key_positions, n_derived_features

def predict_with_advanced_ensemble(X, models, scaler, key_positions, n_derived_features):
    """
    Generate predictions using the advanced ensemble and interpolation
    
    Args:
        X: Features
        models: Dictionary of trained models
        scaler: Fitted scaler
        key_positions: List of key positions
        n_derived_features: Number of derived features
        
    Returns:
        Predictions
    """
    n_samples = X.shape[0]
    n_outputs = 300  # Total number of output positions
    
    # Separate and scale features
    X_original = X[:, :-n_derived_features]
    X_derived = X[:, -n_derived_features:]
    X_original_scaled = scaler.transform(X_original)
    X_processed = np.hstack((X_original_scaled, X_derived))
    
    # Initialize prediction matrix
    predictions = np.zeros((n_samples, n_outputs))
    
    # Generate predictions for key positions
    for pos in key_positions:
        model_pos = pos + 1  # Convert to 1-indexed
        model_info = models[model_pos]
        pos_idx = pos  # Convert to 0-indexed
        
        # Initialize blended predictions for this position
        blended_preds = np.zeros(n_samples)
        
        # Get all models and their weights
        model_dict = model_info['models']
        weight_dict = model_info['weights']
        
        # Generate and blend predictions from all models
        for model_name, model in model_dict.items():
            try:
                # Special handling for LightGBM to avoid the warning
                if model_name == 'lgb' and 'lgb_feature_names' in model_info:
                    # Create a pandas DataFrame with feature names to avoid warning
                    import pandas as pd
                    feature_names = model_info['lgb_feature_names']
                    X_lgb = pd.DataFrame(X_processed, columns=feature_names)
                    preds = model.predict(X_lgb)
                else:
                    # For other models, predict normally
                    preds = model.predict(X_processed)
                
                # Add weighted predictions to the blend
                weight = weight_dict[model_name]
                blended_preds += weight * preds
            except Exception as e:
                print(f"Error predicting with {model_name} at position {pos}: {e}")
        
        predictions[:, pos_idx] = blended_preds
    
    # Improved interpolation strategy - adaptive smoothing
    for i in range(n_samples):
        # Known positions (0-indexed)
        x_known = np.array(key_positions)
        # Known values for these positions
        y_known = predictions[i, key_positions]
        
        # All positions (0-indexed)
        x_all = np.arange(n_outputs)
        
        # Analyze the rate of change to determine appropriate smoothing
        if len(y_known) > 2:
            # Calculate approximate derivatives
            diffs = np.abs(np.diff(y_known))
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            
            # Determine smoothing parameter based on variability
            if max_diff > 5 * mean_diff:
                # High variability - use less smoothing to preserve features
                smoothing = 0.05
            elif max_diff > 2 * mean_diff:
                # Moderate variability
                smoothing = 0.1
            else:
                # Low variability - use more smoothing
                smoothing = 0.2
        else:
            # Default smoothing
            smoothing = 0.1
        
        # Use spline for smooth interpolation with adaptive smoothing
        try:
            if len(x_known) > 3:
                # If we have at least 4 points, use cubic spline with adaptive smoothing
                spline = UnivariateSpline(x_known, y_known, k=3, s=smoothing)
                predictions[i, :] = spline(x_all)
            else:
                # Fallback to linear interpolation
                predictions[i, :] = np.interp(x_all, x_known, y_known)
        except:
            # In case of error, use linear interpolation
            predictions[i, :] = np.interp(x_all, x_known, y_known)
    
    return predictions