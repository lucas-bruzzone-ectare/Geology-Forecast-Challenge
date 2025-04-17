import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from config import SEED
import pandas as pd

def optimize_hyperparameters(X, y, model_type, param_space, n_iter=20, cv=3):
    """
    Optimize hyperparameters using Bayesian optimization
    
    Args:
        X: Features
        y: Target
        model_type: Type of model ('xgb', 'lgb', 'gbr', 'rf')
        param_space: Dictionary of parameter spaces
        n_iter: Number of iterations for optimization
        cv: Number of cross-validation folds
    
    Returns:
        Best parameters
    """
    # Initialize model based on type
    if model_type == 'xgb':
        model = xgb.XGBRegressor(random_state=SEED)
    elif model_type == 'lgb':
        # Add configuration to suppress warnings
        model = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(random_state=SEED)
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=SEED)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Convert param_space dictionary to skopt space
    skopt_param_space = {}
    for param, space in param_space.items():
        if isinstance(space, tuple) and len(space) == 2:
            if isinstance(space[0], int) and isinstance(space[1], int):
                skopt_param_space[param] = Integer(space[0], space[1])
            elif isinstance(space[0], (int, float)) and isinstance(space[1], (int, float)):
                # Check if using log-uniform and handle zero values
                if space[0] == 0:
                    # Use a small positive value instead of zero for log-uniform
                    skopt_param_space[param] = Real(1e-6, space[1], prior='log-uniform')
                else:
                    skopt_param_space[param] = Real(space[0], space[1], prior='log-uniform')
        else:
            # Keep as-is if already in correct format
            skopt_param_space[param] = space
    
    # For LightGBM, we need to add feature names to avoid warnings
    if model_type == 'lgb':
        # Create feature names for LightGBM
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Create DataFrame with feature names
        X_lgb = pd.DataFrame(X, columns=feature_names)
        
        # Create BayesSearchCV object with named features
        opt = BayesSearchCV(
            model,
            skopt_param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=SEED,
            n_jobs=1,  # Use single job for LightGBM to avoid warnings
            verbose=0
        )
        
        # Fit the optimizer with named features
        opt.fit(X_lgb, y)
    else:
        # Create BayesSearchCV object
        opt = BayesSearchCV(
            model,
            skopt_param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=SEED,
            n_jobs=-1 if model_type != 'xgb' else 1,  # XGBoost with n_jobs can cause issues
            verbose=0
        )
        
        # Fit the optimizer
        opt.fit(X, y)
    
    # Return best parameters
    best_params = opt.best_params_
    
    # Add random_state to parameters
    best_params['random_state'] = SEED
    
    return best_params