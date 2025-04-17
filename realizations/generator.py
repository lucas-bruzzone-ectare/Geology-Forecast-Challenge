import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from config import NUM_REALIZATIONS, REALIZATION_STRATEGIES

def generate_calibrated_realizations(X, base_predictions, variance_params, n_derived_features, num_realizations=NUM_REALIZATIONS):
    """
    Generate calibrated realizations based on statistical analysis and base predictions
    
    Args:
        X: Input features
        base_predictions: Base predictions from ensemble
        variance_params: Variance parameters from analysis
        n_derived_features: Number of derived features
        num_realizations: Number of realizations to generate
        
    Returns:
        Array of realizations with shape (n_samples, num_realizations, 300)
    """
    # Use only original features for continuity
    X_original = X[:, :-n_derived_features]
    
    n_samples = X_original.shape[0]
    n_outputs = base_predictions.shape[1]  # Should be 300
    all_realizations = np.zeros((n_samples, num_realizations, n_outputs))
    
    # Extract variance parameters
    mean_autocorr_dist = variance_params['mean_autocorr_dist']
    variance_scale_factors = variance_params['variance_scale_factors']
    output_variance = variance_params['output_variance']
    
    # Calculate global factor to adjust NLL closer to zero
    global_variance_factor = 0.8  # Factor adjusted to bring NLL closer to zero
    
    # First realization: will be the base prediction
    print("Generating calibrated realizations...")
    all_realizations[:, 0, :] = base_predictions
    
    # Define strategies based on config
    strategies = REALIZATION_STRATEGIES
    
    # For each additional realization
    for r in range(1, num_realizations):
        # Select strategy
        strategy_idx = min(r-1, len(strategies)-1)
        strategy_name, scale_base, corr_length_factor, trend_type, trend_amplitude = strategies[strategy_idx]
        
        # Adjust correlation length based on mean_autocorr_dist if factor is provided
        corr_length = mean_autocorr_dist
        if corr_length_factor is not None:
            corr_length *= corr_length_factor
        
        print(f"Generating realization {r} with strategy: {strategy_name}")
        
        for i in range(n_samples):
            # Use base prediction
            realization = base_predictions[i].copy()
            
            # Estimate variability based on input data and first positions of output
            try:
                # Get last valid points
                last_points = X_original[i, -10:]
                valid_mask = ~np.isnan(last_points) & ~np.isinf(last_points)
                valid_last_points = last_points[valid_mask]
                
                if len(valid_last_points) > 0:
                    input_std = np.std(valid_last_points)
                else:
                    input_std = 0.1  # Default value if no valid points
            except:
                input_std = 0.1
            
            # Use average output variance of initial output (more reliable)
            output_std_init = np.sqrt(np.mean(output_variance[:20]))
            
            # Use a blend of both
            base_std = (input_std + output_std_init) / 2
            
            # If base_std is too small, use a minimum value
            base_std = max(base_std, 0.01)
            
            # Calculate noise scale
            noise_scale = base_std * scale_base * global_variance_factor
            
            # Generate base noise
            noise = np.random.normal(0, noise_scale, n_outputs)
            
            # Apply scale factors to adjust to NLL metric
            scaled_noise = noise * variance_scale_factors
            
            # Smooth noise to introduce spatial autocorrelation
            smooth_noise = gaussian_filter1d(scaled_noise, sigma=corr_length)
            
            # Apply modifications according to strategy
            if trend_type == "upward":
                # Add upward trend (more pronounced)
                trend_magnitude = base_std * trend_amplitude
                trend = np.linspace(0, trend_magnitude, n_outputs)
                smooth_noise += trend
                
            elif trend_type == "downward":
                # Add downward trend (more pronounced)
                trend_magnitude = base_std * trend_amplitude
                trend = np.linspace(trend_magnitude, 0, n_outputs)
                smooth_noise += trend
                
            elif trend_type == "late_upward":
                # Trend that starts flat and then goes up
                trend_magnitude = base_std * trend_amplitude
                x = np.linspace(0, 1, n_outputs)
                trend = trend_magnitude * (np.exp(3*x) - 1) / (np.exp(3) - 1)
                smooth_noise += trend
                
            elif trend_type == "late_downward":
                # Trend that starts flat and then goes down
                trend_magnitude = base_std * trend_amplitude
                x = np.linspace(0, 1, n_outputs)
                trend = trend_magnitude * (1 - (np.exp(3*x) - 1) / (np.exp(3) - 1))
                smooth_noise += trend
                
            elif trend_type == "oscillatory":
                # Add oscillatory pattern
                n_cycles = 2 if "long" in strategy_name else 4
                cycle_magnitude = base_std * trend_amplitude
                cycles = np.sin(np.linspace(0, n_cycles*np.pi, n_outputs)) * cycle_magnitude
                smooth_noise += cycles
            
            # Add processed noise
            realization += smooth_noise
            
            # Ensure continuity with input data (last point)
            try:
                # Get last valid point from input
                last_valid_idx = -1
                while last_valid_idx >= -X_original.shape[1]:
                    last_point = X_original[i, last_valid_idx]
                    if not np.isnan(last_point) and not np.isinf(last_point):
                        break
                    last_valid_idx -= 1
                
                if last_valid_idx >= -X_original.shape[1]:
                    offset = realization[0] - X_original[i, last_valid_idx]
                    realization -= offset
            except:
                # In case of error, don't adjust offset
                pass
            
            # For some strategies, apply additional smoothing
            if strategy_name in ["high_fidelity", "long_oscillation"]:
                try:
                    # Use Savitzky-Golay filter for more smoothing
                    window_size = min(15, int(corr_length*2) + 1)
                    window_size = window_size + 1 if window_size % 2 == 0 else window_size  # Ensure it's odd
                    realization = savgol_filter(realization, window_length=window_size, polyorder=2)
                except:
                    # In case of error, don't apply additional smoothing
                    pass
            
            # Check if realization contains invalid values
            if np.isnan(realization).any() or np.isinf(realization).any():
                # Replace with base prediction
                realization = base_predictions[i].copy()
            
            # Store realization
            all_realizations[i, r, :] = realization
    
    return all_realizations

def apply_calibration_scale(realizations, base_predictions, scale_factor):
    """
    Apply calibration scale to realizations
    
    Args:
        realizations: Array of realizations
        base_predictions: Base predictions (first realization)
        scale_factor: Scale factor to apply
        
    Returns:
        Scaled realizations
    """
    # Clone realizations to avoid modifying the original
    scaled_realizations = realizations.copy()
    
    # Apply scale to all realizations except the first one
    for r in range(1, realizations.shape[1]):
        # Calculate difference from base prediction
        diff = realizations[:, r, :] - base_predictions
        
        # Apply scale factor
        scaled_diff = diff * scale_factor
        
        # Update scaled realization
        scaled_realizations[:, r, :] = base_predictions + scaled_diff
    
    return scaled_realizations
