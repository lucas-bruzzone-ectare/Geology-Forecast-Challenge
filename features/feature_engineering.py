import numpy as np
from scipy.fft import fft

def create_derived_features(X):
    """
    Create enhanced derived features from input data with robust error handling
    
    Args:
        X: Input data with shape (n_samples, n_features)
        
    Returns:
        Derived features with shape (n_samples, n_derived)
        
    New features include:
    1. Linear trend (slope)
    2. Moving average of last 5 points
    3. Standard deviation of last 15 points
    4. Second-order coefficient (curvature)
    5. Rate of change in last 10 points
    6. Ratio between recent and prior mean
    7. Spectral power in different frequency bands
    8. Entropy measure of signal variability
    9. Adaptative trend strength indicator
    10. Pattern change detection feature
    """
    n_samples, _ = X.shape
    n_derived = 10  # 10 features
    derived_features = np.zeros((n_samples, n_derived))
    
    # Calculate features for each sample
    for i in range(n_samples):
        # Get the most recent points
        recent_points = X[i, -15:]  # Last 15 points
        x_vals = np.arange(-14, 1)  # Corresponding positions
        
        # Full series for spectral features
        full_series = X[i, :]
        
        # Check for NaN or inf values
        valid_mask = ~np.isnan(recent_points) & ~np.isinf(recent_points)
        valid_points = recent_points[valid_mask]
        valid_x = x_vals[valid_mask]
        
        # Clean full series for spectral analysis
        full_valid_mask = ~np.isnan(full_series) & ~np.isinf(full_series)
        full_valid = full_series[full_valid_mask]
        
        # Default values for safety
        derived_features[i, :] = np.zeros(n_derived)
        
        # 1. Linear trend (slope)
        try:
            if len(valid_points) >= 2:
                slope, intercept = np.polyfit(valid_x, valid_points, 1)
                derived_features[i, 0] = slope
            else:
                derived_features[i, 0] = 0.0
        except:
            derived_features[i, 0] = 0.0
        
        # 2. Moving average of last 5 points
        try:
            last_5_valid = X[i, -5:]
            last_5_valid = last_5_valid[~np.isnan(last_5_valid) & ~np.isinf(last_5_valid)]
            if len(last_5_valid) > 0:
                derived_features[i, 1] = np.mean(last_5_valid)
            else:
                derived_features[i, 1] = 0.0
        except:
            derived_features[i, 1] = 0.0
        
        # 3. Standard deviation of last 15 points
        try:
            if len(valid_points) > 1:
                derived_features[i, 2] = np.std(valid_points)
            else:
                derived_features[i, 2] = 0.01
        except:
            derived_features[i, 2] = 0.01
        
        # 4. Second-order coefficient (curvature)
        try:
            if len(valid_points) >= 3:
                coeffs = np.polyfit(valid_x, valid_points, 2)
                derived_features[i, 3] = coeffs[0]  # Quadratic coefficient
            else:
                derived_features[i, 3] = 0.0
        except:
            derived_features[i, 3] = 0.0
        
        # 5. Rate of change in last 10 points - Enhanced with relative change
        try:
            if len(valid_points) >= 10:
                first_valid = valid_points[0]
                last_valid = valid_points[-1]
                absolute_change = last_valid - first_valid
                # Calculate relative change to better capture magnitude-independent rates
                if abs(first_valid) > 1e-6:
                    derived_features[i, 4] = absolute_change / abs(first_valid)
                else:
                    derived_features[i, 4] = absolute_change
            else:
                derived_features[i, 4] = 0.0
        except:
            derived_features[i, 4] = 0.0
        
        # 6. Ratio between recent and prior mean - Enhanced with robust calculation
        try:
            recent_valid = X[i, -5:]
            recent_valid = recent_valid[~np.isnan(recent_valid) & ~np.isinf(recent_valid)]
            
            prior_valid = X[i, -15:-5]
            prior_valid = prior_valid[~np.isnan(prior_valid) & ~np.isinf(prior_valid)]
            
            if len(recent_valid) > 0 and len(prior_valid) > 0:
                recent_mean = np.mean(recent_valid)
                prior_mean = np.mean(prior_valid)
                
                # More robust ratio calculation with sigmoid normalization
                if abs(prior_mean) > 1e-6:
                    ratio = recent_mean / prior_mean
                    # Sigmoid-like normalization to handle extreme values
                    derived_features[i, 5] = 2 / (1 + np.exp(-ratio)) - 1
                else:
                    derived_features[i, 5] = 0.0
            else:
                derived_features[i, 5] = 0.0
        except:
            derived_features[i, 5] = 0.0
        
        # 7. Spectral power in different frequency bands
        try:
            if len(full_valid) >= 10:
                # Apply windowing to reduce spectral leakage
                window = np.hanning(len(full_valid))
                windowed_data = full_valid * window
                
                # Compute FFT
                fft_values = fft(windowed_data)
                fft_magnitude = np.abs(fft_values[:len(fft_values)//2])
                
                # Calculate power in different frequency bands
                if len(fft_magnitude) >= 3:
                    low_freq_power = np.sum(fft_magnitude[:len(fft_magnitude)//3])
                    mid_freq_power = np.sum(fft_magnitude[len(fft_magnitude)//3:2*len(fft_magnitude)//3])
                    high_freq_power = np.sum(fft_magnitude[2*len(fft_magnitude)//3:])
                    
                    # Calculate ratio of high to low frequency power (captures noise vs trend)
                    total_power = low_freq_power + mid_freq_power + high_freq_power
                    if total_power > 1e-6:
                        derived_features[i, 6] = (high_freq_power - low_freq_power) / total_power
                    else:
                        derived_features[i, 6] = 0.0
                else:
                    derived_features[i, 6] = 0.0
            else:
                derived_features[i, 6] = 0.0
        except:
            derived_features[i, 6] = 0.0
        
        # 8. Entropy measure of signal variability (approximate entropy)
        try:
            if len(valid_points) >= 4:
                # Simple approximation of entropy using differences between consecutive points
                diffs = np.diff(valid_points)
                if len(diffs) > 0:
                    # Calculate normalized entropy-like measure using absolute differences
                    abs_diffs = np.abs(diffs)
                    if np.sum(abs_diffs) > 1e-6:
                        probs = abs_diffs / np.sum(abs_diffs)
                        # Higher values for more uniform differences (predictable), lower for more variable (chaotic)
                        non_zero_probs = probs[probs > 0]
                        if len(non_zero_probs) > 0:
                            entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
                            # Normalize by maximum possible entropy for this number of points
                            max_entropy = np.log(len(non_zero_probs))
                            if max_entropy > 0:
                                derived_features[i, 7] = entropy / max_entropy
                            else:
                                derived_features[i, 7] = 0.0
                        else:
                            derived_features[i, 7] = 0.0
                    else:
                        derived_features[i, 7] = 0.0  # No variation
                else:
                    derived_features[i, 7] = 0.0
            else:
                derived_features[i, 7] = 0.5  # Default value
        except:
            derived_features[i, 7] = 0.5  # Default value
        
        # 9. Adaptive trend strength indicator
        try:
            if len(valid_points) >= 5:
                # Calculate linear trend
                slope, intercept = np.polyfit(valid_x, valid_points, 1)
                
                # Generate trend line
                trend_line = slope * valid_x + intercept
                
                # Calculate residuals
                residuals = valid_points - trend_line
                
                # Trend strength is ratio of trend variance to total variance
                residual_var = np.var(residuals) if len(residuals) > 1 else 0
                total_var = np.var(valid_points) if len(valid_points) > 1 else 0
                
                if total_var > 1e-6:
                    # 1 means perfect trend, 0 means no trend
                    trend_strength = 1 - (residual_var / total_var)
                    derived_features[i, 8] = trend_strength
                else:
                    derived_features[i, 8] = 0.0
            else:
                derived_features[i, 8] = 0.0
        except:
            derived_features[i, 8] = 0.0
        
        # 10. Pattern change detection feature
        try:
            if len(valid_points) >= 8:
                # Split the recent points into first and second half
                mid_point = len(valid_points) // 2
                first_half = valid_points[:mid_point]
                second_half = valid_points[mid_point:]
                
                if len(first_half) > 0 and len(second_half) > 0:
                    # Calculate statistics for each half
                    first_mean = np.mean(first_half)
                    second_mean = np.mean(second_half)
                    first_std = np.std(first_half) if len(first_half) > 1 else 0.01
                    second_std = np.std(second_half) if len(second_half) > 1 else 0.01
                    
                    # Calculate various change metrics
                    mean_change = abs(second_mean - first_mean)
                    std_change = abs(second_std - first_std)
                    
                    # Normalize by first half statistics to get relative change
                    mean_base = max(abs(first_mean), 0.01)
                    std_base = max(first_std, 0.01)
                    
                    # Combine into unified change detection metric
                    rel_mean_change = mean_change / mean_base
                    rel_std_change = std_change / std_base
                    
                    # Average of relative changes in mean and std (higher = more change)
                    derived_features[i, 9] = (rel_mean_change + rel_std_change) / 2
                else:
                    derived_features[i, 9] = 0.0
            else:
                derived_features[i, 9] = 0.0
        except:
            derived_features[i, 9] = 0.0
    
    # Check for invalid final values
    if np.isnan(derived_features).any() or np.isinf(derived_features).any():
        derived_features = np.nan_to_num(derived_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return derived_features
