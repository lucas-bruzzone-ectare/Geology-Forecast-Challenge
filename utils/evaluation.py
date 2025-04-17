import numpy as np
import matplotlib.pyplot as plt
from config import LOG_SLOPES, LOG_OFFSETS, VISUALIZATION_PATH_VARIANCE

def calculate_inverse_covariance_vector():
    """
    Calculate the D_T^{-1}(x) vector for all positions from 1 to 300
    
    Returns:
        Inverse covariance vector
    """
    inverse_cov_vector = np.zeros(300)
    for x in range(1, 301):
        if 1 <= x <= 60:
            k = 0  # First region (1-60)
        elif 61 <= x <= 244:
            k = 1  # Second region (61-244)
        else:
            k = 2  # Third region (245-300)
        
        # Apply formula: D_T^{-1}(x) = exp(log(x) * a_k + b_k)
        inverse_cov_vector[x-1] = np.exp(np.log(x) * LOG_SLOPES[k] + LOG_OFFSETS[k])
    
    return inverse_cov_vector

def analyze_variance_structure(X_train, y_train, n_derived_features):
    """
    Analyze variance structure in the data to calibrate realization generation
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_derived_features: Number of derived features
    
    Returns:
        Dictionary of variance parameters
    """
    print("Analyzing variance structure in the data...")
    
    # Use only original features for variance analysis
    X_train_original = X_train[:, :-n_derived_features]
    
    # Number of samples and points
    n_samples = X_train_original.shape[0]
    n_input_points = X_train_original.shape[1]
    n_output_points = y_train.shape[1]
    
    # Calculate variance at each position of input data
    input_variance = np.var(X_train_original, axis=0)
    
    # Calculate variance at each position of output data
    output_variance = np.var(y_train, axis=0)
    
    # Calculate average autocorrelation of known data
    autocorr_dists = []
    
    for sample_idx in range(min(500, n_samples)):  # Limit to avoid being too slow
        # Get data for this sample
        sample = X_train_original[sample_idx]
        
        # Check for invalid values
        if np.isnan(sample).any() or np.isinf(sample).any():
            continue
            
        try:
            # Calculate autocorrelation
            corr = np.correlate(sample, sample, mode='full')
            
            # Normalize for correlation
            corr = corr / np.max(corr)
            
            # Extract positive part (excluding zero)
            half_len = len(corr) // 2
            corr = corr[half_len+1:]
            
            # Store distance where correlation drops to 0.5
            half_corr_dist = np.argmax(corr < 0.5)
            if half_corr_dist > 0:
                autocorr_dists.append(half_corr_dist)
        except:
            continue
    
    # Calculate average autocorrelation distance
    if autocorr_dists:
        mean_autocorr_dist = np.mean(autocorr_dists)
    else:
        mean_autocorr_dist = 5.0  # Default value if we can't calculate
    
    # Calculate variance scale factors based on evaluation metric
    inverse_cov_vector = calculate_inverse_covariance_vector()
    variance_scale_factors = 1.0 / np.sqrt(inverse_cov_vector)
    
    # Normalize to a reasonable range
    variance_scale_factors = variance_scale_factors / np.mean(variance_scale_factors)
    
    # Visualize analysis results
    plt.figure(figsize=(12, 10))
    
    # Plot variance
    plt.subplot(2, 1, 1)
    plt.plot(range(-n_input_points, 0), input_variance, 'b-', label='Input')
    plt.plot(range(1, n_output_points + 1), output_variance, 'r-', label='Output')
    plt.title('Variance by Position')
    plt.xlabel('Position')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot variance scale factors
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 301), variance_scale_factors)
    plt.title('Variance Scale Factors (based on NLL metric)')
    plt.xlabel('Position')
    plt.ylabel('Scale Factor')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH_VARIANCE)
    print(f"Refined variance analysis visualization saved as '{VISUALIZATION_PATH_VARIANCE}'")
    
    # Return estimated parameters
    variance_params = {
        'input_variance': input_variance,
        'output_variance': output_variance,
        'mean_autocorr_dist': mean_autocorr_dist,
        'variance_scale_factors': variance_scale_factors
    }
    
    return variance_params

def calculate_nll_loss(y_true, predictions):
    """
    Calculate Negative Log Likelihood Loss as defined in the evaluation
    
    Args:
        y_true: Array of true values with shape (n_samples, 300)
        predictions: Array of realizations with shape (n_samples, num_realizations, 300)
    
    Returns:
        Average NLL loss for all samples
    """
    n_samples = y_true.shape[0]
    num_realizations = predictions.shape[1]
    
    # Calculate idealized inverse covariance vector
    inverse_cov_vector = calculate_inverse_covariance_vector()
    
    # Calculate NLL for each sample
    losses = np.zeros(n_samples)
    
    # Probability of each realization (equiprobable)
    p_i = 1.0 / num_realizations
    
    for i in range(n_samples):
        # Array to store gaussian misfit for each realization
        gaussian_misfits = np.zeros(num_realizations)
        
        for r in range(num_realizations):
            # Calculate error vector (residual)
            error_vector = y_true[i] - predictions[i, r]
            
            # Check for invalid values in error
            if np.isnan(error_vector).any() or np.isinf(error_vector).any():
                error_vector = np.nan_to_num(error_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # According to metric documentation:
            # E_i,G = exp(e_i(x) ⋅ D_T^{-1}(x) ⋅ e_i(x))
            weighted_error_sum = np.sum(error_vector**2 * inverse_cov_vector)
            
            # Limit to avoid numerical problems
            weighted_error_sum = np.clip(weighted_error_sum, -700, 700)
            
            # Calculate gaussian misfit
            gaussian_misfits[r] = np.exp(weighted_error_sum)
        
        # Calculate weighted sum of misfits (NLL)
        sum_weighted_misfits = np.sum(p_i * gaussian_misfits)
        
        # Avoid numerical problems with logarithm
        if sum_weighted_misfits > 1e-300:
            losses[i] = -np.log(sum_weighted_misfits)
        else:
            # Maximum value for extreme cases
            losses[i] = 700.0
    
    # Calculate mean and display statistics
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    high_loss_count = np.sum(losses > 100)
    
    print(f"NLL Statistics: mean={mean_loss:.2f}, median={median_loss:.2f}, high NLL samples: {high_loss_count}/{n_samples}")
    
    return mean_loss

def calibrate_nll_target(val_realizations, y_val, initial_value=0.8, target_nll=-69):
    """
    Adjust variance scale of realizations to bring NLL closer to target
    
    Args:
        val_realizations: Validation realizations
        y_val: Validation targets
        initial_value: Initial scale factor
        target_nll: Target NLL value
        
    Returns:
        Best scale factor, best NLL
    """
    print("Calibrating variance for target NLL...")
    
    # Clone original realizations
    scaled_realizations = val_realizations.copy()
    
    # First realization is not altered
    base_predictions = val_realizations[:, 0, :]
    
    # Function to scale alternative realizations
    def scale_realizations(scale_factor):
        for r in range(1, scaled_realizations.shape[1]):
            # Get base and alternative realizations
            base = base_predictions
            alt = val_realizations[:, r, :]
            
            # Calculate difference
            diff = alt - base
            
            # Apply scale factor to difference
            scaled_diff = diff * scale_factor
            
            # Update scaled realization
            scaled_realizations[:, r, :] = base + scaled_diff
        
        # Calculate NLL with scaled realizations
        nll = calculate_nll_loss(y_val, scaled_realizations)
        return nll
    
    # Test different scales and choose the best
    best_scale = initial_value
    best_nll = scale_realizations(initial_value)
    print(f"Initial NLL with scale {initial_value}: {best_nll}")
    
    # Fine-tuning if we're far from target
    scales_to_try = []
    if best_nll < target_nll:  # NLL too negative, need to reduce variance
        scales_to_try = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    else:  # NLL too positive, need to increase variance
        scales_to_try = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for scale in scales_to_try:
        nll = scale_realizations(scale)
        print(f"NLL with scale {scale}: {nll}")
        
        # Check if we're closer to target
        if abs(nll - target_nll) < abs(best_nll - target_nll):
            best_scale = scale
            best_nll = nll
    
    print(f"Best scale: {best_scale}, Resulting NLL: {best_nll}")
    return best_scale, best_nll
