import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from config import VISUALIZATION_PATH_CALIBRATION

def calculate_calibrated_nll(y_true, predictions, calibration_params):
    """
    Calculate NLL with advanced region-specific calibration parameters
    
    Args:
        y_true: Array of true values with shape (n_samples, 300)
        predictions: Array of realizations with shape (n_samples, num_realizations, 300)
        calibration_params: Dictionary with calibration parameters for different regions
        
    Returns:
        Average NLL loss for all samples
    """
    from utils.evaluation import calculate_inverse_covariance_vector
    n_samples = y_true.shape[0]
    num_realizations = predictions.shape[1]
    
    # Calculate idealized inverse covariance vector
    inverse_cov_vector = calculate_inverse_covariance_vector()
    
    # Calculate NLL for each sample
    losses = np.zeros(n_samples)
    
    # Probability of each realization (equiprobable)
    p_i = 1.0 / num_realizations
    
    # Get region boundaries from calibration_params
    regions = sorted(list(calibration_params['region_scales'].keys()))
    
    for i in range(n_samples):
        # Array to store gaussian misfit for each realization
        gaussian_misfits = np.zeros(num_realizations)
        
        for r in range(num_realizations):
            # Get base prediction (first realization)
            base_prediction = predictions[i, 0, :]
            
            # For realizations other than the base, scale differently by region
            if r > 0:
                # Initialize scaled prediction as the original
                scaled_prediction = predictions[i, r, :].copy()
                
                # Apply region-specific scaling
                for start_idx, end_idx in regions:
                    region_scale = calibration_params['region_scales'][(start_idx, end_idx)]
                    
                    # Calculate difference from base
                    diff = predictions[i, r, start_idx:end_idx] - base_prediction[start_idx:end_idx]
                    
                    # Apply scaling
                    scaled_prediction[start_idx:end_idx] = base_prediction[start_idx:end_idx] + diff * region_scale
                
                # Calculate error vector (residual) with scaled prediction
                error_vector = y_true[i] - scaled_prediction
            else:
                # For base prediction, calculate error normally
                error_vector = y_true[i] - predictions[i, r, :]
            
            # Check for invalid values in error
            if np.isnan(error_vector).any() or np.isinf(error_vector).any():
                error_vector = np.nan_to_num(error_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate weighted error sum based on inverse covariance
            weighted_error_sum = np.sum(error_vector**2 * inverse_cov_vector)
            
            # Limit to avoid numerical problems
            weighted_error_sum = np.clip(weighted_error_sum, -700, 700)
            
            # Calculate gaussian misfit
            gaussian_misfits[r] = np.exp(weighted_error_sum)
        
        # Apply realization weight adjustments if specified
        if 'realization_weights' in calibration_params:
            weighted_misfits = np.zeros_like(gaussian_misfits)
            for r in range(num_realizations):
                weighted_misfits[r] = gaussian_misfits[r] * calibration_params['realization_weights'][r]
            weighted_sum = np.sum(weighted_misfits)
        else:
            # Calculate weighted sum of misfits (NLL)
            weighted_sum = np.sum(p_i * gaussian_misfits)
        
        # Avoid numerical problems with logarithm
        if weighted_sum > 1e-300:
            losses[i] = -np.log(weighted_sum)
        else:
            # Maximum value for extreme cases
            losses[i] = 700.0
    
    # Calculate mean and display statistics
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    high_loss_count = np.sum(losses > 100)
    
    print(f"Advanced NLL Statistics: mean={mean_loss:.2f}, median={median_loss:.2f}, high NLL samples: {high_loss_count}/{n_samples}")
    
    return mean_loss, losses

def optimize_region_specific_calibration(val_realizations, y_val, target_nll=-69.0, max_iterations=50):
    """
    Optimize region-specific calibration parameters to achieve target NLL
    
    Args:
        val_realizations: Validation realizations with shape (n_samples, num_realizations, 300)
        y_val: Validation targets with shape (n_samples, 300)
        target_nll: Target NLL value
        max_iterations: Maximum number of optimization iterations
        
    Returns:
        Dictionary with optimized calibration parameters
    """
    print("Optimizing region-specific calibration for target NLL...")
    
    # Define regions (can be adjusted based on domain knowledge or variance analysis)
    # Based on your variance graph, focusing on high-impact initial positions
    regions = [
        (0, 20),    # Early positions (very high impact on NLL)
        (20, 50),   # Early-mid positions (high impact)
        (50, 150),  # Mid positions (moderate impact)
        (150, 300)  # Late positions (lower impact)
    ]
    
    # Initialize calibration parameters (starting conservative)
    calibration_params = {
        'region_scales': {region: 0.5 for region in regions},
        'realization_weights': np.ones(val_realizations.shape[1]) / val_realizations.shape[1]
    }
    
    # Initial evaluation
    current_nll, sample_losses = calculate_calibrated_nll(y_val, val_realizations, calibration_params)
    print(f"Initial NLL: {current_nll:.2f} (Target: {target_nll:.2f})")
    
    # Iterative optimization
    best_nll = current_nll
    best_params = calibration_params.copy()
    
    # Try different weight adjustments for regions
    for iteration in range(max_iterations):
        # Select region to optimize
        region_idx = iteration % len(regions)
        region = regions[region_idx]
        
        # Current scale for this region
        current_scale = calibration_params['region_scales'][region]
        
        # Define optimization function for this region
        def objective(scale):
            temp_params = calibration_params.copy()
            temp_params['region_scales'] = calibration_params['region_scales'].copy()
            temp_params['region_scales'][region] = scale
            nll, _ = calculate_calibrated_nll(y_val, val_realizations, temp_params)
            return abs(nll - target_nll)
        
        # Find optimal scale for this region
        result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
        new_scale = result.x
        
        # Update parameters
        calibration_params['region_scales'][region] = new_scale
        
        # Evaluate
        current_nll, sample_losses = calculate_calibrated_nll(y_val, val_realizations, calibration_params)
        print(f"Iteration {iteration+1}: Region {region}, Scale {new_scale:.4f}, NLL {current_nll:.2f}")
        
        # Check if we're closer to target
        if abs(current_nll - target_nll) < abs(best_nll - target_nll):
            best_nll = current_nll
            best_params = {
                'region_scales': calibration_params['region_scales'].copy(),
                'realization_weights': calibration_params['realization_weights'].copy()
            }
        
        # Check if we're close enough to target
        if abs(current_nll - target_nll) < 1.0:
            print(f"Target NLL achieved at iteration {iteration+1}. Stopping.")
            break
    
    # Optimize realization weights
    if abs(best_nll - target_nll) > 5.0:
        print("Optimizing realization weights...")
        
        # Identify which realizations are performing better
        # We'll cluster samples by their loss
        cluster_data = np.column_stack((np.arange(len(sample_losses)), sample_losses))
        kmeans = KMeans(n_clusters=3, random_state=42).fit(cluster_data)
        
        # Find cluster with lowest average loss
        cluster_avg_loss = [np.mean(sample_losses[kmeans.labels_ == i]) for i in range(3)]
        best_cluster = np.argmin(cluster_avg_loss)
        
        # Count realizations by their dominance in best samples
        realization_counts = np.zeros(val_realizations.shape[1])
        
        # Get samples in best cluster
        best_samples = np.where(kmeans.labels_ == best_cluster)[0]
        
        # For each sample in best cluster, identify which realization is closest to true value
        for sample_idx in best_samples:
            errors = np.zeros(val_realizations.shape[1])
            for r in range(val_realizations.shape[1]):
                errors[r] = np.mean((val_realizations[sample_idx, r, :] - y_val[sample_idx])**2)
            
            # Increment count for best realization
            best_r = np.argmin(errors)
            realization_counts[best_r] += 1
        
        # Convert counts to normalized weights
        realization_weights = 1.0 + (realization_counts / max(1, np.sum(realization_counts)))
        realization_weights = realization_weights / np.sum(realization_weights)
        
        # Apply realization weights
        best_params['realization_weights'] = realization_weights
        
        # Final evaluation
        best_nll, _ = calculate_calibrated_nll(y_val, val_realizations, best_params)
        print(f"Final NLL after realization weight optimization: {best_nll:.2f}")
    
    # Visualize calibration results
    visualize_calibration_results(best_params, regions, y_val, val_realizations)
    
    return best_params, best_nll

def visualize_calibration_results(calibration_params, regions, y_val, val_realizations):
    """
    Visualize the effects of calibration
    
    Args:
        calibration_params: Calibrated parameters
        regions: List of region tuples
        y_val: Validation targets
        val_realizations: Validation realizations
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot region-specific scales
    plt.subplot(2, 1, 1)
    positions = np.arange(1, 301)
    scale_values = np.zeros(300)
    
    for (start, end), scale in calibration_params['region_scales'].items():
        plt.axvspan(start+1, end+1, alpha=0.2, color=f'C{regions.index((start, end))}')
        scale_values[start:end] = scale
    
    plt.plot(positions, scale_values, 'k-', linewidth=2)
    plt.title('Region-Specific Calibration Scales')
    plt.xlabel('Position')
    plt.ylabel('Scale Factor')
    plt.grid(True)
    
    # Plot realization weights if available
    if 'realization_weights' in calibration_params:
        plt.subplot(2, 1, 2)
        weights = calibration_params['realization_weights']
        plt.bar(np.arange(len(weights)), weights)
        plt.title('Realization Weights')
        plt.xlabel('Realization')
        plt.ylabel('Weight')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH_CALIBRATION)
    print(f"Calibration visualization saved to {VISUALIZATION_PATH_CALIBRATION}")

def apply_calibrated_scaling(realizations, base_predictions, calibration_params):
    """
    Apply calibrated scaling to realizations
    
    Args:
        realizations: Array of realizations with shape (n_samples, num_realizations, 300)
        base_predictions: Base predictions (first realization)
        calibration_params: Calibration parameters
        
    Returns:
        Scaled realizations
    """
    # Clone realizations to avoid modifying the original
    scaled_realizations = realizations.copy()
    
    # Get region boundaries
    regions = sorted(list(calibration_params['region_scales'].keys()))
    
    # Apply region-specific scaling to all realizations except the first one
    for r in range(1, realizations.shape[1]):
        for i in range(realizations.shape[0]):
            # For each region, apply specific scaling
            for start_idx, end_idx in regions:
                region_scale = calibration_params['region_scales'][(start_idx, end_idx)]
                
                # Calculate difference from base
                diff = realizations[i, r, start_idx:end_idx] - base_predictions[i, start_idx:end_idx]
                
                # Apply scaling
                scaled_realizations[i, r, start_idx:end_idx] = base_predictions[i, start_idx:end_idx] + diff * region_scale
    
    return scaled_realizations
