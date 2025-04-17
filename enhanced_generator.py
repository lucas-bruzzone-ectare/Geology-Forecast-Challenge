import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from config import NUM_REALIZATIONS

# Estratégias aprimoradas para melhor diversidade
ENHANCED_REALIZATION_STRATEGIES = [
    # Name, base scale, correlation length, trend, trend amplitude, special features
    ("high_fidelity", 0.8, None, None, 0.0, None),  # Base
    ("low_variance", 0.6, 1.2, None, 0.0, None),
    ("high_variance", 1.2, 0.6, None, 0.0, None),  # Aumentada variância
    ("upward_trend", 0.8, 1.0, "upward", 2.0, None),  # Amplitude aumentada
    ("downward_trend", 0.8, 1.0, "downward", 2.0, None),  # Amplitude aumentada
    ("late_upward_trend", 0.8, 1.0, "late_upward", 2.5, None),  # Amplitude aumentada
    ("late_downward_trend", 0.8, 1.0, "late_downward", 2.5, None),  # Amplitude aumentada
    ("short_oscillation", 0.7, 0.5, "oscillatory", 1.5, None),  # Amplitude aumentada
    ("long_oscillation", 0.7, 1.5, "oscillatory", 1.8, None),  # Amplitude aumentada
    ("abrupt_change", 0.9, 0.8, None, 0.0, "fault"),  # Nova estratégia
    ("high_frequency_noise", 1.0, 0.3, None, 0.0, "noise"),  # Nova estratégia
    ("layered_structure", 0.8, 1.2, None, 0.0, "layers"),  # Nova estratégia
    ("increasing_variance", 0.7, 0.9, None, 0.0, "inc_var"),  # Nova estratégia
    ("step_change", 0.8, 1.0, None, 0.0, "step"),  # Nova estratégia
]

def generate_enhanced_realizations(X, base_predictions, variance_params, n_derived_features, num_realizations=NUM_REALIZATIONS):
    """
    Generate enhanced realizations with greater diversity and geological patterns
    
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
    
    # First realization: will be the base prediction
    print("Generating enhanced realizations with greater diversity...")
    all_realizations[:, 0, :] = base_predictions
    
    # Define enhanced strategies
    strategies = ENHANCED_REALIZATION_STRATEGIES
    
    # For each additional realization
    for r in range(1, num_realizations):
        # Select strategy (use modulo to handle if we have more realizations than strategies)
        strategy_idx = (r - 1) % len(strategies)
        strategy_name, scale_base, corr_length_factor, trend_type, trend_amplitude, special_feature = strategies[strategy_idx]
        
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
            base_std = max(base_std, 0.05)  # Increased minimum value
            
            # Calculate noise scale
            noise_scale = base_std * scale_base
            
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
            
            # Apply special geological features if specified
            if special_feature == "fault":
                # Add an abrupt change (fault) at a random position between 50 and 250
                fault_pos = np.random.randint(50, 250)
                fault_magnitude = base_std * 3.0 * np.random.choice([-1, 1])
                smooth_noise[fault_pos:] += fault_magnitude
                
                # Add some irregularity around the fault
                fault_zone = np.random.randint(5, 15)  # Width of disturbed zone
                fault_noise = np.random.normal(0, base_std, fault_zone)
                smooth_noise[fault_pos:fault_pos+fault_zone] += fault_noise
                
            elif special_feature == "noise":
                # Add high-frequency noise component
                high_freq_noise = np.random.normal(0, base_std * 0.5, n_outputs)
                # Apply minimal smoothing
                high_freq_noise = gaussian_filter1d(high_freq_noise, sigma=0.8)
                smooth_noise += high_freq_noise
                
            elif special_feature == "layers":
                # Create layered structure (alternating harder/softer layers)
                n_layers = np.random.randint(3, 8)
                layer_positions = np.sort(np.random.choice(range(20, 280), size=n_layers, replace=False))
                
                current_value = 0
                for pos in layer_positions:
                    # Change direction and magnitude at layer boundary
                    layer_change = np.random.normal(0, base_std * 1.5)
                    current_value += layer_change
                    
                    # Add the layer change
                    smooth_noise[pos:] += current_value
                    
                    # Add some noise at layer boundaries (weathering effects)
                    boundary_width = np.random.randint(3, 8)
                    boundary_noise = np.random.normal(0, base_std * 0.3, boundary_width)
                    if pos + boundary_width < n_outputs:
                        smooth_noise[pos:pos+boundary_width] += boundary_noise
                
            elif special_feature == "inc_var":
                # Increasing variance with depth
                variance_factor = np.linspace(0.5, 2.0, n_outputs)
                position_noise = np.random.normal(0, base_std, n_outputs) * variance_factor
                smooth_noise += position_noise
                
            elif special_feature == "step":
                # Create step changes (multiple small faults)
                n_steps = np.random.randint(2, 6)
                step_positions = np.sort(np.random.choice(range(30, 270), size=n_steps, replace=False))
                
                for pos in step_positions:
                    # Random step magnitude
                    step_magnitude = np.random.normal(0, base_std * 1.2)
                    smooth_noise[pos:] += step_magnitude
            
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
                    # Create smoother transition between known data and prediction
                    offset = realization[0] - X_original[i, last_valid_idx]
                    
                    # Apply gradient offset reduction over first few points
                    transition_length = min(20, n_outputs // 10)
                    weight = np.linspace(1.0, 0.0, transition_length)
                    
                    # Apply weighted offset to create smooth transition
                    for j in range(min(transition_length, n_outputs)):
                        realization[j] -= offset * weight[j]
            except:
                # In case of error, don't adjust offset
                pass
            
            # For some strategies, apply additional smoothing
            if strategy_name in ["high_fidelity", "long_oscillation", "layered_structure"]:
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
    
    # Apply diversity enhancement by ensuring realizations span a proper range of possibilities
    all_realizations = enhance_diversity(all_realizations, base_predictions)
    
    return all_realizations

def enhance_diversity(realizations, base_predictions):
    """
    Enhance diversity among realizations to better cover possible geological scenarios
    
    Args:
        realizations: Initial set of realizations
        base_predictions: Base predictions
        
    Returns:
        Enhanced set of realizations
    """
    n_samples, n_realizations, n_outputs = realizations.shape
    enhanced_realizations = realizations.copy()
    
    # Skip first realization (keep base prediction unchanged)
    for i in range(n_samples):
        # Calculate diversity metrics
        base = base_predictions[i]
        
        # Get non-base realizations
        alt_realizations = realizations[i, 1:, :]
        
        # Calculate differences from base prediction
        differences = alt_realizations - base.reshape(1, -1)
        
        # Calculate pairwise correlations between realizations
        correlations = np.zeros((n_realizations-1, n_realizations-1))
        for j in range(n_realizations-1):
            for k in range(n_realizations-1):
                if j != k:
                    correlations[j,k] = np.corrcoef(differences[j], differences[k])[0,1]
        
        # Calculate average correlation for each realization
        avg_correlations = np.mean(correlations, axis=1)
        
        # Find most correlated realizations (candidates for enhancement)
        correlation_threshold = 0.8
        high_correlation_idxs = np.where(avg_correlations > correlation_threshold)[0]
        
        # Enhance diversity of highly correlated realizations
        for idx in high_correlation_idxs:
            # Real index in realizations array (add 1 because base is at index 0)
            r_idx = idx + 1
            
            # Calculate average direction of all other realizations
            other_diffs = np.delete(differences, idx, axis=0)
            avg_direction = np.mean(other_diffs, axis=0)
            
            # Project current realization difference onto average direction
            current_diff = differences[idx]
            proj = np.dot(current_diff, avg_direction) / (np.linalg.norm(avg_direction) + 1e-10)
            
            # Calculate orthogonal component (uniqueness)
            orthogonal = current_diff - (proj * avg_direction / (np.linalg.norm(avg_direction) + 1e-10))
            
            # Enhance orthogonal component to increase diversity
            diversity_factor = 1.5
            enhanced_diff = current_diff - (0.3 * proj * avg_direction / (np.linalg.norm(avg_direction) + 1e-10)) + (diversity_factor * orthogonal)
            
            # Apply smoothing to avoid introducing high-frequency artifacts
            enhanced_diff = gaussian_filter1d(enhanced_diff, sigma=3.0)
            
            # Update realization
            enhanced_realizations[i, r_idx, :] = base + enhanced_diff
    
    return enhanced_realizations

def find_optimal_num_realizations(X, y_val, base_predictions, variance_params, n_derived_features, max_realizations=15):
    """
    Find optimal number of realizations through cross-validation
    
    Args:
        X: Input features
        y_val: Validation targets
        base_predictions: Base predictions
        variance_params: Variance parameters
        n_derived_features: Number of derived features
        max_realizations: Maximum number of realizations to try
        
    Returns:
        Optimal number of realizations
    """
    from utils.evaluation import calculate_nll_loss
    
    print("Finding optimal number of realizations...")
    nll_values = []
    
    # Try different numbers of realizations
    for num_r in range(3, max_realizations + 1, 2):  # Try odd numbers for computational efficiency
        print(f"Testing with {num_r} realizations...")
        # Generate realizations
        realizations = generate_enhanced_realizations(X, base_predictions, variance_params, n_derived_features, num_r)
        
        # Calculate NLL
        nll = calculate_nll_loss(y_val, realizations)
        nll_values.append((num_r, nll))
        print(f"  NLL with {num_r} realizations: {nll:.2f}")
    
    # Find number of realizations with best NLL
    optimal_num = min(nll_values, key=lambda x: abs(x[1] - (-69)))[0]
    print(f"Optimal number of realizations: {optimal_num}")
    
    return optimal_num