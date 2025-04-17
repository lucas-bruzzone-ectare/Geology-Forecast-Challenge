import numpy as np
import matplotlib.pyplot as plt
from config import VISUALIZATION_PATH_PREDICTIONS

def visualize_examples(X_val, y_val, val_realizations, n_derived_features, num_examples=3):
    """
    Visualize some example predictions with multiple realizations
    
    Args:
        X_val: Validation features
        y_val: Validation targets
        val_realizations: Validation realizations
        n_derived_features: Number of derived features
        num_examples: Number of examples to visualize
    """
    print("Visualizing examples...")
    
    # Use only original features
    X_val_original = X_val[:, :-n_derived_features]
    
    # Choose some random examples
    indices = np.random.choice(X_val_original.shape[0], num_examples, replace=False)
    
    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(indices):
        plt.subplot(num_examples, 1, i+1)
        
        # Known data (input)
        plt.plot(range(-49, 1), X_val_original[idx], 'b-', label='Known Data')
        
        # True value (target)
        plt.plot(range(1, 301), y_val[idx], 'g-', label='Ground Truth')
        
        # Multiple realizations (predictions)
        for r in range(val_realizations.shape[1]):
            if r == 0:
                plt.plot(range(1, 301), val_realizations[idx, r, :], 'r-', alpha=0.7, label='Base Prediction')
            else:
                plt.plot(range(1, 301), val_realizations[idx, r, :], 'r-', alpha=0.3)
        
        plt.axvline(x=0, color='k', linestyle='--')
        plt.title(f'Example #{idx}')
        plt.xlabel('Position X')
        plt.ylabel('Coordinate Z')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH_PREDICTIONS)
    print(f"Visualization saved as '{VISUALIZATION_PATH_PREDICTIONS}'")
