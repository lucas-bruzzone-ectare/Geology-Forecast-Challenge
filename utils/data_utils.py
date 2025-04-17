import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings
from features.feature_engineering import create_derived_features
from config import SEED

# Ignore warnings for clarity
warnings.filterwarnings("ignore")

def load_data(train_path, test_path=None):
    """
    Load and prepare the dataset
    
    Args:
        train_path: Path to training data file
        test_path: Path to test data file (optional)
        
    Returns:
        Training dataframe, IDs, input columns, output columns
        And test dataframe and IDs if test_path is provided
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    # Extract IDs
    train_ids = train_df['geology_id'].values
    
    # Extract input columns (-49 to 0)
    input_cols = [str(i) for i in range(-49, 1)]
    
    # Extract output columns (1 to 300)
    output_cols = [str(i) for i in range(1, 301)]
    
    # Handle test data if provided
    if test_path:
        test_df = pd.read_csv(test_path)
        test_ids = test_df['geology_id'].values
        
        return train_df, train_ids, input_cols, output_cols, test_df, test_ids
    
    return train_df, train_ids, input_cols, output_cols

def prepare_data(train_df, input_cols, output_cols):
    """
    Prepare data with improved imputation and derived features
    
    Args:
        train_df: Training dataframe
        input_cols: List of input column names
        output_cols: List of output column names
        
    Returns:
        X_train, X_val, y_train, y_val, number of derived features
    """
    print("Preparing data...")
    
    # Extract matrices
    X = train_df[input_cols].values
    y = train_df[output_cols].values
    
    # Use KNNImputer to better preserve spatial structure
    print("Handling missing values with KNNImputer...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed = imputer.fit_transform(X)
    
    # Check if NaNs or infinites still exist
    if np.isnan(X_imputed).any() or np.isinf(X_imputed).any():
        print("WARNING: NaN or infinite values still exist after imputation!")
        X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create derived features
    print("Creating derived features...")
    derived_features = create_derived_features(X_imputed)
    
    # Combine with original features
    X_enhanced = np.hstack((X_imputed, derived_features))
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_enhanced, y, test_size=0.2, random_state=SEED)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val, derived_features.shape[1]

def prepare_test_data(test_df, input_cols, n_derived_features):
    """
    Prepare test data with imputation and derived features
    
    Args:
        test_df: Test dataframe
        input_cols: List of input column names
        n_derived_features: Number of derived features
        
    Returns:
        Enhanced test data
    """
    print("Preparing test data...")
    X_test = test_df[input_cols].values
    
    # Handle missing values in test set
    print("Handling missing values in test set...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_test_imputed = imputer.fit_transform(X_test)
    
    # Check and fix residual NaN or infinite values
    if np.isnan(X_test_imputed).any() or np.isinf(X_test_imputed).any():
        print("Fixing residual NaN/Inf values...")
        X_test_imputed = np.nan_to_num(X_test_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create derived features for test set
    print("Creating derived features for test set...")
    test_derived_features = create_derived_features(X_test_imputed)
    X_test_enhanced = np.hstack((X_test_imputed, test_derived_features))
    
    return X_test_enhanced

def prepare_submission(test_ids, all_realizations, output_path):
    """
    Prepare submission file
    
    Args:
        test_ids: Array of test IDs
        all_realizations: Array of realizations with shape (n_samples, num_realizations, 300)
        output_path: Path to save submission file
    """
    print("Preparing submission file...")
    
    # Create dictionary for DataFrame
    submission_dict = {'geology_id': test_ids}
    
    # Add columns for first realization (1 to 300)
    for i in range(300):
        submission_dict[str(i+1)] = all_realizations[:, 0, i]
    
    # Add columns for other realizations
    for r in range(1, all_realizations.shape[1]):
        for i in range(300):
            submission_dict[f'r_{r}_pos_{i+1}'] = all_realizations[:, r, i]
    
    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_dict)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
