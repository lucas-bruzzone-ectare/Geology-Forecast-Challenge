import time
import warnings
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_PATH, NUM_REALIZATIONS, TARGET_NLL
from utils.data_utils import load_data, prepare_data, prepare_test_data, prepare_submission
from utils.evaluation import analyze_variance_structure, calculate_nll_loss
from models.ensemble import train_advanced_ensemble_models, predict_with_advanced_ensemble
from enhanced_generator import generate_enhanced_realizations
from enhanced_calibration import optimize_region_specific_calibration, apply_calibrated_scaling
from visualization.plots import visualize_examples
import warnings
warnings.filterwarnings('ignore')  # Suprime todos os avisos

def main():
    """
    Main function to run the enhanced pipeline with improved calibration and diversity
    """
    # Garantir que avisos estejam desativados
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Start timer to measure total execution time
    start_time = time.time()
    
    # Load data
    print("\n1. Loading data...")
    train_df, _, input_cols, output_cols, test_df, test_ids = load_data(TRAIN_PATH, TEST_PATH)
    
    # Prepare data with derived features
    print("\n2. Preparing data...")
    X_train, X_val, y_train, y_val, n_derived_features = prepare_data(train_df, input_cols, output_cols)
    
    # Analyze data statistics
    print("\n3. Analyzing variance structure...")
    variance_params = analyze_variance_structure(X_train, y_train, n_derived_features)
    
    # Train advanced ensemble with XGBoost, LightGBM and stacking
    print("\n4. Training ensemble models...")
    models, scaler, key_positions, n_derived_features = train_advanced_ensemble_models(X_train, y_train, n_derived_features)
    
    # Generate base predictions for validation using advanced ensemble
    print("\n5. Generating validation predictions...")
    val_base_predictions = predict_with_advanced_ensemble(X_val, models, scaler, key_positions, n_derived_features)
    

    num_realizations = NUM_REALIZATIONS  # Use config value for now
    
    # Generate enhanced realizations with improved diversity for validation
    print(f"\n6. Generating enhanced validation realizations (n={num_realizations})...")
    val_realizations = generate_enhanced_realizations(
        X_val, val_base_predictions, variance_params, n_derived_features, num_realizations
    )
    
    # Calculate initial NLL
    print("\n7. Calculating initial NLL...")
    nll_loss = calculate_nll_loss(y_val, val_realizations)
    print(f"Initial Negative Log Likelihood (NLL): {nll_loss}")
    
    # Optimize region-specific calibration for target NLL
    print("\n8. Optimizing region-specific calibration for target NLL...")
    calibration_params, best_nll = optimize_region_specific_calibration(
        val_realizations, y_val, target_nll=TARGET_NLL, max_iterations=20
    )
    
    # Visualize some examples
    print("\n9. Visualizing examples...")
    visualize_examples(X_val, y_val, val_realizations, n_derived_features)
    
    # Prepare test data
    print("\n10. Preparing test data...")
    X_test_enhanced = prepare_test_data(test_df, input_cols, n_derived_features)
    
    # Generate base predictions for test using advanced ensemble
    print("\n11. Generating test predictions...")
    test_base_predictions = predict_with_advanced_ensemble(X_test_enhanced, models, scaler, key_positions, n_derived_features)
    
    # Generate enhanced realizations for test set
    print(f"\n12. Generating enhanced test realizations (n={num_realizations})...")
    test_realizations = generate_enhanced_realizations(
        X_test_enhanced, test_base_predictions, variance_params, n_derived_features, num_realizations
    )
    
    # Apply optimized calibration
    print("\n13. Applying calibrated scaling...")
    test_realizations = apply_calibrated_scaling(test_realizations, test_base_predictions, calibration_params)
    
    # Prepare and save submission file
    print("\n14. Preparing submission file...")
    submission_path = SUBMISSION_PATH.replace(".csv", "_enhanced.csv")
    prepare_submission(test_ids, test_realizations, submission_path)
    
    # Calculate and display execution time
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nProcess completed successfully! Calibrated NLL: {best_nll}")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Enhanced submission saved to: {submission_path}")

if __name__ == "__main__":
    main()