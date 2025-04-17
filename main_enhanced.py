import time
import warnings
import joblib
from config import (
    TRAIN_PATH, TEST_PATH, SUBMISSION_PATH, 
    NUM_REALIZATIONS, TARGET_NLL, SEED
)
from utils.data_utils import (
    load_data, prepare_data, prepare_test_data, prepare_submission
)
from utils.evaluation import analyze_variance_structure, calculate_nll_loss
from models.ensemble import train_advanced_ensemble_models, predict_with_advanced_ensemble
from enhanced_generator import (
    generate_enhanced_realizations, 
    find_optimal_num_realizations
)
from enhanced_calibration import optimize_region_specific_calibration, apply_calibrated_scaling
from visualization.plots import visualize_examples
from probabilistic_model import (
    train_and_predict_with_probabilistic_model, 
    calculate_nll_loss as prob_calculate_nll_loss
)
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

def main():
    """
    Main function to run the enhanced pipeline with improved calibration, 
    ensemble models, and probabilistic modeling
    """
    # Ensure warnings are disabled
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    # Start timer to measure total execution time
    start_time = time.time()
    
    # 1. Load data
    print("\n1. Loading data...")
    train_df, _, input_cols, output_cols, test_df, test_ids = load_data(TRAIN_PATH, TEST_PATH)
    
    # 2. Prepare data with derived features
    print("\n2. Preparing data...")
    X_train, X_val, y_train, y_val, n_derived_features = prepare_data(train_df, input_cols, output_cols)
    
    # 3. Analyze data statistics
    print("\n3. Analyzing variance structure...")
    variance_params = analyze_variance_structure(X_train, y_train, n_derived_features)
    
    # 4. Train advanced ensemble models
    print("\n4. Training ensemble models...")
    models, scaler, key_positions, n_derived_features = train_advanced_ensemble_models(
        X_train, y_train, n_derived_features
    )
    
    # 5. Generate base predictions for validation using advanced ensemble
    print("\n5. Generating validation predictions...")
    val_base_predictions = predict_with_advanced_ensemble(
        X_val, models, scaler, key_positions, n_derived_features
    )
    
    # 6. Prepare test data
    print("\n6. Preparing test data...")
    X_test_enhanced = prepare_test_data(test_df, input_cols, n_derived_features)
    
    # 6. Find optimal number of realizations (optional)
    print("\n6. Finding optimal number of realizations...")
    optimal_num_realizations = find_optimal_num_realizations(
        X_val, y_val, val_base_predictions, variance_params, 
        n_derived_features, max_realizations=15
    )
    num_realizations = optimal_num_realizations
    print(f"Optimal number of realizations: {num_realizations}")
    
    # 7. Generate base predictions for test using advanced ensemble
    print("\n7. Generating test base predictions...")
    test_base_predictions = predict_with_advanced_ensemble(
        X_test_enhanced, models, scaler, key_positions, n_derived_features
    )
    
    # 8. Use probabilistic model for generating realizations
    print(f"\n8. Generating enhanced realizations with probabilistic model (n={num_realizations})...")
    prob_model, val_realizations, test_realizations = train_and_predict_with_probabilistic_model(
        X_train, y_train, X_val, X_test_enhanced, n_derived_features, 
        num_realizations=num_realizations
    )
    
    # 9. Calculate initial NLL to compare approaches
    print("\n9. Calculating initial NLL...")
    prob_nll = prob_calculate_nll_loss(y_val, val_realizations)
    print(f"Probabilistic Model Negative Log Likelihood: {prob_nll}")
    
    # 10. Optimize region-specific calibration
    print("\n10. Optimizing region-specific calibration...")
    calibration_params, best_nll = optimize_region_specific_calibration(
        val_realizations, y_val, target_nll=TARGET_NLL, max_iterations=20
    )
    
    # 11. Visualize some examples
    print("\n11. Visualizing examples...")
    visualize_examples(X_val, y_val, val_realizations, n_derived_features)
    
    # 12. Apply calibrated scaling to test realizations
    print("\n12. Applying calibrated scaling...")
    test_realizations = apply_calibrated_scaling(
        test_realizations, test_base_predictions, calibration_params
    )
    
    # 13. Prepare and save submission file
    print("\n13. Preparing submission file...")
    submission_path = SUBMISSION_PATH.replace(".csv", "_probabilistic.csv")
    prepare_submission(test_ids, test_realizations, submission_path)
    
    # 14. Save models and key artifacts
    print("\n14. Saving models and artifacts...")
    model_artifacts = {
        'ensemble_models': models,
        'probabilistic_model': prob_model,
        'scaler': scaler,
        'key_positions': key_positions,
        'n_derived_features': n_derived_features,
        'calibration_params': calibration_params,
        'variance_params': variance_params
    }
    joblib.dump(model_artifacts, 'model_artifacts_probabilistic.pkl')
    
    # Calculate and display execution time
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nProcess completed successfully!")
    print(f"Calibrated NLL: {best_nll}")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Enhanced probabilistic submission saved to: {submission_path}")

if __name__ == "__main__":
    main()