import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from config import SEED

class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    Custom stacking regressor with more advanced features than sklearn's StackingRegressor.
    This implementation uses out-of-fold predictions for the meta-model training.
    
    Args:
        base_models: List of base models
        meta_model: Model to use as meta-learner
        n_folds: Number of cross-validation folds
        use_features_in_meta: Whether to use original features in meta-model
    """
    def __init__(self, base_models, meta_model, n_folds=5, use_features_in_meta=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_features_in_meta = use_features_in_meta
        self.base_models_ = None
        self.meta_model_ = None
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        
    def fit(self, X, y):
        """
        Fit the stacking regressor
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            self
        """
        self.base_models_ = [clone(model) for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        
        # Generate out-of-fold predictions for training meta-model
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        out_of_fold_preds = np.zeros((n_samples, n_models))
        
        # For each base model, generate out-of-fold predictions
        for i, model in enumerate(self.base_models_):
            oof_pred = np.zeros(n_samples)
            
            # For each fold
            for train_idx, valid_idx in self.kf.split(X):
                # Split data
                X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
                y_train_fold = y[train_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                oof_pred[valid_idx] = model.predict(X_valid_fold)
            
            # Store out-of-fold predictions
            out_of_fold_preds[:, i] = oof_pred
            
            # Retrain on all data
            model.fit(X, y)
        
        # Prepare meta features
        if self.use_features_in_meta:
            meta_features = np.hstack((out_of_fold_preds, X))
        else:
            meta_features = out_of_fold_preds
        
        # Train meta model
        self.meta_model_.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the stacking regressor
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        # Make predictions with base models
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_
        ])
        
        # Add original features if specified
        if self.use_features_in_meta:
            meta_features = np.hstack((meta_features, X))
        
        # Make final prediction
        return self.meta_model_.predict(meta_features)
