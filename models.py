"""
Models and Preprocessing Module for Trading Predictor
===================================================

This module contains all model creation, training functions, and preprocessing logic
for the trading prediction system.

Features include:
- Advanced preprocessing with adaptive scaling and feature selection
- Multiple ML algorithms with optimized hyperparameters
- Ensemble methods and voting classifiers
- Time-series cross-validation
- Support for XGBoost and LightGBM (if available)
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import logging
from typing import Dict, Any, Optional

# Scikit-learn imports
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, RFE
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit

# Optional advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Initialize logger
logger = logging.getLogger(__name__)

class AdvancedPreprocessor:
    """Advanced preprocessing pipeline with adaptive methods"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.outlier_bounds = {}
        self.outlier_stats = {}
        self.scalers = {}
        self.scaling_methods = {}
        self.variance_selector = None
        self.correlation_mask = None
        self.feature_selector = None
        self.rfe_selector = None
        self.use_rfe = False
        
    def advanced_preprocessing(self, X, y=None, fit=True):
        """Enhanced preprocessing with better outlier handling and feature selection"""
        print("🔧 Advanced preprocessing...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            print("📊 Analyzing data distribution and outliers...")
            # Enhanced outlier detection using multiple methods
            self.outlier_bounds = {}
            self.outlier_stats = {}
            
            for col in numeric_cols:
                col_data = X[col].dropna()
                if len(col_data) == 0:
                    continue
                
                # Method 1: IQR-based (more conservative)
                Q1 = col_data.quantile(0.05)
                Q3 = col_data.quantile(0.95)
                IQR = Q3 - Q1
                iqr_lower = Q1 - 2.0 * IQR  # More conservative multiplier
                iqr_upper = Q3 + 2.0 * IQR
                
                # Method 2: Z-score based (for normal distributions)
                mean_val = col_data.mean()
                std_val = col_data.std()
                z_lower = mean_val - 3 * std_val
                z_upper = mean_val + 3 * std_val
                
                # Method 3: Percentile-based (most robust)
                perc_lower = col_data.quantile(0.01)
                perc_upper = col_data.quantile(0.99)
                
                # Use the most conservative bounds
                final_lower = max(iqr_lower, z_lower, perc_lower)
                final_upper = min(iqr_upper, z_upper, perc_upper)
                
                self.outlier_bounds[col] = (final_lower, final_upper)
                self.outlier_stats[col] = {
                    'mean': mean_val,
                    'std': std_val,
                    'median': col_data.median(),
                    'skew': skew(col_data),
                    'kurt': kurtosis(col_data)
                }
        
        # Apply outlier bounds with smart handling
        print("🧹 Handling outliers...")
        X_processed = X.copy()
        outlier_counts = {}
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X_processed.columns:
                original_outliers = ((X_processed[col] < lower) | (X_processed[col] > upper)).sum()
                outlier_counts[col] = original_outliers
                
                # Smart clipping: use median for extreme outliers, gradual adjustment for mild ones
                col_data = X_processed[col]
                median_val = self.outlier_stats[col]['median']
                
                # For extreme outliers, replace with median
                extreme_mask = (col_data < lower * 0.5) | (col_data > upper * 1.5)
                X_processed.loc[extreme_mask, col] = median_val
                
                # For mild outliers, use clipping
                mild_mask = ~extreme_mask
                X_processed.loc[mild_mask, col] = X_processed.loc[mild_mask, col].clip(lower, upper)
        
        if fit:
            print("📏 Selecting optimal scaling method...")
            # Choose scaling method based on data distribution
            self.scalers = {}
            self.scaling_methods = {}
            
            for col in numeric_cols:
                if col in self.outlier_stats:
                    skew_val = abs(self.outlier_stats[col]['skew'])
                    kurt_val = abs(self.outlier_stats[col]['kurt'])
                    
                    # Choose scaler based on distribution characteristics
                    if skew_val > 2 or kurt_val > 7:  # Highly skewed or heavy-tailed
                        scaler = RobustScaler()
                        method = 'robust'
                    elif skew_val > 1:  # Moderately skewed
                        scaler = RobustScaler()
                        method = 'robust'
                    else:  # Approximately normal
                        scaler = StandardScaler()
                        method = 'standard'
                    
                    self.scalers[col] = scaler
                    self.scaling_methods[col] = method
            
            # Fit scalers
            for col, scaler in self.scalers.items():
                if col in X_processed.columns:
                    scaler.fit(X_processed[[col]])
            
            if y is not None:
                print("🎯 Enhanced feature selection...")
                
                # Apply scaling
                X_scaled = X_processed.copy()
                for col, scaler in self.scalers.items():
                    if col in X_scaled.columns:
                        X_scaled[col] = scaler.transform(X_scaled[[col]]).flatten()
                
                # Step 1: Remove low-variance features
                print("   📊 Removing low-variance features...")
                self.variance_selector = VarianceThreshold(threshold=0.005)  # More strict
                X_var = self.variance_selector.fit_transform(X_scaled[numeric_cols])
                remaining_features = numeric_cols[self.variance_selector.get_support()]
                
                # Step 2: Remove highly correlated features
                print("   🔗 Removing highly correlated features...")
                if len(remaining_features) > 1:
                    corr_matrix = pd.DataFrame(X_var, columns=remaining_features).corr().abs()
                    upper_triangle = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    
                    # Find features with correlation > 0.95
                    high_corr_features = [column for column in upper_triangle.columns
                                        if any(upper_triangle[column] > 0.95)]
                    
                    # Remove highly correlated features
                    features_to_keep = [f for f in remaining_features if f not in high_corr_features]
                    self.correlation_mask = np.array([f in features_to_keep for f in remaining_features])
                    X_var = X_var[:, self.correlation_mask]
                    remaining_features = np.array(features_to_keep)
                else:
                    self.correlation_mask = np.ones(len(remaining_features), dtype=bool)
                
                # Step 3: Mutual information feature selection
                print("   🧠 Mutual information feature selection...")
                n_features = min(50, max(20, X_var.shape[1] // 2))  # Adaptive feature count
                self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
                X_selected = self.feature_selector.fit_transform(X_var, y)
                
                # Step 4: Recursive feature elimination with cross-validation (optional for top features)
                if X_selected.shape[1] > 30:
                    print("   🔄 Recursive feature elimination...")
                    from sklearn.ensemble import RandomForestClassifier
                    rf_selector = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
                    
                    final_n_features = min(30, X_selected.shape[1])
                    self.rfe_selector = RFE(estimator=rf_selector, n_features_to_select=final_n_features, step=0.1)
                    self.rfe_selector.fit(X_selected, y)
                    self.use_rfe = True
                else:
                    self.use_rfe = False
                
                print(f"📊 Feature selection pipeline: {len(numeric_cols)} -> {X_selected.shape[1]} -> {final_n_features if hasattr(self, 'rfe_selector') else X_selected.shape[1]} features")
        
        # Apply preprocessing pipeline
        print("⚙️ Applying preprocessing pipeline...")
        
        # Apply scaling
        X_scaled = X_processed.copy()
        for col, scaler in self.scalers.items():
            if col in X_scaled.columns:
                X_scaled[col] = scaler.transform(X_scaled[[col]]).flatten()
        
        # Apply feature selection steps
        X_var = self.variance_selector.transform(X_scaled[numeric_cols])
        X_var = X_var[:, self.correlation_mask]
        X_selected = self.feature_selector.transform(X_var)
        
        if hasattr(self, 'use_rfe') and self.use_rfe:
            X_final = self.rfe_selector.transform(X_selected)
        else:
            X_final = X_selected
        
        return X_final

def create_actionable_models(random_state: int = 42):
    """Create advanced models optimized for actionable predictions"""
    print("🤖 Creating advanced actionable models...")
    
    models = {}
    
    # Enhanced Random Forest with better hyperparameters
    models['rf_enhanced'] = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample',
        random_state=random_state, n_jobs=-1, bootstrap=True,
        oob_score=True, max_samples=0.8
    )
    
    # Enhanced Extra Trees with diversity focus
    models['et_enhanced'] = ExtraTreesClassifier(
        n_estimators=300, max_depth=25, min_samples_split=2,
        min_samples_leaf=1, max_features='log2', class_weight='balanced_subsample',
        random_state=random_state, n_jobs=-1, bootstrap=True,
        oob_score=True, max_samples=0.9
    )
    
    # Enhanced Gradient Boosting
    models['gb_enhanced'] = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=6,
        min_samples_split=4, min_samples_leaf=2, subsample=0.85,
        max_features='sqrt', random_state=random_state,
        validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
    )
    
    # Enhanced Logistic Regression with regularization
    models['lr_enhanced'] = LogisticRegression(
        C=0.1, penalty='elasticnet', l1_ratio=0.5, class_weight='balanced',
        random_state=random_state, max_iter=3000, solver='saga'
    )
    
    # Enhanced MLP with better architecture
    models['mlp_enhanced'] = MLPClassifier(
        hidden_layer_sizes=(150, 100, 50), activation='relu', solver='adam',
        alpha=0.001, learning_rate='adaptive', max_iter=500,
        random_state=random_state, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=15, tol=1e-4
    )
    
    # Ridge Classifier for linear baseline
    models['ridge'] = RidgeClassifier(
        alpha=1.0, class_weight='balanced', random_state=random_state
    )
    
    # AdaBoost for boosting diversity
    models['ada_boost'] = AdaBoostClassifier(
        n_estimators=100, learning_rate=0.8, algorithm='SAMME.R',
        random_state=random_state
    )
    
    # XGBoost if available
    if XGBOOST_AVAILABLE:
        print("   🚀 Adding XGBoost models...")
        
        # Standard XGBoost
        models['xgb_standard'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=random_state,
            n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False
        )
        
        # XGBoost with different hyperparameters for diversity
        models['xgb_deep'] = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.7, colsample_bylevel=0.9,
            reg_alpha=0.05, reg_lambda=0.5, random_state=random_state + 1,
            n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False
        )
        
        # XGBoost optimized for imbalanced classes
        models['xgb_balanced'] = xgb.XGBClassifier(
            n_estimators=250, max_depth=7, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.9, colsample_bylevel=0.8,
            reg_alpha=0.2, reg_lambda=1.5, scale_pos_weight=1.5,
            random_state=random_state + 2, n_jobs=-1,
            eval_metric='mlogloss', use_label_encoder=False
        )
    
    # LightGBM if available
    if LIGHTGBM_AVAILABLE:
        print("   ⚡ Adding LightGBM models...")
        
        # Standard LightGBM
        models['lgb_standard'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=random_state, n_jobs=-1,
            objective='multiclass', metric='multi_logloss', verbose=-1
        )
        
        # LightGBM with leaf-wise growth
        models['lgb_leafwise'] = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=50, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.7, reg_alpha=0.05,
            reg_lambda=0.5, random_state=random_state + 3, n_jobs=-1,
            objective='multiclass', metric='multi_logloss', verbose=-1,
            boosting_type='gbdt', min_child_samples=10
        )
        
        # LightGBM optimized for feature importance
        models['lgb_dart'] = lgb.LGBMClassifier(
            n_estimators=250, max_depth=8, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.9, reg_alpha=0.2,
            reg_lambda=1.5, random_state=random_state + 4, n_jobs=-1,
            objective='multiclass', metric='multi_logloss', verbose=-1,
            boosting_type='dart', drop_rate=0.1, skip_drop=0.5
        )
    
    # Create diverse ensemble combinations
    print("   🎭 Creating ensemble models...")
    
    # Ensemble 1: Tree-based models
    tree_estimators = [
        ('rf', models['rf_enhanced']),
        ('et', models['et_enhanced']),
        ('gb', models['gb_enhanced'])
    ]
    
    if XGBOOST_AVAILABLE:
        tree_estimators.append(('xgb', models['xgb_standard']))
    if LIGHTGBM_AVAILABLE:
        tree_estimators.append(('lgb', models['lgb_standard']))
    
    models['ensemble_trees'] = VotingClassifier(
        estimators=tree_estimators, voting='soft'
    )
    
    # Ensemble 2: Diverse algorithms
    diverse_estimators = [
        ('rf', models['rf_enhanced']),
        ('lr', models['lr_enhanced']),
        ('mlp', models['mlp_enhanced']),
        ('ada', models['ada_boost'])
    ]
    
    if XGBOOST_AVAILABLE:
        diverse_estimators.append(('xgb', models['xgb_balanced']))
    
    models['ensemble_diverse'] = VotingClassifier(
        estimators=diverse_estimators, voting='soft'
    )
    
    # Ensemble 3: All available models (if we have enough)
    if len(models) >= 5:
        all_estimators = []
        model_keys = ['rf_enhanced', 'et_enhanced', 'gb_enhanced', 'lr_enhanced']
        
        if XGBOOST_AVAILABLE:
            model_keys.append('xgb_standard')
        if LIGHTGBM_AVAILABLE:
            model_keys.append('lgb_standard')
        
        for key in model_keys:
            if key in models:
                all_estimators.append((key.split('_')[0], models[key]))
        
        if len(all_estimators) >= 3:
            models['ensemble_all'] = VotingClassifier(
                estimators=all_estimators, voting='soft'
            )
    
    print(f"✅ Created {len(models)} advanced models")
    
    # Print model summary
    model_types = {}
    for name, model in models.items():
        model_type = type(model).__name__
        if model_type not in model_types:
            model_types[model_type] = 0
        model_types[model_type] += 1
    
    print("📊 Model distribution:")
    for model_type, count in model_types.items():
        print(f"   {model_type}: {count}")
    
    return models

def time_series_cross_validation(X, y, model, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    # Import evaluation function
    from evaluate import evaluate_model_actionable
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit model on fold
        model_fold = type(model)(**model.get_params())
        model_fold.fit(X_train_fold, y_train_fold)
        
        # Predict and evaluate
        y_pred_fold = model_fold.predict(X_val_fold)
        fold_metrics = evaluate_model_actionable(y_val_fold, y_pred_fold)
        cv_scores.append(fold_metrics)
    
    # Calculate mean and std of CV scores
    cv_results = {}
    for metric in cv_scores[0].keys():
        scores = [fold[metric] for fold in cv_scores]
        cv_results[f'{metric}_mean'] = np.mean(scores)
        cv_results[f'{metric}_std'] = np.std(scores)
    
    return cv_results