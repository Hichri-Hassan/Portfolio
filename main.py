"""
Main Orchestration Module for Trading Predictor
=============================================

This is the main module that orchestrates the entire trading prediction pipeline.
It imports and coordinates all other modules: features, labeling, models, and evaluate.

This module provides:
- Complete pipeline orchestration
- Data validation and loading
- Integration of all components
- Experiment tracking and logging
- Results aggregation and reporting
"""

import os
import sys
import json
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Import our custom modules
from features import comprehensive_feature_engineering
from labeling import create_actionable_targets
from models import AdvancedPreprocessor, create_actionable_models, time_series_cross_validation
from evaluate import (evaluate_model_actionable, plot_confusion_matrix, 
                     print_detailed_classification_report, plot_feature_importance,
                     plot_model_comparison, plot_prediction_distribution, 
                     generate_evaluation_summary)

# Configuration and constants
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

# Global configuration for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Optional advanced ML libraries availability check
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up comprehensive logging for the trading predictor.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs to console only.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('TradingPredictor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_input_data(df: pd.DataFrame, required_columns: List[str],
                       allow_null_columns: bool = True, min_rows: int = 100) -> None:
    """
    Validate input data for the trading predictor with flexible validation rules.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        allow_null_columns: Whether to allow columns with all null values
        min_rows: Minimum number of rows required
    
    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Check for required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check if required columns have all null values
    required_null_columns = []
    for col in required_columns:
        if col in df.columns and df[col].isnull().all():
            required_null_columns.append(col)
    
    if required_null_columns:
        raise ValueError(f"Required columns with all null values: {required_null_columns}")
    
    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(f"Dataset too small. Has {len(df)} rows, minimum {min_rows} required.")
    
    # Check for completely empty dataset
    if df.select_dtypes(include=[np.number]).empty:
        raise ValueError("No numeric columns found in dataset")
    
    logger.info(f"Input validation passed. Shape: {df.shape}")

def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Set seeds for optional libraries if available
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to {seed}")

class ExperimentLogger:
    """
    Automated experiment logging system for reproducibility and analysis.
    
    This class handles:
    - Parameter logging and serialization
    - Metrics tracking across experiments
    - Model artifact saving
    - Visualization export
    - Experiment comparison utilities
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the current experiment
            base_dir: Base directory for experiment storage
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # Create experiment directory
        self.experiment_dir = Path(base_dir) / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'random_seed': RANDOM_SEED,
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
            'parameters': {},
            'metrics': {},
            'artifacts': []
        }
        
        logger.info(f"Initialized experiment: {self.experiment_id}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        self.metadata['parameters'].update(params)
        logger.info(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log experiment metrics."""
        self.metadata['metrics'].update(metrics)
        logger.info(f"Logged metrics: {list(metrics.keys())}")
    
    def save_artifact(self, obj: Any, filename: str, artifact_type: str = "pickle") -> str:
        """
        Save experiment artifact.
        
        Args:
            obj: Object to save
            filename: Filename for the artifact
            artifact_type: Type of artifact (pickle, json, csv, etc.)
        
        Returns:
            Path to saved artifact
        """
        filepath = self.experiment_dir / filename
        
        try:
            if artifact_type == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(obj, f)
            elif artifact_type == "json":
                with open(filepath, 'w') as f:
                    json.dump(obj, f, indent=2, default=str)
            elif artifact_type == "csv" and hasattr(obj, 'to_csv'):
                obj.to_csv(filepath, index=True)
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
            self.metadata['artifacts'].append({
                'filename': filename,
                'type': artifact_type,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Saved artifact: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save artifact {filename}: {e}")
            raise
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """Save matplotlib figure as artifact."""
        filepath = self.experiment_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        self.metadata['artifacts'].append({
            'filename': filename,
            'type': 'plot',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Saved plot: {filename}")
        return str(filepath)
    
    def finalize_experiment(self) -> str:
        """Finalize experiment and save metadata."""
        metadata_path = self.experiment_dir / "experiment_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        logger.info(f"Experiment finalized: {self.experiment_id}")
        return str(metadata_path)

class TradingPredictorTests:
    """
    Unit tests for the trading predictor components.
    """
    
    def __init__(self):
        """Initialize test suite."""
        self.logger = logging.getLogger(f'{__name__}.TradingPredictorTests')
        self.test_results = {}
    
    def create_sample_data(self, n_samples: int = 1000, n_tickers: int = 5) -> pd.DataFrame:
        """
        Create synthetic financial data for testing.
        
        Args:
            n_samples: Number of data points per ticker
            n_tickers: Number of different tickers
            
        Returns:
            DataFrame with synthetic financial data
        """
        np.random.seed(RANDOM_SEED)
        
        data = []
        tickers = [f'STOCK_{i:02d}' for i in range(n_tickers)]
        
        for ticker in tickers:
            # Generate synthetic price data with realistic patterns
            dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
            
            # Random walk with drift for price
            returns = np.random.normal(0.0005, 0.02, n_samples)  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))  # Price from returns
            
            # Generate OHLC data
            high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
            volume = np.random.lognormal(10, 1, n_samples)
            
            ticker_data = pd.DataFrame({
                'Date': dates,
                'Ticker': ticker,
                'Open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
                'High': high,
                'Low': low,
                'Close': prices,
                'Volume': volume
            })
            
            data.append(ticker_data)
        
        return pd.concat(data, ignore_index=True)
    
    def test_data_validation(self) -> bool:
        """Test input data validation functionality."""
        self.logger.info("Testing data validation...")
        
        try:
            # Test valid data
            valid_data = self.create_sample_data(100, 2)
            required_cols = ['Date', 'Ticker', 'Close', 'Volume']
            validate_input_data(valid_data, required_cols)
            
            # Test invalid data cases
            test_cases = [
                (None, "None input"),
                (pd.DataFrame(), "Empty DataFrame"),
                (valid_data.drop('Close', axis=1), "Missing required column"),
                (valid_data.head(50), "Too few samples")
            ]
            
            for invalid_data, description in test_cases:
                try:
                    validate_input_data(invalid_data, required_cols)
                    self.logger.error(f"Validation should have failed for: {description}")
                    return False
                except ValueError:
                    pass  # Expected behavior
            
            self.logger.info("Data validation tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation test failed: {e}")
            return False
    
    def test_feature_engineering(self) -> bool:
        """Test feature engineering functionality."""
        self.logger.info("Testing feature engineering...")
        
        try:
            # Create test data
            test_data = self.create_sample_data(200, 2)
            
            # Test feature engineering
            features_df = comprehensive_feature_engineering(test_data)
            
            # Validate results
            assert features_df.shape[0] == test_data.shape[0], "Row count mismatch"
            assert features_df.shape[1] > test_data.shape[1], "No new features created"
            
            # Check for expected feature types
            expected_features = ['Return_1D', 'SMA_20', 'RSI_14', 'MACD', 'BB_Position_20']
            for feature in expected_features:
                assert feature in features_df.columns, f"Missing expected feature: {feature}"
            
            # Check for no infinite values
            assert not np.isinf(features_df.select_dtypes(include=[np.number])).any().any(), "Infinite values found"
            
            self.logger.info("Feature engineering tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Feature engineering test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all unit tests and return results.
        
        Returns:
            Dictionary with test names and their pass/fail status
        """
        self.logger.info("Starting comprehensive unit tests...")
        
        tests = [
            ('data_validation', self.test_data_validation),
            ('feature_engineering', self.test_feature_engineering)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                self.logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        self.logger.info(f"Unit tests completed: {passed}/{total} passed")
        
        if passed == total:
            self.logger.info("🎉 All unit tests passed!")
        else:
            self.logger.warning(f"⚠️ {total - passed} unit tests failed")
        
        return results

def train_and_evaluate_actionable(df, target_buy_pct=20, target_sell_pct=20, test_size=0.2, validation_size=0.1):
    """Enhanced training and evaluation pipeline with robust validation"""
    print("🚀 Starting enhanced actionable training pipeline...")
    
    # Prepare features and target
    exclude_cols = ['Date', 'Ticker', 'Target', 'Action_Score', 'Market_Vol_Regime',
                   'Adjusted_Buy_Pct', 'Adjusted_Sell_Pct']
    
    # Add all future return and risk-adjusted return columns to exclude
    future_cols = [col for col in df.columns if col.startswith(('Future_Return_', 'Risk_Adj_Return_'))]
    exclude_cols.extend(future_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    y = df['Target'].copy()
    dates = df['Date'].copy() if 'Date' in df.columns else None
    
    # Remove rows with missing targets
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    if dates is not None:
        dates = dates[valid_mask]
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Target distribution: {dict(pd.Series(y).value_counts().sort_index())}")
    
    # Enhanced time-based splits
    total_size = len(X)
    train_size = int(total_size * (1 - test_size - validation_size))
    val_size = int(total_size * validation_size)
    
    # Create splits
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size + val_size]
    y_test = y.iloc[train_size + val_size:]
    
    print(f"📊 Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # Check class distribution in each split
    print("📊 Class distribution by split:")
    for split_name, split_y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        dist = pd.Series(split_y).value_counts(normalize=True).sort_index() * 100
        print(f"   {split_name}: " + " | ".join([f"{int(k)}:{v:.1f}%" for k, v in dist.items()]))
    
    # Preprocessing
    print("🔧 Applying enhanced preprocessing...")
    preprocessor = AdvancedPreprocessor(random_state=RANDOM_SEED)
    X_train_processed = preprocessor.advanced_preprocessing(X_train, y_train, fit=True)
    X_val_processed = preprocessor.advanced_preprocessing(X_val, fit=False)
    X_test_processed = preprocessor.advanced_preprocessing(X_test, fit=False)
    
    print(f"📊 Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
    
    # Create and train models
    models = create_actionable_models(random_state=RANDOM_SEED)
    
    print("🎓 Training models with validation...")
    trained_models = {}
    validation_results = {}
    
    for name, model in models.items():
        print(f"  🔄 Training {name}...")
        try:
            # Train model
            model.fit(X_train_processed, y_train)
            trained_models[name] = model
            
            # Validate on validation set
            y_val_pred = model.predict(X_val_processed)
            val_metrics = evaluate_model_actionable(y_val, y_val_pred, name)
            validation_results[name] = val_metrics
            
            print(f"    ✅ {name} - Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1_macro']:.4f}, "
                  f"Actionable: {val_metrics['actionability_score']:.2f}")
            
        except Exception as e:
            print(f"    ⚠️ Failed to train {name}: {e}")
    
    # Time series cross-validation for top models
    print("🔄 Performing time series cross-validation...")
    cv_results = {}
    
    # Select top 5 models based on validation performance
    val_df = pd.DataFrame(validation_results).T
    if not val_df.empty:
        val_df['val_combined_score'] = (val_df['accuracy'] * 0.6) + (val_df['actionability_score'] * 0.4)
        top_models = val_df.nlargest(5, 'val_combined_score').index.tolist()
        
        for name in top_models:
            if name in trained_models:
                print(f"  🔄 CV for {name}...")
                try:
                    cv_result = time_series_cross_validation(
                        X_train_processed, y_train, trained_models[name], n_splits=5
                    )
                    cv_results[name] = cv_result
                    print(f"    ✅ {name} - CV Acc: {cv_result['accuracy_mean']:.4f}±{cv_result['accuracy_std']:.4f}")
                except Exception as e:
                    print(f"    ⚠️ CV failed for {name}: {e}")
    
    # Final evaluation on test set
    print("📊 Final evaluation on test set...")
    test_results = {}
    best_model_name = None
    best_model_obj = None
    best_y_pred = None
    
    for name, model in trained_models.items():
        print(f"  📈 Testing {name}...")
        try:
            y_test_pred = model.predict(X_test_processed)
            test_metrics = evaluate_model_actionable(y_test, y_test_pred, name)
            test_results[name] = test_metrics
            
            # Store best model for visualization
            if best_model_name is None or (
                'accuracy' in test_metrics and 'actionability_score' in test_metrics and
                (test_metrics['accuracy'] * 0.6 + test_metrics['actionability_score'] * 0.4) >
                (test_results[best_model_name]['accuracy'] * 0.6 + test_results[best_model_name]['actionability_score'] * 0.4)
            ):
                best_model_name = name
                best_model_obj = model
                best_y_pred = y_test_pred
            
            print(f"    ✅ {name}: Acc={test_metrics['accuracy']:.4f}, "
                  f"F1={test_metrics['f1_macro']:.4f}, "
                  f"Actionable={test_metrics['actionability_score']:.2f} "
                  f"(Buy={test_metrics['pred_buy_pct']:.1f}%, Sell={test_metrics['pred_sell_pct']:.1f}%)")
            
        except Exception as e:
            print(f"    ⚠️ Failed to evaluate {name}: {e}")
    
    # Combine all results
    print("📊 Combining results...")
    final_results = {}
    
    for name in trained_models.keys():
        final_results[name] = {}
        
        # Add validation results
        if name in validation_results:
            for key, value in validation_results[name].items():
                final_results[name][f'val_{key}'] = value
        
        # Add CV results
        if name in cv_results:
            for key, value in cv_results[name].items():
                final_results[name][f'cv_{key}'] = value
        
        # Add test results
        if name in test_results:
            for key, value in test_results[name].items():
                final_results[name][f'test_{key}'] = value
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(final_results).T
    
    # Calculate final combined scores
    if 'test_accuracy' in results_df.columns and 'test_actionability_score' in results_df.columns:
        results_df['final_combined_score'] = (results_df['test_accuracy'] * 0.6) + (results_df['test_actionability_score'] * 0.4)
        results_df = results_df.sort_values('final_combined_score', ascending=False)
    elif 'val_accuracy' in results_df.columns and 'val_actionability_score' in results_df.columns:
        results_df['final_combined_score'] = (results_df['val_accuracy'] * 0.6) + (results_df['val_actionability_score'] * 0.4)
        results_df = results_df.sort_values('final_combined_score', ascending=False)
    
    # Generate visualizations and detailed analysis for best model
    if best_model_obj is not None and best_y_pred is not None:
        print(f"\n📊 Generating detailed analysis for best model: {best_model_name}")
        print("=" * 60)
        
        try:
            # Generate confusion matrix
            cm = plot_confusion_matrix(y_test, best_y_pred,
                                      save_path=f'confusion_matrix_{best_model_name}.png')
            
            # Print detailed classification report
            classification_report_dict = print_detailed_classification_report(y_test, best_y_pred)
            
            # Generate feature importance plot (if model supports it)
            if hasattr(best_model_obj, 'feature_importances_'):
                # Create feature names (simplified for processed features)
                feature_names = [f'Feature_{i}' for i in range(X_test_processed.shape[1])]
                feature_importance_df = plot_feature_importance(
                    best_model_obj, feature_names, top_n=20,
                    save_path=f'feature_importance_{best_model_name}.png'
                )
            else:
                print("⚠️ Best model does not support feature importance extraction")
                feature_importance_df = None
            
            # Generate additional plots
            plot_model_comparison(results_df, save_path='model_comparison.png')
            plot_prediction_distribution(y_test, best_y_pred, save_path='prediction_distribution.png')
            generate_evaluation_summary(results_df, best_model_name, save_path='evaluation_summary.txt')
            
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {e}")
            cm = None
            classification_report_dict = None
            feature_importance_df = None
    else:
        print("⚠️ No valid model found for detailed analysis")
        cm = None
        classification_report_dict = None
        feature_importance_df = None
    
    print("✅ Enhanced pipeline completed!")
    
    return {
        'results': results_df,
        'models': trained_models,
        'validation_results': validation_results,
        'cv_results': cv_results,
        'test_results': test_results,
        'best_model_name': best_model_name,
        'best_model': best_model_obj,
        'confusion_matrix': cm,
        'classification_report': classification_report_dict,
        'feature_importance': feature_importance_df,
        'preprocessing_info': {
            'original_features': len(feature_cols),
            'processed_features': X_train_processed.shape[1],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }

def main():
    """
    Enhanced main function for IEEE publication-ready stock prediction system.
    
    This function demonstrates the complete workflow including:
    - Comprehensive data exploration and validation
    - Advanced feature engineering with 100+ indicators
    - Sophisticated target creation with risk adjustment
    - Robust preprocessing and feature selection
    - Advanced model training with ensemble methods
    - Time-series cross-validation
    - Comprehensive evaluation and visualization
    - Automated experiment logging and artifact saving
    """
    print("🚀 ENHANCED ACTIONABLE STOCK PREDICTION SYSTEM")
    print("=" * 70)
    print("🎯 IEEE Publication-Ready Trading Signal Prediction Framework")
    print("📊 Features: Advanced ML, Risk-Adjusted Targets, Comprehensive Validation")
    print("🔬 Includes: Data Exploration, Unit Tests, Experiment Logging")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    set_random_seeds(RANDOM_SEED)
    
    # Initialize logger
    global logger
    logger = setup_logging("INFO", f"trading_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Run unit tests first
    print("\n🧪 RUNNING UNIT TESTS")
    print("-" * 50)
    test_suite = TradingPredictorTests()
    test_results = test_suite.run_all_tests()
    
    if not all(test_results.values()):
        print("⚠️ Some unit tests failed. Proceeding with caution...")
    
    # Initialize experiment logging
    experiment_name = f"enhanced_trading_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_logger = ExperimentLogger(experiment_name)
    
    # Log experiment parameters
    experiment_logger.log_parameters({
        'unit_tests_passed': all(test_results.values()),
        'failed_tests': [k for k, v in test_results.items() if not v],
        'system_info': {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE
        }
    })
    
    # Load and validate data
    print("\n📂 LOADING AND VALIDATING DATA")
    print("-" * 50)
    
    data_file = "us_stocks_5years_with_fundamentals.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        print("📝 Creating synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        test_suite = TradingPredictorTests()
        df = test_suite.create_sample_data(n_samples=5000, n_tickers=10)
        print(f"✅ Created synthetic data. Shape: {df.shape}")
        
        # Save synthetic data
        df.to_csv("synthetic_stock_data.csv", index=False)
        data_file = "synthetic_stock_data.csv"
    else:
        print(f"📂 Loading data from {data_file}...")
        try:
            df = pd.read_csv(data_file)
            print(f"✅ Data loaded successfully. Shape: {df.shape}")
            
            # Sample data if too large
            if len(df) > 100000:
                print("📊 Sampling data for processing...")
                tickers = df['Ticker'].unique()
                sampled_tickers = np.random.choice(tickers, size=min(10, len(tickers)), replace=False)
                df = df[df['Ticker'].isin(sampled_tickers)].copy()
                print(f"📊 Sampled data shape: {df.shape}")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    
    # Validate input data
    try:
        required_columns = ['Date', 'Ticker', 'Close', 'Volume']
        validate_input_data(df, required_columns)
        print("✅ Data validation passed")
    except ValueError as e:
        print(f"❌ Data validation failed: {e}")
        return
    
    # Feature engineering
    print("\n🔧 COMPREHENSIVE FEATURE ENGINEERING")
    print("-" * 50)
    df_features = comprehensive_feature_engineering(df)
    
    # Create actionable targets
    print("\n🎯 ACTIONABLE TARGET CREATION")
    print("-" * 50)
    df_with_targets = create_actionable_targets(
        df_features,
        target_buy_pct=20,
        target_sell_pct=20
    )
    
    # Train and evaluate
    print("\n🎓 ENHANCED TRAINING & EVALUATION")
    print("-" * 50)
    pipeline_results = train_and_evaluate_actionable(df_with_targets)
    
    # Save all results and artifacts
    print("\n💾 SAVING EXPERIMENT ARTIFACTS")
    print("-" * 50)
    
    # Save results
    experiment_logger.save_artifact(
        pipeline_results['results'], 'model_results.csv', 'csv'
    )
    
    # Save best model
    if 'best_model' in pipeline_results and pipeline_results['best_model']:
        experiment_logger.save_artifact(
            pipeline_results['best_model'], 'best_model.pkl', 'pickle'
        )
    
    # Save preprocessing artifacts
    experiment_logger.save_artifact(
        pipeline_results['preprocessing_info'], 'preprocessing_info.json', 'json'
    )
    
    # Log final metrics
    if not pipeline_results['results'].empty:
        best_metrics = pipeline_results['results'].iloc[0].to_dict()
        experiment_logger.log_metrics(best_metrics)
    
    # Finalize experiment
    metadata_path = experiment_logger.finalize_experiment()
    print(f"📋 Experiment metadata saved: {metadata_path}")
    
    # Display results summary
    print("\n📊 ENHANCED RESULTS SUMMARY")
    print("-" * 50)
    results_df = pipeline_results['results']
    
    if not results_df.empty:
        # Show key metrics
        key_metrics = []
        for prefix in ['test_', 'val_', '']:
            for metric in ['accuracy', 'balanced_accuracy', 'f1_macro', 'actionability_score']:
                col_name = f"{prefix}{metric}" if prefix else metric
                if col_name in results_df.columns:
                    key_metrics.append(col_name)
        
        if 'final_combined_score' in results_df.columns:
            key_metrics.append('final_combined_score')
        
        available_metrics = [m for m in key_metrics if m in results_df.columns]
        if available_metrics:
            print("🏆 Model Performance Summary:")
            print(results_df[available_metrics].round(4))
        
        # Best model summary
        best_model = results_df.iloc[0]
        print(f"\n🏆 Best Model: {results_df.index[0]}")
        
        # Find best available metrics
        for metric_type in ['test_', 'val_', '']:
            accuracy_key = f"{metric_type}accuracy"
            if accuracy_key in best_model and pd.notna(best_model[accuracy_key]):
                print(f"📊 Accuracy ({accuracy_key}): {best_model[accuracy_key]:.4f} ({best_model[accuracy_key]*100:.2f}%)")
                break
        
        for metric_type in ['test_', 'val_', '']:
            f1_key = f"{metric_type}f1_macro"
            if f1_key in best_model and pd.notna(best_model[f1_key]):
                print(f"📊 F1-Score ({f1_key}): {best_model[f1_key]:.4f}")
                break
        
        for metric_type in ['test_', 'val_', '']:
            actionability_key = f"{metric_type}actionability_score"
            if actionability_key in best_model and pd.notna(best_model[actionability_key]):
                print(f"📊 Actionability ({actionability_key}): {best_model[actionability_key]:.4f}")
                break
    
    print(f"\n🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
    print("📊 All artifacts saved to experiment directory")
    print("🔬 Ready for IEEE publication analysis")
    
    return pipeline_results

if __name__ == "__main__":
    results = main()