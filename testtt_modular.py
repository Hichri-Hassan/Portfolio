"""
Actionable Trading Predictor - Modular Version

This module provides a comprehensive stock trading prediction system that generates
actionable BUY/SELL/HOLD signals while maintaining accuracy. The system is designed
to provide approximately 20% BUY and 20% SELL signals for practical trading use.

Key Features:
- Advanced feature engineering with 100+ technical indicators
- Multi-horizon risk-adjusted return calculations
- Sophisticated preprocessing with adaptive scaling
- Ensemble models including XGBoost and LightGBM
- Time-series cross-validation
- Comprehensive evaluation metrics

Author: Trading Predictor System
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import warnings
import os
from datetime import datetime
from scipy.stats import skew, kurtosis, zscore
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports for advanced models
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

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Handles comprehensive feature engineering for stock prediction.
    
    This class creates advanced technical indicators, statistical features,
    and market-relative features to improve prediction accuracy.
    
    Attributes:
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the FeatureEngineer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
    
    def create_basic_features(self, df):
        """
        Create basic return and volume features.
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with basic features added
        """
        print("📊 Creating basic features...")
        features_df = df.copy()
        
        # Basic return features with multiple periods
        for period in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
            features_df[f'Return_{period}D'] = features_df.groupby('Ticker')['Close'].pct_change(periods=period)
            features_df[f'Volume_Change_{period}D'] = features_df.groupby('Ticker')['Volume'].pct_change(periods=period)
            
            # Log returns for better distribution
            features_df[f'Log_Return_{period}D'] = features_df.groupby('Ticker')['Close'].transform(
                lambda x: np.log(x / x.shift(period))
            )
        
        return features_df
    
    def create_moving_averages(self, df):
        """
        Create various moving averages and volatility indicators.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with moving average features
        """
        print("📈 Creating moving averages...")
        features_df = df.copy()
        
        # Enhanced moving averages
        for window in [3, 5, 8, 10, 13, 20, 21, 34, 50, 89]:
            features_df[f'SMA_{window}'] = features_df.groupby('Ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features_df[f'EMA_{window}'] = features_df.groupby('Ticker')['Close'].transform(
                lambda x: x.ewm(span=window).mean()
            )
            features_df[f'Volatility_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            
            # Volume-weighted moving averages
            features_df[f'VWMA_{window}'] = features_df.groupby('Ticker').apply(
                lambda group: (group['Close'] * group['Volume']).rolling(window, min_periods=1).sum() /
                             group['Volume'].rolling(window, min_periods=1).sum()
            ).reset_index(level=0, drop=True)
        
        return features_df
    
    def create_momentum_indicators(self, df):
        """
        Create momentum and oscillator indicators.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with momentum indicators
        """
        print("📈 Creating momentum indicators...")
        features_df = df.copy()
        
        # Price ratios and momentum indicators
        for period in [5, 10, 20, 50]:
            features_df[f'Price_SMA_Ratio_{period}'] = features_df['Close'] / (features_df[f'SMA_{period}'] + 1e-10)
            features_df[f'Price_EMA_Ratio_{period}'] = features_df['Close'] / (features_df[f'EMA_{period}'] + 1e-10)
            
            # Momentum oscillator
            features_df[f'Momentum_{period}'] = features_df.groupby('Ticker')['Close'].transform(
                lambda x: (x / x.shift(period) - 1) * 100
            )
        
        # RSI calculation
        features_df = self._add_rsi_features(features_df)
        
        # MACD calculation
        features_df = self._add_macd_features(features_df)
        
        # Bollinger Bands
        features_df = self._add_bollinger_bands(features_df)
        
        return features_df
    
    def _add_rsi_features(self, df):
        """
        Add RSI (Relative Strength Index) features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with RSI features
        """
        def calculate_rsi(prices, period=14):
            """Calculate RSI for given prices and period."""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))
        
        for rsi_period in [9, 14, 21]:
            df[f'RSI_{rsi_period}'] = df.groupby('Ticker')['Close'].transform(
                lambda x: calculate_rsi(x, rsi_period)
            )
        
        return df
    
    def _add_macd_features(self, df):
        """
        Add MACD (Moving Average Convergence Divergence) features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with MACD features
        """
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            """Calculate MACD line, signal line, and histogram."""
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
        
        # Standard MACD
        macd_results = df.groupby('Ticker')['Close'].apply(lambda x: calculate_macd(x, 12, 26, 9))
        macd_values, signal_values, hist_values = [], [], []
        
        for ticker in df['Ticker'].unique():
            ticker_data = macd_results[ticker]
            macd_values.extend(ticker_data[0].values)
            signal_values.extend(ticker_data[1].values)
            hist_values.extend(ticker_data[2].values)
        
        df['MACD'] = macd_values
        df['MACD_Signal'] = signal_values
        df['MACD_Hist'] = hist_values
        
        # Fast MACD
        macd_fast_results = df.groupby('Ticker')['Close'].apply(lambda x: calculate_macd(x, 5, 13, 5))
        macd_fast_values, signal_fast_values, hist_fast_values = [], [], []
        
        for ticker in df['Ticker'].unique():
            ticker_data = macd_fast_results[ticker]
            macd_fast_values.extend(ticker_data[0].values)
            signal_fast_values.extend(ticker_data[1].values)
            hist_fast_values.extend(ticker_data[2].values)
        
        df['MACD_Fast'] = macd_fast_values
        df['MACD_Signal_Fast'] = signal_fast_values
        df['MACD_Hist_Fast'] = hist_fast_values
        
        return df
    
    def _add_bollinger_bands(self, df):
        """
        Add Bollinger Bands features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with Bollinger Bands features
        """
        def calculate_bb(prices, period=20, std_dev=2):
            """Calculate Bollinger Bands upper, middle, and lower bands."""
            sma = prices.rolling(period, min_periods=1).mean()
            std = prices.rolling(period, min_periods=1).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        
        # Multiple Bollinger Band periods
        for bb_period in [10, 20, 50]:
            bb_results = df.groupby('Ticker')['Close'].apply(lambda x: calculate_bb(x, bb_period, 2))
            bb_upper_values, bb_middle_values, bb_lower_values = [], [], []
            
            for ticker in df['Ticker'].unique():
                ticker_data = bb_results[ticker]
                bb_upper_values.extend(ticker_data[0].values)
                bb_middle_values.extend(ticker_data[1].values)
                bb_lower_values.extend(ticker_data[2].values)
            
            df[f'BB_Upper_{bb_period}'] = bb_upper_values
            df[f'BB_Middle_{bb_period}'] = bb_middle_values
            df[f'BB_Lower_{bb_period}'] = bb_lower_values
            df[f'BB_Position_{bb_period}'] = (df['Close'] - df[f'BB_Lower_{bb_period}']) / (df[f'BB_Upper_{bb_period}'] - df[f'BB_Lower_{bb_period}'] + 1e-10)
            df[f'BB_Width_{bb_period}'] = (df[f'BB_Upper_{bb_period}'] - df[f'BB_Lower_{bb_period}']) / df[f'BB_Middle_{bb_period}']
        
        return df
    
    def create_statistical_features(self, df):
        """
        Create statistical features for returns and volumes.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with statistical features
        """
        print("📊 Creating statistical features...")
        features_df = df.copy()
        
        # Statistical features for returns
        for window in [5, 10, 20]:
            features_df[f'Return_Skew_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
                lambda x: x.rolling(window, min_periods=3).apply(lambda y: skew(y) if len(y) >= 3 else 0)
            )
            features_df[f'Return_Kurt_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
                lambda x: x.rolling(window, min_periods=3).apply(lambda y: kurtosis(y) if len(y) >= 3 else 0)
            )
        
        # Volume features
        print("📊 Creating volume-based features...")
        for period in [5, 10, 20]:
            features_df[f'Volume_SMA_{period}'] = features_df.groupby('Ticker')['Volume'].transform(
                lambda x: x.rolling(period, min_periods=1).mean()
            )
            features_df[f'Volume_Ratio_{period}'] = features_df['Volume'] / (features_df[f'Volume_SMA_{period}'] + 1e-10)
            
            # On-Balance Volume
            features_df[f'OBV_{period}'] = features_df.groupby('Ticker').apply(
                lambda group: (group['Volume'] * np.sign(group['Return_1D'])).rolling(period, min_periods=1).sum()
            ).reset_index(level=0, drop=True)
        
        return features_df
    
    def create_market_features(self, df):
        """
        Create market-relative features and correlations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with market features
        """
        print("📊 Creating market-relative features...")
        features_df = df.copy()
        
        # Market features
        features_df['Market_Return_1D'] = features_df.groupby('Date')['Return_1D'].transform('mean')
        features_df['Market_Volume_1D'] = features_df.groupby('Date')['Volume_Change_1D'].transform('mean')
        features_df['Relative_Return_1D'] = features_df['Return_1D'] - features_df['Market_Return_1D']
        features_df['Relative_Volume_1D'] = features_df['Volume_Change_1D'] - features_df['Market_Volume_1D']
        
        # Market correlation features
        print("📊 Creating market correlation features...")
        for window in [5, 10, 20]:
            features_df[f'Market_Corr_{window}'] = features_df.groupby('Ticker').apply(
                lambda group: group['Return_1D'].rolling(window, min_periods=5).corr(group['Market_Return_1D'])
            ).reset_index(level=0, drop=True)
        
        # Beta calculation (rolling)
        for window in [20, 50]:
            features_df[f'Beta_{window}'] = features_df.groupby('Ticker').apply(
                lambda group: group['Return_1D'].rolling(window, min_periods=10).cov(group['Market_Return_1D']) /
                             (group['Market_Return_1D'].rolling(window, min_periods=10).var() + 1e-10)
            ).reset_index(level=0, drop=True)
        
        return features_df
    
    def create_volatility_features(self, df):
        """
        Create volatility-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with volatility features
        """
        print("📊 Creating volatility features...")
        features_df = df.copy()
        
        for window in [5, 10, 20]:
            # Realized volatility
            features_df[f'Realized_Vol_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
                lambda x: x.rolling(window, min_periods=1).std() * np.sqrt(252)
            )
            
            # GARCH-like volatility
            features_df[f'Vol_of_Vol_{window}'] = features_df.groupby('Ticker')[f'Realized_Vol_{window}'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        return features_df
    
    def clean_features(self, df):
        """
        Clean and finalize features by handling missing values and infinities.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        print("🧹 Cleaning and finalizing features...")
        features_df = df.copy()
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if features_df[col].isna().any():
                # Forward fill, then backward fill within each ticker
                features_df[col] = features_df.groupby('Ticker')[col].fillna(method='ffill').fillna(method='bfill')
                # If still NaN, use median
                if features_df[col].isna().any():
                    median_val = features_df[col].median()
                    features_df[col] = features_df[col].fillna(median_val)
        
        print(f"✅ Feature engineering completed. Shape: {features_df.shape}")
        return features_df
    
    def engineer_features(self, df):
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        print("🔧 Starting comprehensive feature engineering...")
        
        # Prepare data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Apply feature engineering steps
        df = self.create_basic_features(df)
        df = self.create_moving_averages(df)
        df = self.create_momentum_indicators(df)
        df = self.create_statistical_features(df)
        df = self.create_market_features(df)
        df = self.create_volatility_features(df)
        df = self.clean_features(df)
        
        return df