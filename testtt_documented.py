"""
Actionable Trading Predictor System

This module implements a comprehensive machine learning system for stock trading predictions.
The system is designed to provide actionable signals (BUY/SELL/HOLD) while maintaining
high accuracy and practical trading frequency.

Key Features:
- Advanced feature engineering with 100+ technical indicators  
- Multi-horizon risk-adjusted target creation
- Sophisticated preprocessing with adaptive scaling
- Ensemble learning with multiple ML algorithms
- Time-series cross-validation
- Comprehensive evaluation metrics

Author: Trading AI System
Version: 2.0 - Enhanced Documentation
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

class ActionableTradingPredictor:
    """
    Actionable Trading Predictor that forces 20% BUY and 20% SELL signals while maintaining accuracy
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        
    def comprehensive_feature_engineering(self, df):
        """Enhanced feature engineering with advanced technical indicators and statistical features"""
        print("🔧 Starting comprehensive feature engineering...")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        features_df = df.copy()
        
        print("📊 Creating basic features...")
        # Basic return features with more periods
        for period in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
            features_df[f'Return_{period}D'] = features_df.groupby('Ticker')['Close'].pct_change(periods=period)
            features_df[f'Volume_Change_{period}D'] = features_df.groupby('Ticker')['Volume'].pct_change(periods=period)
            
            # Log returns for better distribution
            features_df[f'Log_Return_{period}D'] = features_df.groupby('Ticker')['Close'].transform(
                lambda x: np.log(x / x.shift(period))
            )
        
        print("📈 Creating advanced technical indicators...")
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
        
        # Price ratios and momentum indicators
        for period in [5, 10, 20, 50]:
            features_df[f'Price_SMA_Ratio_{period}'] = features_df['Close'] / (features_df[f'SMA_{period}'] + 1e-10)
            features_df[f'Price_EMA_Ratio_{period}'] = features_df['Close'] / (features_df[f'EMA_{period}'] + 1e-10)
            
            # Momentum oscillator
            features_df[f'Momentum_{period}'] = features_df.groupby('Ticker')['Close'].transform(
                lambda x: (x / x.shift(period) - 1) * 100
            )
        
        # Multiple RSI periods
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))
        
        for rsi_period in [9, 14, 21]:
            features_df[f'RSI_{rsi_period}'] = features_df.groupby('Ticker')['Close'].transform(
                lambda x: calculate_rsi(x, rsi_period)
            )
        
        # Stochastic Oscillator
        def calculate_stochastic(high, low, close, k_period=14, d_period=3):
            lowest_low = low.rolling(k_period, min_periods=1).min()
            highest_high = high.rolling(k_period, min_periods=1).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
            d_percent = k_percent.rolling(d_period, min_periods=1).mean()
            return k_percent, d_percent
        
        if 'High' in features_df.columns and 'Low' in features_df.columns:
            stoch_results = features_df.groupby('Ticker').apply(
                lambda group: calculate_stochastic(group['High'], group['Low'], group['Close'])
            )
            k_values, d_values = [], []
            for ticker in features_df['Ticker'].unique():
                ticker_data = stoch_results[ticker]
                k_values.extend(ticker_data[0].values)
                d_values.extend(ticker_data[1].values)
            features_df['Stoch_K'] = k_values
            features_df['Stoch_D'] = d_values
        
        # Enhanced MACD with multiple timeframes
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
        
        # Standard MACD
        macd_results = features_df.groupby('Ticker')['Close'].apply(lambda x: calculate_macd(x, 12, 26, 9))
        macd_values, signal_values, hist_values = [], [], []
        
        for ticker in features_df['Ticker'].unique():
            ticker_data = macd_results[ticker]
            macd_values.extend(ticker_data[0].values)
            signal_values.extend(ticker_data[1].values)
            hist_values.extend(ticker_data[2].values)
        
        features_df['MACD'] = macd_values
        features_df['MACD_Signal'] = signal_values
        features_df['MACD_Hist'] = hist_values
        
        # Fast MACD
        macd_fast_results = features_df.groupby('Ticker')['Close'].apply(lambda x: calculate_macd(x, 5, 13, 5))
        macd_fast_values, signal_fast_values, hist_fast_values = [], [], []
        
        for ticker in features_df['Ticker'].unique():
            ticker_data = macd_fast_results[ticker]
            macd_fast_values.extend(ticker_data[0].values)
            signal_fast_values.extend(ticker_data[1].values)
            hist_fast_values.extend(ticker_data[2].values)
        
        features_df['MACD_Fast'] = macd_fast_values
        features_df['MACD_Signal_Fast'] = signal_fast_values
        features_df['MACD_Hist_Fast'] = hist_fast_values
        
        # Enhanced Bollinger Bands
        def calculate_bb(prices, period=20, std_dev=2):
            sma = prices.rolling(period, min_periods=1).mean()
            std = prices.rolling(period, min_periods=1).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        
        # Multiple Bollinger Band periods
        for bb_period in [10, 20, 50]:
            bb_results = features_df.groupby('Ticker')['Close'].apply(lambda x: calculate_bb(x, bb_period, 2))
            bb_upper_values, bb_middle_values, bb_lower_values = [], [], []
            
            for ticker in features_df['Ticker'].unique():
                ticker_data = bb_results[ticker]
                bb_upper_values.extend(ticker_data[0].values)
                bb_middle_values.extend(ticker_data[1].values)
                bb_lower_values.extend(ticker_data[2].values)
            
            features_df[f'BB_Upper_{bb_period}'] = bb_upper_values
            features_df[f'BB_Middle_{bb_period}'] = bb_middle_values
            features_df[f'BB_Lower_{bb_period}'] = bb_lower_values
            features_df[f'BB_Position_{bb_period}'] = (features_df['Close'] - features_df[f'BB_Lower_{bb_period}']) / (features_df[f'BB_Upper_{bb_period}'] - features_df[f'BB_Lower_{bb_period}'] + 1e-10)
            features_df[f'BB_Width_{bb_period}'] = (features_df[f'BB_Upper_{bb_period}'] - features_df[f'BB_Lower_{bb_period}']) / features_df[f'BB_Middle_{bb_period}']
        
        # Williams %R
        def calculate_williams_r(high, low, close, period=14):
            highest_high = high.rolling(period, min_periods=1).max()
            lowest_low = low.rolling(period, min_periods=1).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
            return williams_r
        
        if 'High' in features_df.columns and 'Low' in features_df.columns:
            features_df['Williams_R'] = features_df.groupby('Ticker').apply(
                lambda group: calculate_williams_r(group['High'], group['Low'], group['Close'])
            ).reset_index(level=0, drop=True)
        
        # Commodity Channel Index (CCI)
        def calculate_cci(high, low, close, period=20):
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(period, min_periods=1).mean()
            mad = tp.rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
            return cci
        
        if 'High' in features_df.columns and 'Low' in features_df.columns:
            features_df['CCI'] = features_df.groupby('Ticker').apply(
                lambda group: calculate_cci(group['High'], group['Low'], group['Close'])
            ).reset_index(level=0, drop=True)
        
        print("📊 Creating statistical features...")
        # Statistical features for returns
        for window in [5, 10, 20]:
            features_df[f'Return_Skew_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
                lambda x: x.rolling(window, min_periods=3).apply(lambda y: skew(y) if len(y) >= 3 else 0)
            )
            features_df[f'Return_Kurt_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
                lambda x: x.rolling(window, min_periods=3).apply(lambda y: kurtosis(y) if len(y) >= 3 else 0)
            )
            
            # Rolling correlation with market (will be calculated after market features are created)
            pass
        
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
        
        # Price and volume interaction features
        features_df['Price_Volume_Trend'] = features_df['Return_1D'] * features_df['Volume_Change_1D']
        features_df['Volume_Price_Trend'] = features_df['Volume_Change_1D'] * features_df['Return_1D']
        
        # Trend strength indicators
        features_df['SMA_5_20_Ratio'] = features_df['SMA_5'] / (features_df['SMA_20'] + 1e-10)
        features_df['SMA_10_50_Ratio'] = features_df['SMA_10'] / (features_df['SMA_50'] + 1e-10)
        features_df['EMA_5_20_Ratio'] = features_df['EMA_5'] / (features_df['EMA_20'] + 1e-10)
        
        # Market features
        print("📊 Creating market-relative features...")
        features_df['Market_Return_1D'] = features_df.groupby('Date')['Return_1D'].transform('mean')
        features_df['Market_Volume_1D'] = features_df.groupby('Date')['Volume_Change_1D'].transform('mean')
        features_df['Relative_Return_1D'] = features_df['Return_1D'] - features_df['Market_Return_1D']
        features_df['Relative_Volume_1D'] = features_df['Volume_Change_1D'] - features_df['Market_Volume_1D']
        
        # Now calculate market correlation features
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
        
        # Volatility features
        print("📊 Creating volatility features...")
        for window in [5, 10, 20]:
            # Realized volatility
            features_df[f'Realized_Vol_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
                lambda x: x.rolling(window, min_periods=1).std() * np.sqrt(252)
            )
            
            # GARCH-like volatility
            features_df[f'Vol_of_Vol_{window}'] = features_df.groupby('Ticker')[f'Realized_Vol_{window}'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Clean data
        print("🧹 Cleaning and finalizing features...")
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
        
        print(f"✅ Enhanced feature engineering completed. Shape: {features_df.shape}")
        return features_df
    
    def create_actionable_targets(self, df, target_buy_pct=20, target_sell_pct=20):
        """Create targets with enhanced scoring system and risk-adjusted returns"""
        print(f"🎯 Creating actionable targets with {target_buy_pct}% BUY and {target_sell_pct}% SELL...")
        
        # Calculate multiple future return horizons
        for horizon in [1, 2, 3, 5]:
            df[f'Future_Return_{horizon}D'] = df.groupby('Ticker')['Close'].pct_change(periods=horizon).shift(-horizon)
        
        # Calculate risk-adjusted returns (Sharpe-like ratio)
        for horizon in [1, 2, 3, 5]:
            df[f'Risk_Adj_Return_{horizon}D'] = df.groupby('Ticker').apply(
                lambda group: group[f'Future_Return_{horizon}D'] / (group[f'Volatility_20'] + 1e-10)
            ).reset_index(level=0, drop=True)
        
        # Enhanced comprehensive scoring system
        def calculate_action_score(row):
            score = 0
            
            # Multi-horizon future returns (35% weight)
            future_weights = {1: 0.4, 2: 0.3, 3: 0.2, 5: 0.1}
            for horizon, weight in future_weights.items():
                future_return = row.get(f'Future_Return_{horizon}D', 0)
                if pd.notna(future_return):
                    score += future_return * 35 * weight
            
            # Risk-adjusted returns (15% weight)
            risk_adj_return = row.get('Risk_Adj_Return_1D', 0)
            if pd.notna(risk_adj_return):
                score += np.clip(risk_adj_return, -0.1, 0.1) * 15
            
            # Multiple RSI signals (15% weight)
            rsi_signals = 0
            rsi_count = 0
            for period in [9, 14, 21]:
                rsi_val = row.get(f'RSI_{period}', 50)
                if pd.notna(rsi_val):
                    if rsi_val < 30:
                        rsi_signals += 0.15
                    elif rsi_val < 40:
                        rsi_signals += 0.05
                    elif rsi_val > 70:
                        rsi_signals -= 0.15
                    elif rsi_val > 60:
                        rsi_signals -= 0.05
                    rsi_count += 1
            if rsi_count > 0:
                score += (rsi_signals / rsi_count) * 15
            
            # Enhanced MACD signals (10% weight)
            macd_score = 0
            macd = row.get('MACD', 0)
            macd_signal = row.get('MACD_Signal', 0)
            macd_fast = row.get('MACD_Fast', 0)
            macd_signal_fast = row.get('MACD_Signal_Fast', 0)
            
            if pd.notna(macd) and pd.notna(macd_signal):
                macd_diff = macd - macd_signal
                macd_score += np.clip(macd_diff * 50, -0.1, 0.1)
            
            if pd.notna(macd_fast) and pd.notna(macd_signal_fast):
                macd_fast_diff = macd_fast - macd_signal_fast
                macd_score += np.clip(macd_fast_diff * 30, -0.05, 0.05)
            
            score += macd_score * 10
            
            # Multiple Bollinger Band signals (8% weight)
            bb_score = 0
            bb_count = 0
            for period in [10, 20, 50]:
                bb_position = row.get(f'BB_Position_{period}', 0.5)
                if pd.notna(bb_position):
                    if bb_position < 0.1:
                        bb_score += 0.2
                    elif bb_position < 0.3:
                        bb_score += 0.1
                    elif bb_position > 0.9:
                        bb_score -= 0.2
                    elif bb_position > 0.7:
                        bb_score -= 0.1
                    bb_count += 1
            if bb_count > 0:
                score += (bb_score / bb_count) * 8
            
            # Moving average trend signals (7% weight)
            ma_score = 0
            sma_5_20 = row.get('SMA_5_20_Ratio', 1)
            sma_10_50 = row.get('SMA_10_50_Ratio', 1)
            ema_5_20 = row.get('EMA_5_20_Ratio', 1)
            
            if pd.notna(sma_5_20):
                if sma_5_20 > 1.02:
                    ma_score += 0.1
                elif sma_5_20 < 0.98:
                    ma_score -= 0.1
            
            if pd.notna(sma_10_50):
                if sma_10_50 > 1.01:
                    ma_score += 0.05
                elif sma_10_50 < 0.99:
                    ma_score -= 0.05
            
            if pd.notna(ema_5_20):
                if ema_5_20 > 1.015:
                    ma_score += 0.05
                elif ema_5_20 < 0.985:
                    ma_score -= 0.05
            
            score += ma_score * 7
            
            # Volume signals (5% weight)
            volume_score = 0
            for period in [5, 10, 20]:
                vol_ratio = row.get(f'Volume_Ratio_{period}', 1)
                if pd.notna(vol_ratio):
                    if vol_ratio > 1.5:
                        volume_score += 0.05
                    elif vol_ratio > 1.2:
                        volume_score += 0.02
            score += volume_score * 5
            
            # Momentum signals (5% weight)
            momentum_score = 0
            for period in [5, 10, 20]:
                momentum = row.get(f'Momentum_{period}', 0)
                if pd.notna(momentum):
                    momentum_score += np.clip(momentum / 100, -0.05, 0.05)
            score += momentum_score * 5
            
            # Market relative performance (5% weight)
            rel_return = row.get('Relative_Return_1D', 0)
            if pd.notna(rel_return):
                score += np.clip(rel_return * 100, -0.1, 0.1) * 5
            
            # Volatility consideration (3% weight) - prefer lower volatility for same returns
            volatility = row.get('Volatility_20', 0.02)
            if pd.notna(volatility) and volatility > 0:
                vol_penalty = min(volatility * 10, 0.1)  # Cap penalty
                score -= vol_penalty * 3
            
            # Statistical features (2% weight)
            skew_5 = row.get('Return_Skew_5', 0)
            if pd.notna(skew_5):
                # Positive skew is good for long positions
                score += np.clip(skew_5 / 2, -0.05, 0.05) * 2
            
            return score
        
        print("📊 Calculating enhanced action scores...")
        df['Action_Score'] = df.apply(calculate_action_score, axis=1)
        
        # Dynamic threshold adjustment based on market conditions
        print("🎯 Creating adaptive targets...")
        
        # Calculate market regime (trending vs sideways)
        market_vol = df.groupby('Date')['Return_1D'].std().rolling(20, min_periods=5).mean()
        high_vol_threshold = market_vol.quantile(0.7)
        
        # Adjust target percentages based on market volatility
        df['Market_Vol_Regime'] = df['Date'].map(
            lambda x: market_vol.loc[market_vol.index <= x].iloc[-1] if len(market_vol.loc[market_vol.index <= x]) > 0 else market_vol.mean()
        )
        
        # In high volatility periods, be more selective (reduce action percentages)
        df['Adjusted_Buy_Pct'] = target_buy_pct
        df['Adjusted_Sell_Pct'] = target_sell_pct
        
        high_vol_mask = df['Market_Vol_Regime'] > high_vol_threshold
        df.loc[high_vol_mask, 'Adjusted_Buy_Pct'] = target_buy_pct * 0.8
        df.loc[high_vol_mask, 'Adjusted_Sell_Pct'] = target_sell_pct * 0.8
        
        # Create targets with adaptive thresholds
        valid_scores = df.dropna(subset=['Action_Score'])
        
        # Use time-based rolling thresholds for better adaptation
        df['Target'] = 1  # Default HOLD
        
        # Calculate rolling percentile thresholds
        window_size = min(1000, len(valid_scores) // 5)  # Adaptive window size
        
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['Action_Score']):
                continue
                
            # Get recent data for threshold calculation
            start_idx = max(0, i - window_size)
            recent_scores = df.iloc[start_idx:i+1]['Action_Score'].dropna()
            
            if len(recent_scores) < 50:  # Need minimum data
                continue
            
            buy_pct = df.iloc[i]['Adjusted_Buy_Pct']
            sell_pct = df.iloc[i]['Adjusted_Sell_Pct']
            
            buy_threshold = recent_scores.quantile(1 - buy_pct / 100)
            sell_threshold = recent_scores.quantile(sell_pct / 100)
            
            current_score = df.iloc[i]['Action_Score']
            
            if current_score >= buy_threshold:
                df.iloc[i, df.columns.get_loc('Target')] = 2  # BUY
            elif current_score <= sell_threshold:
                df.iloc[i, df.columns.get_loc('Target')] = 0  # SELL
        
        # Final distribution
        final_counts = df['Target'].value_counts().sort_index()
        final_total = df['Target'].count()
        print(f"📊 Final target distribution:")
        for target, count in final_counts.items():
            if pd.notna(target):
                target_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}.get(int(target), f'Class_{int(target)}')
                pct = count / final_total * 100
                print(f"   {target_name}: {count} ({pct:.1f}%)")
        
        return df
    
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
    
    def create_actionable_models(self):
        """Create advanced models optimized for actionable predictions"""
        print("🤖 Creating advanced actionable models...")
        
        models = {}
        
        # Enhanced Random Forest with better hyperparameters
        models['rf_enhanced'] = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=3,
            min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample',
            random_state=self.random_state, n_jobs=-1, bootstrap=True,
            oob_score=True, max_samples=0.8
        )
        
        # Enhanced Extra Trees with diversity focus
        models['et_enhanced'] = ExtraTreesClassifier(
            n_estimators=300, max_depth=25, min_samples_split=2,
            min_samples_leaf=1, max_features='log2', class_weight='balanced_subsample',
            random_state=self.random_state, n_jobs=-1, bootstrap=True,
            oob_score=True, max_samples=0.9
        )
        
        # Enhanced Gradient Boosting
        models['gb_enhanced'] = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=6,
            min_samples_split=4, min_samples_leaf=2, subsample=0.85,
            max_features='sqrt', random_state=self.random_state,
            validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
        )
        
        # Enhanced Logistic Regression with regularization
        models['lr_enhanced'] = LogisticRegression(
            C=0.1, penalty='elasticnet', l1_ratio=0.5, class_weight='balanced',
            random_state=self.random_state, max_iter=3000, solver='saga'
        )
        
        # Enhanced MLP with better architecture
        models['mlp_enhanced'] = MLPClassifier(
            hidden_layer_sizes=(150, 100, 50), activation='relu', solver='adam',
            alpha=0.001, learning_rate='adaptive', max_iter=500,
            random_state=self.random_state, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15, tol=1e-4
        )
        
        # Ridge Classifier for linear baseline
        models['ridge'] = RidgeClassifier(
            alpha=1.0, class_weight='balanced', random_state=self.random_state
        )
        
        # AdaBoost for boosting diversity
        models['ada_boost'] = AdaBoostClassifier(
            n_estimators=100, learning_rate=0.8, algorithm='SAMME.R',
            random_state=self.random_state
        )
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            print("   🚀 Adding XGBoost models...")
            
            # Standard XGBoost
            models['xgb_standard'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=self.random_state,
                n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False
            )
            
            # XGBoost with different hyperparameters for diversity
            models['xgb_deep'] = xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.7, colsample_bylevel=0.9,
                reg_alpha=0.05, reg_lambda=0.5, random_state=self.random_state + 1,
                n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False
            )
            
            # XGBoost optimized for imbalanced classes
            models['xgb_balanced'] = xgb.XGBClassifier(
                n_estimators=250, max_depth=7, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.9, colsample_bylevel=0.8,
                reg_alpha=0.2, reg_lambda=1.5, scale_pos_weight=1.5,
                random_state=self.random_state + 2, n_jobs=-1,
                eval_metric='mlogloss', use_label_encoder=False
            )
        
        # LightGBM if available
        if LIGHTGBM_AVAILABLE:
            print("   ⚡ Adding LightGBM models...")
            
            # Standard LightGBM
            models['lgb_standard'] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=self.random_state, n_jobs=-1,
                objective='multiclass', metric='multi_logloss', verbose=-1
            )
            
            # LightGBM with leaf-wise growth
            models['lgb_leafwise'] = lgb.LGBMClassifier(
                n_estimators=300, num_leaves=50, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.7, reg_alpha=0.05,
                reg_lambda=0.5, random_state=self.random_state + 3, n_jobs=-1,
                objective='multiclass', metric='multi_logloss', verbose=-1,
                boosting_type='gbdt', min_child_samples=10
            )
            
            # LightGBM optimized for feature importance
            models['lgb_dart'] = lgb.LGBMClassifier(
                n_estimators=250, max_depth=8, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.9, reg_alpha=0.2,
                reg_lambda=1.5, random_state=self.random_state + 4, n_jobs=-1,
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
    
    def evaluate_model_actionable(self, y_true, y_pred, model_name="Model"):
        """Comprehensive evaluation with actionability metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Prediction distribution
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        total_pred = len(y_pred)
        
        metrics['pred_sell_pct'] = (pred_counts.get(0, 0) / total_pred) * 100
        metrics['pred_hold_pct'] = (pred_counts.get(1, 0) / total_pred) * 100
        metrics['pred_buy_pct'] = (pred_counts.get(2, 0) / total_pred) * 100
        
        # Actionability score
        metrics['actionability_score'] = (metrics['pred_buy_pct'] + metrics['pred_sell_pct']) / 100
        
        # Class-specific metrics
        unique_classes = np.unique(y_true)
        for cls in unique_classes:
            cls_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}.get(cls, f'Class_{cls}')
            try:
                precision_cls = precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)
                recall_cls = recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)
                f1_cls = f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)
                
                metrics[f'precision_{cls_name}'] = precision_cls[0] if len(precision_cls) > 0 else 0
                metrics[f'recall_{cls_name}'] = recall_cls[0] if len(recall_cls) > 0 else 0
                metrics[f'f1_{cls_name}'] = f1_cls[0] if len(f1_cls) > 0 else 0
            except:
                metrics[f'precision_{cls_name}'] = 0
                metrics[f'recall_{cls_name}'] = 0
                metrics[f'f1_{cls_name}'] = 0
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Generate and save confusion matrix plot comparing true labels vs predicted labels
        with class names SELL, HOLD, BUY.
        """
        print("📊 Generating confusion matrix...")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Define class names
        class_names = ['SELL', 'HOLD', 'BUY']
        
        # Create figure and axis
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        # Add labels and title
        plt.title('Confusion Matrix - Stock Action Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        
        # Add accuracy information
        accuracy = accuracy_score(y_true, y_pred)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
                   fontsize=10, ha='left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved as '{save_path}'")
        
        return cm
    
    def print_detailed_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report showing precision, recall, and F1-score for each class.
        """
        print("\n📊 DETAILED CLASSIFICATION REPORT:")
        print("=" * 60)
        
        # Define class names
        class_names = ['SELL', 'HOLD', 'BUY']
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=class_names,
                                     digits=4, output_dict=True)
        
        # Print header
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        
        # Print metrics for each class
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                metrics = report[str(i)]
                print(f"{class_name:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                      f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")
        
        print("-" * 60)
        
        # Print overall metrics
        if 'macro avg' in report:
            macro = report['macro avg']
            print(f"{'Macro Avg':<10} {macro['precision']:<12.4f} {macro['recall']:<12.4f} "
                  f"{macro['f1-score']:<12.4f} {int(macro['support']):<10}")
        
        if 'weighted avg' in report:
            weighted = report['weighted avg']
            print(f"{'Weighted Avg':<10} {weighted['precision']:<12.4f} {weighted['recall']:<12.4f} "
                  f"{weighted['f1-score']:<12.4f} {int(weighted['support']):<10}")
        
        # Print accuracy
        if 'accuracy' in report:
            accuracy = report['accuracy']
            print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return report
    
    def plot_feature_importance(self, model, feature_names=None, top_n=20, save_path='feature_importance.png'):
        """
        Extract and plot feature importances from trained model.
        Assumes model has feature_importances_ attribute (tree-based models).
        """
        print("📊 Generating feature importance plot...")
        
        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_'):
            print("⚠️ Model does not have feature_importances_ attribute")
            return None
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame for easier handling
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = feature_df.head(top_n)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['importance'],
                       color='steelblue', alpha=0.7)
        
        # Customize plot
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {type(model).__name__}',
                 fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        # Invert y-axis to show highest importance at top
        plt.gca().invert_yaxis()
        
        # Add grid for better readability
        plt.grid(axis='x', alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Feature importance plot saved as '{save_path}'")
        
        # Print top features
        print(f"\n📊 Top {min(10, len(top_features))} Most Important Features:")
        print("-" * 50)
        for i, (_, row) in enumerate(top_features.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.6f}")
        
        return feature_df
    
    def time_series_cross_validation(self, X, y, model, n_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit model on fold
            model_fold = type(model)(**model.get_params())
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Predict and evaluate
            y_pred_fold = model_fold.predict(X_val_fold)
            fold_metrics = self.evaluate_model_actionable(y_val_fold, y_pred_fold)
            cv_scores.append(fold_metrics)
        
        # Calculate mean and std of CV scores
        cv_results = {}
        for metric in cv_scores[0].keys():
            scores = [fold[metric] for fold in cv_scores]
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        return cv_results
    
    def train_and_evaluate_actionable(self, df, test_size=0.2, validation_size=0.1):
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
        X_train_processed = self.advanced_preprocessing(X_train, y_train, fit=True)
        X_val_processed = self.advanced_preprocessing(X_val, fit=False)
        X_test_processed = self.advanced_preprocessing(X_test, fit=False)
        
        print(f"📊 Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
        
        # Create and train models
        models = self.create_actionable_models()
        
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
                val_metrics = self.evaluate_model_actionable(y_val, y_val_pred, name)
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
                        cv_result = self.time_series_cross_validation(
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
                test_metrics = self.evaluate_model_actionable(y_test, y_test_pred, name)
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
                cm = self.plot_confusion_matrix(y_test, best_y_pred,
                                              save_path=f'confusion_matrix_{best_model_name}.png')
                
                # Print detailed classification report
                classification_report_dict = self.print_detailed_classification_report(y_test, best_y_pred)
                
                # Generate feature importance plot (if model supports it)
                if hasattr(best_model_obj, 'feature_importances_'):
                    # Create feature names (simplified for processed features)
                    feature_names = [f'Feature_{i}' for i in range(X_test_processed.shape[1])]
                    feature_importance_df = self.plot_feature_importance(
                        best_model_obj, feature_names, top_n=20,
                        save_path=f'feature_importance_{best_model_name}.png'
                    )
                else:
                    print("⚠️ Best model does not support feature importance extraction")
                    feature_importance_df = None
                
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
    """Main function for actionable stock prediction"""
    print("🚀 ACTIONABLE STOCK PREDICTION SYSTEM")
    print("=" * 60)
    print("🎯 Designed to provide 20% BUY and 20% SELL signals while maintaining accuracy")
    
    # Initialize predictor
    predictor = ActionableTradingPredictor(random_state=42)
    
    # Load data
    data_file = "us_stocks_5years_with_fundamentals.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        return
    
    print(f"📂 Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # Sample data if too large
        if len(df) > 100000:
            print("📊 Sampling data for processing...")
            tickers = df['Ticker'].unique()
            sampled_tickers = np.random.choice(tickers, size=min(5, len(tickers)), replace=False)
            df = df[df['Ticker'].isin(sampled_tickers)].copy()
            print(f"📊 Sampled data shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Feature engineering
    print("\n🔧 COMPREHENSIVE FEATURE ENGINEERING")
    print("-" * 50)
    df_features = predictor.comprehensive_feature_engineering(df)
    
    # Create actionable targets
    print("\n🎯 ACTIONABLE TARGET CREATION")
    print("-" * 50)
    df_with_targets = predictor.create_actionable_targets(df_features, target_buy_pct=20, target_sell_pct=20)
    
    # Train and evaluate
    print("\n🎓 ACTIONABLE TRAINING & EVALUATION")
    print("-" * 50)
    pipeline_results = predictor.train_and_evaluate_actionable(df_with_targets)
    
    # Display results
    print("\n📊 ACTIONABLE RESULTS")
    print("-" * 50)
    results_df = pipeline_results['results']
    
    # Show all models
    print("🏆 All Models Performance:")
    if not results_df.empty:
        # Find available metrics dynamically
        test_metrics = [col for col in results_df.columns if col.startswith('test_')]
        val_metrics = [col for col in results_df.columns if col.startswith('val_')]
        cv_metrics = [col for col in results_df.columns if col.startswith('cv_')]
        
        # Show key metrics if available
        key_metrics = []
        for prefix in ['test_', 'val_', '']:
            for metric in ['accuracy', 'balanced_accuracy', 'f1_macro', 'actionability_score', 'pred_buy_pct', 'pred_sell_pct']:
                col_name = f"{prefix}{metric}" if prefix else metric
                if col_name in results_df.columns:
                    key_metrics.append(col_name)
        
        if 'final_combined_score' in results_df.columns:
            key_metrics.append('final_combined_score')
        
        available_metrics = [m for m in key_metrics if m in results_df.columns]
        if available_metrics:
            print(results_df[available_metrics].round(4))
        else:
            print(results_df.round(4))
    else:
        print("No results to display")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_results_{timestamp}.csv"
    results_df.to_csv(results_file)
    print(f"\n💾 Results saved to {results_file}")
    
    # Performance summary
    print("\n📈 ENHANCED PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    if not results_df.empty:
        best_model = results_df.iloc[0]
        print(f"🏆 Best Model: {results_df.index[0]}")
        
        # Find the best available accuracy metric
        accuracy_key = None
        for key in ['test_accuracy', 'val_accuracy', 'accuracy']:
            if key in best_model and pd.notna(best_model[key]):
                accuracy_key = key
                break
        
        if accuracy_key:
            print(f"📊 Accuracy ({accuracy_key}): {best_model[accuracy_key]:.4f} ({best_model[accuracy_key]*100:.2f}%)")
        
        # Find the best available F1 metric
        f1_key = None
        for key in ['test_f1_macro', 'val_f1_macro', 'f1_macro']:
            if key in best_model and pd.notna(best_model[key]):
                f1_key = key
                break
        
        if f1_key:
            print(f"📊 F1-Score ({f1_key}): {best_model[f1_key]:.4f}")
        
        # Find the best available actionability metric
        actionability_key = None
        for key in ['test_actionability_score', 'val_actionability_score', 'actionability_score']:
            if key in best_model and pd.notna(best_model[key]):
                actionability_key = key
                break
        
        if actionability_key:
            print(f"📊 Actionability Score ({actionability_key}): {best_model[actionability_key]:.4f}")
        
        if 'final_combined_score' in best_model:
            print(f"📊 Final Combined Score: {best_model['final_combined_score']:.4f}")
        
        # Prediction distribution
        buy_key = None
        sell_key = None
        hold_key = None
        
        for prefix in ['test_', 'val_', '']:
            if f"{prefix}pred_buy_pct" in best_model:
                buy_key = f"{prefix}pred_buy_pct"
                sell_key = f"{prefix}pred_sell_pct"
                hold_key = f"{prefix}pred_hold_pct"
                break
        
        if buy_key and sell_key and hold_key:
            print(f"\n📊 Prediction Distribution (Best Model):")
            print(f"   🔴 SELL: {best_model[sell_key]:.1f}%")
            print(f"   ⚪ HOLD: {best_model[hold_key]:.1f}%")
            print(f"   🟢 BUY: {best_model[buy_key]:.1f}%")
            
            # Success evaluation
            print(f"\n📈 ACTIONABILITY ANALYSIS:")
            print("-" * 50)
            total_actions = best_model[buy_key] + best_model[sell_key]
            
            if total_actions >= 35:
                print("🎉 EXCELLENT: High actionability with good accuracy!")
            elif total_actions >= 25:
                print("✅ GOOD: Reasonable actionability achieved!")
            elif total_actions >= 15:
                print("📈 MODERATE: Some actionability, but could be improved!")
            else:
                print("⚠️ LOW: Still too conservative, needs more tuning!")
        
        # Class-specific performance
        print(f"\n📊 Class-Specific Performance (Best Model):")
        for class_name in ['SELL', 'HOLD', 'BUY']:
            f1_found = False
            for prefix in ['test_', 'val_', '']:
                f1_col = f"{prefix}f1_{class_name}"
                precision_col = f"{prefix}precision_{class_name}"
                recall_col = f"{prefix}recall_{class_name}"
                
                if f1_col in best_model and pd.notna(best_model[f1_col]):
                    print(f"   {class_name} ({prefix}): F1={best_model[f1_col]:.4f}, "
                          f"Precision={best_model.get(precision_col, 0):.4f}, "
                          f"Recall={best_model.get(recall_col, 0):.4f}")
                    f1_found = True
                    break
            
            if not f1_found:
                print(f"   {class_name}: No metrics available")
    
    print(f"\n🔍 Key Enhancements Implemented:")
    print("   ✅ Advanced feature engineering with 100+ technical indicators")
    print("   ✅ Enhanced target creation with multi-horizon risk-adjusted returns")
    print("   ✅ Sophisticated preprocessing with adaptive scaling and feature selection")
    print("   ✅ Advanced models including XGBoost and LightGBM variants")
    print("   ✅ Robust time-series cross-validation")
    print("   ✅ Comprehensive ensemble methods")
    print("   ✅ Adaptive threshold logic based on market volatility")
    
    # Trading recommendations
    print(f"\n💡 TRADING RECOMMENDATIONS:")
    print("-" * 50)
    if not results_df.empty and accuracy_key and actionability_key:
        accuracy_val = best_model[accuracy_key]
        actionability_val = best_model[actionability_key]
        total_actions = best_model.get(buy_key, 0) + best_model.get(sell_key, 0) if buy_key and sell_key else 0
        
        if accuracy_val > 0.6 and total_actions > 30:
            print("🚀 READY FOR LIVE TRADING: High accuracy with good actionability!")
        elif accuracy_val > 0.5 and total_actions > 20:
            print("📊 PAPER TRADING READY: Test with virtual money first")
        else:
            print("🔧 NEEDS OPTIMIZATION: Continue improving before trading")
    else:
        print("📊 Model evaluation completed - review results for trading readiness")
    
    print("\n✅ ACTIONABLE PIPELINE COMPLETED!")
    print("🎯 This system balances accuracy with trading frequency for practical use.")
    
    return pipeline_results

if __name__ == "__main__":
    results = main()