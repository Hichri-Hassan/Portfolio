"""
Technical Indicators Module

This module contains functions for calculating various technical indicators
used in stock market analysis including RSI, MACD, Bollinger Bands, etc.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI) for a given price series.
    
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100, where values above 70 indicate overbought conditions
    and values below 30 indicate oversold conditions.
    
    Args:
        prices (pd.Series): Series of closing prices
        period (int): Number of periods for RSI calculation (default: 14)
        
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    The Stochastic Oscillator compares a particular closing price of a security 
    to a range of its prices over a certain period of time.
    
    Args:
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of closing prices
        k_period (int): Number of periods for %K calculation (default: 14)
        d_period (int): Number of periods for %D smoothing (default: 3)
        
    Returns:
        tuple: (%K values, %D values)
    """
    lowest_low = low.rolling(k_period, min_periods=1).min()
    highest_high = high.rolling(k_period, min_periods=1).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d_percent = k_percent.rolling(d_period, min_periods=1).mean()
    return k_percent, d_percent


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    
    Args:
        prices (pd.Series): Series of closing prices
        fast (int): Fast EMA period (default: 12)
        slow (int): Slow EMA period (default: 26)
        signal (int): Signal line EMA period (default: 9)
        
    Returns:
        tuple: (MACD line, Signal line, MACD histogram)
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands
    that are standard deviations away from the middle band.
    
    Args:
        prices (pd.Series): Series of closing prices
        period (int): Period for SMA calculation (default: 20)
        std_dev (float): Number of standard deviations (default: 2)
        
    Returns:
        tuple: (Upper band, Middle band (SMA), Lower band)
    """
    sma = prices.rolling(period, min_periods=1).mean()
    std = prices.rolling(period, min_periods=1).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_williams_r(high, low, close, period=14):
    """
    Calculate Williams %R indicator.
    
    Williams %R is a momentum indicator that measures overbought and oversold levels.
    Values range from -100 to 0, where values above -20 are considered overbought
    and values below -80 are considered oversold.
    
    Args:
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of closing prices
        period (int): Number of periods for calculation (default: 14)
        
    Returns:
        pd.Series: Williams %R values
    """
    highest_high = high.rolling(period, min_periods=1).max()
    lowest_low = low.rolling(period, min_periods=1).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    return williams_r


def calculate_cci(high, low, close, period=20):
    """
    Calculate Commodity Channel Index (CCI).
    
    CCI measures the variation of a security's price from its statistical mean.
    Values above +100 indicate that the price is well above the average,
    while values below -100 indicate that the price is well below the average.
    
    Args:
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of closing prices
        period (int): Number of periods for calculation (default: 20)
        
    Returns:
        pd.Series: CCI values
    """
    tp = (high + low + close) / 3  # Typical Price
    sma_tp = tp.rolling(period, min_periods=1).mean()
    mad = tp.rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    return cci


def calculate_moving_averages(prices, windows):
    """
    Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
    for multiple time windows.
    
    Args:
        prices (pd.Series): Series of closing prices
        windows (list): List of window periods for calculation
        
    Returns:
        dict: Dictionary containing SMA and EMA values for each window
    """
    ma_dict = {}
    
    for window in windows:
        # Simple Moving Average
        ma_dict[f'SMA_{window}'] = prices.rolling(window, min_periods=1).mean()
        
        # Exponential Moving Average
        ma_dict[f'EMA_{window}'] = prices.ewm(span=window).mean()
    
    return ma_dict


def calculate_volume_indicators(volume, prices, windows):
    """
    Calculate volume-based technical indicators.
    
    Args:
        volume (pd.Series): Series of volume data
        prices (pd.Series): Series of closing prices
        windows (list): List of window periods for calculation
        
    Returns:
        dict: Dictionary containing volume-based indicators
    """
    volume_dict = {}
    returns = prices.pct_change()
    
    for window in windows:
        # Volume moving average
        volume_dict[f'Volume_SMA_{window}'] = volume.rolling(window, min_periods=1).mean()
        
        # Volume ratio
        volume_dict[f'Volume_Ratio_{window}'] = volume / (volume_dict[f'Volume_SMA_{window}'] + 1e-10)
        
        # On-Balance Volume
        obv_signal = np.sign(returns)
        volume_dict[f'OBV_{window}'] = (volume * obv_signal).rolling(window, min_periods=1).sum()
        
        # Volume-Weighted Moving Average
        volume_dict[f'VWMA_{window}'] = (
            (prices * volume).rolling(window, min_periods=1).sum() /
            volume.rolling(window, min_periods=1).sum()
        )
    
    return volume_dict


def calculate_momentum_indicators(prices, periods):
    """
    Calculate momentum indicators for different time periods.
    
    Args:
        prices (pd.Series): Series of closing prices
        periods (list): List of periods for momentum calculation
        
    Returns:
        dict: Dictionary containing momentum indicators
    """
    momentum_dict = {}
    
    for period in periods:
        # Rate of Change (Momentum)
        momentum_dict[f'Momentum_{period}'] = ((prices / prices.shift(period)) - 1) * 100
        
        # Price returns
        momentum_dict[f'Return_{period}D'] = prices.pct_change(periods=period)
        
        # Log returns
        momentum_dict[f'Log_Return_{period}D'] = np.log(prices / prices.shift(period))
    
    return momentum_dict


def calculate_volatility_indicators(returns, windows):
    """
    Calculate volatility-based indicators.
    
    Args:
        returns (pd.Series): Series of price returns
        windows (list): List of window periods for calculation
        
    Returns:
        dict: Dictionary containing volatility indicators
    """
    volatility_dict = {}
    
    for window in windows:
        # Standard volatility
        volatility_dict[f'Volatility_{window}'] = returns.rolling(window, min_periods=1).std()
        
        # Realized volatility (annualized)
        volatility_dict[f'Realized_Vol_{window}'] = (
            returns.rolling(window, min_periods=1).std() * np.sqrt(252)
        )
        
        # Volatility of volatility
        volatility_dict[f'Vol_of_Vol_{window}'] = (
            volatility_dict[f'Realized_Vol_{window}'].rolling(window, min_periods=1).std()
        )
        
        # Return skewness and kurtosis
        volatility_dict[f'Return_Skew_{window}'] = returns.rolling(
            window, min_periods=3
        ).apply(lambda x: skew(x) if len(x) >= 3 else 0)
        
        volatility_dict[f'Return_Kurt_{window}'] = returns.rolling(
            window, min_periods=3
        ).apply(lambda x: kurtosis(x) if len(x) >= 3 else 0)
    
    return volatility_dict


def calculate_trend_indicators(prices, short_windows, long_windows):
    """
    Calculate trend strength indicators using moving average ratios.
    
    Args:
        prices (pd.Series): Series of closing prices
        short_windows (list): List of short-term periods
        long_windows (list): List of long-term periods
        
    Returns:
        dict: Dictionary containing trend indicators
    """
    trend_dict = {}
    
    # Calculate moving averages
    ma_dict = calculate_moving_averages(prices, short_windows + long_windows)
    
    # Calculate ratios for trend analysis
    trend_combinations = [
        (5, 20), (10, 50), (20, 50), (5, 10)
    ]
    
    for short, long in trend_combinations:
        if f'SMA_{short}' in ma_dict and f'SMA_{long}' in ma_dict:
            trend_dict[f'SMA_{short}_{long}_Ratio'] = (
                ma_dict[f'SMA_{short}'] / (ma_dict[f'SMA_{long}'] + 1e-10)
            )
        
        if f'EMA_{short}' in ma_dict and f'EMA_{long}' in ma_dict:
            trend_dict[f'EMA_{short}_{long}_Ratio'] = (
                ma_dict[f'EMA_{short}'] / (ma_dict[f'EMA_{long}'] + 1e-10)
            )
    
    # Price to moving average ratios
    for window in short_windows + long_windows:
        if f'SMA_{window}' in ma_dict:
            trend_dict[f'Price_SMA_Ratio_{window}'] = prices / (ma_dict[f'SMA_{window}'] + 1e-10)
        
        if f'EMA_{window}' in ma_dict:
            trend_dict[f'Price_EMA_Ratio_{window}'] = prices / (ma_dict[f'EMA_{window}'] + 1e-10)
    
    return trend_dict