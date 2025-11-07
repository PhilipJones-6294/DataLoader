#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
                    Data Pipeline for Nvidia DGX
==============================================================================

Refactored from Google Colab notebook for Nvidia DGX environment.
This script performs market data acquisition and feature engineering for SPY/VIX
and other market indicators.

Original: Afterford_Load_Data.ipynb
Refactored: 2025-11-07
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
import time

import pandas as pd
import numpy as np
import talib
import yfinance as yf

# HMM and Scientific Computing
from hmmlearn import hmm
from scipy.stats import lognorm
from numpy.linalg import matrix_power
from scipy.optimize import minimize
from scipy.signal import find_peaks, detrend


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class Config:
    """Configuration parameters for the data pipeline"""

    def __init__(self, base_output_dir: str = None):
        # --- Base Output Directory ---
        self.BASE_OUTPUT_DIR = base_output_dir or os.environ.get(
            'DATA_PIPELINE_OUTPUT_DIR',
            '/data/afterford/output'
        )

        # --- Core Directories ---
        self.CORE_DIRECTORIES = {
            'output': self.BASE_OUTPUT_DIR,
            'data': os.path.join(self.BASE_OUTPUT_DIR, 'data'),
            'logs': os.path.join(self.BASE_OUTPUT_DIR, 'logs'),
            'models': os.path.join(self.BASE_OUTPUT_DIR, 'models'),
            'visualizations': os.path.join(self.BASE_OUTPUT_DIR, 'visualizations'),
            'raw_data': os.path.join(self.BASE_OUTPUT_DIR, 'data', 'raw'),
            'processed_data': os.path.join(self.BASE_OUTPUT_DIR, 'data', 'processed'),
            'temp': os.path.join(self.BASE_OUTPUT_DIR, 'temp'),
        }

        # --- Key File Paths ---
        self.RAW_DATA_FILE = os.path.join(self.CORE_DIRECTORIES['raw_data'], 'data_raw.csv')
        self.MARKET_RAW_FILE = os.path.join(self.CORE_DIRECTORIES['raw_data'], 'market_raw.csv')
        self.PROCESSED_DATA_FILE_PATH = os.path.join(self.CORE_DIRECTORIES['processed_data'], 'processed_data.csv')
        self.FINAL_DATA_FILE_PATH = os.path.join(self.CORE_DIRECTORIES['output'], 'data.csv')
        self.ETF_DATA_FILE_PATH = os.path.join(self.CORE_DIRECTORIES['output'], 'data_etf.csv')

        # --- General Settings ---
        self.RANDOM_SEED = 42
        self.DATE_COLUMN = 'Date'
        self.TARGET_COLUMN = 'Target'
        self.MOE_COLUMN = 'MOE'
        self.ETF_SYMBOLS = ["SPXS", "SPXL"]

        # --- Data Pipeline Configuration ---
        self.TARGET_LOOKAHEAD_DAYS = 5
        self.TARGET_MOE_THRESHOLD = 5.0
        self.TARGET_ROUND_DECIMALS = 4

        # Column Definitions
        self.SPY_OHLCV_COLS = ['SPY_Open', 'SPY_High', 'SPY_Low', 'SPY_Close', 'SPY_Volume']
        self.BASE_INPUT_COLS = self.SPY_OHLCV_COLS + ['VIXClose']
        self.PIPELINE_COLS_NOT_TO_LAG = self.SPY_OHLCV_COLS + [self.TARGET_COLUMN, self.MOE_COLUMN]
        self.FEATURE_DROP_COLS = ['wcl']

        # Lag Generation
        self.PIPELINE_MAX_LAG = 15

        # MACD Configuration
        self.MACD_FAST_PERIOD = 12
        self.MACD_SLOW_PERIOD = 26
        self.MACD_SIGNAL_PERIOD = 9

        # Final Data Cleaning
        self.PIPELINE_DROP_INITIAL_ROWS = 100
        self.PIPELINE_ROUND_DECIMALS = 4

        # --- Processor-Specific Configurations ---
        # Cycle Processor
        self.CYCLE_PRICE_COLUMN = 'SPY_Close'
        self.CYCLE_LOOKBACK_WINDOW = 2520
        self.CYCLE_ROUND_DECIMALS = 4
        self.CYCLE_FFT_PARAMS = None

        # HMM Processor
        self.HMM_VIX_COLUMN = 'VIXClose'
        self.HMM_STOCH_K_COLUMN = 'StochRSI_K'
        self.HMM_N_FORWARD_DAYS = 5
        self.HMM_VIX_N_STATES = 3
        self.HMM_RSI_N_STATES = 4
        self.HMM_VIX_N_ITER = 150
        self.HMM_RSI_N_ITER = 150
        self.HMM_ROUND_DECIMALS = 4

        # Mandelbrot Processor
        self.MANDELBROT_PRICE_COLUMN = 'SPY_Close'
        self.MANDELBROT_VOLUME_COLUMN = 'SPY_Volume'
        self.MANDELBROT_MAX_ITER = 100
        self.MANDELBROT_ROUND_DECIMALS = 4

        # Technical Analyzer
        self.TA_STOCHRSI_PERIOD = 14

        # Download settings
        self.DOWNLOAD_DELAY = 5.0  # Seconds between downloads to avoid rate limits


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """Configure logging for the pipeline"""
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    logger = logging.getLogger()

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='hmmlearn')
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
    warnings.filterwarnings("ignore", category=UserWarning, message="Model is not converging")
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def setup_directories(directories: Dict[str, str]) -> None:
    """Create necessary directories"""
    logger = logging.getLogger()
    logger.info("Setting up directories...")
    for key, path in directories.items():
        if path:
            try:
                os.makedirs(path, exist_ok=True)
                logger.debug(f"Directory '{key}' ready at: {path}")
            except OSError as e:
                logger.error(f"Could not create directory '{key}' at '{path}': {e}")
                raise
    logger.info("Directory setup complete.")


# ============================================================================
# DATA ACQUISITION SERVICE
# ============================================================================

class DataAcquisitionService:
    """Downloads and processes financial market data using yfinance"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = config.CORE_DIRECTORIES['raw_data']
        self.default_start = "1993-01-29"  # SPY inception date

    def download_market_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_futures: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Download, process, and save financial market data sequentially"""

        if not os.path.exists(self.output_dir):
            self.logger.error(f"Output directory does not exist: {self.output_dir}")
            return None, None

        # Determine date range
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                start_datetime = end_dt - timedelta(days=7305)  # ~20 years
            except ValueError:
                end_dt = datetime.now()
                start_datetime = end_dt - timedelta(days=7305)

            spy_inception_dt = datetime.strptime(self.default_start, "%Y-%m-%d")
            start_date = self.default_start if start_datetime < spy_inception_dt else start_datetime.strftime("%Y-%m-%d")

        self.logger.info(f"Downloading data from {start_date} to {end_date}")

        # Define symbols
        spy_symbol = "SPY"
        vix_symbol = "^VIX"
        other_symbols = [
            "^DJI", "^RUT", "^FTSE", "^N225", "^HSI", "^BSESN", "^KS11",
            "TLT", "GOVT", "TFLO", "USFR", "^TNX", "^TYX", "GC=F", "CL=F"
        ]

        if include_futures:
            other_symbols.append("ES=F")
            self.logger.info("Including ES=F futures.")

        all_symbols = [spy_symbol, vix_symbol] + other_symbols

        # Download data sequentially
        downloaded_data = {}
        failed_symbols = []

        self.logger.info(f"Downloading {len(all_symbols)} symbols with {self.config.DOWNLOAD_DELAY}s delay...")
        for symbol in all_symbols:
            try:
                self.logger.debug(f"Downloading: {symbol}")
                data_single = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                    timeout=90
                )

                if data_single.empty:
                    self.logger.warning(f"No data returned for symbol: {symbol}")
                    failed_symbols.append(symbol)
                else:
                    downloaded_data[symbol] = data_single

                self.logger.debug(f"Pausing for {self.config.DOWNLOAD_DELAY}s...")
                time.sleep(self.config.DOWNLOAD_DELAY)

            except Exception as e:
                self.logger.error(f"Failed to download symbol {symbol}: {e}")
                failed_symbols.append(symbol)
                time.sleep(self.config.DOWNLOAD_DELAY)

        if not downloaded_data:
            self.logger.error("No symbols were successfully downloaded.")
            return None, None

        if failed_symbols:
            self.logger.warning(f"Failed to download: {failed_symbols}")

        # Process SPY and VIX data
        spy_data_raw_df = None
        base_index = None

        try:
            if spy_symbol in downloaded_data:
                spy_ohlcv = downloaded_data[spy_symbol][['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                if spy_ohlcv['Close'].squeeze().isnull().all():
                    self.logger.error("SPY Close is all NaN.")
                    return None, None

                new_spy_cols = []
                for col in spy_ohlcv.columns:
                    actual_col_name = col[0] if isinstance(col, tuple) else col
                    new_spy_cols.append(f"SPY_{actual_col_name}")

                spy_ohlcv.columns = new_spy_cols
                spy_data_raw_df = spy_ohlcv
                base_index = spy_data_raw_df.index
                self.logger.debug(f"SPY data processed. Base index length: {len(base_index)}")
            else:
                self.logger.error(f"SPY symbol download failed. Cannot proceed.")
                return None, None

            if vix_symbol in downloaded_data:
                vix_close_series = downloaded_data[vix_symbol]['Close']

                if not (vix_close_series.isnull().all()).item():
                    vix_series_reindexed = vix_close_series.reindex(base_index, method='ffill')
                    vix_series_reindexed.name = vix_symbol
                    spy_data_raw_df = spy_data_raw_df.join(vix_series_reindexed)

                    if vix_symbol in spy_data_raw_df.columns:
                        spy_data_raw_df.rename(columns={vix_symbol: 'VIXClose'}, inplace=True)
                        self.logger.info("Renamed column '^VIX' to 'VIXClose'.")
                    else:
                        self.logger.error("Could not find '^VIX' column to rename.")
                else:
                    self.logger.warning("VIX Close data is all NaN.")
            else:
                self.logger.warning("VIX symbol download failed or missing.")

            spy_cols_exist = [col for col in spy_data_raw_df.columns if col.startswith('SPY_')]
            spy_data_raw_df = spy_data_raw_df.ffill().dropna(subset=spy_cols_exist)

            if spy_data_raw_df.empty:
                self.logger.error("SPY+VIX data empty after cleaning.")
                return None, None

            if not isinstance(spy_data_raw_df.index, pd.DatetimeIndex):
                spy_data_raw_df.index = pd.to_datetime(spy_data_raw_df.index)

            for col in spy_data_raw_df.select_dtypes(include=np.number).columns:
                spy_data_raw_df[col] = spy_data_raw_df[col].round(2)

            self.logger.debug(f"SPY/VIX processing complete. Shape: {spy_data_raw_df.shape}")

        except Exception as e:
            self.logger.error(f"SPY/VIX processing error: {e}", exc_info=True)
            return None, None

        # Process other market data
        market_data_raw_df = None
        try:
            market_data_closes = {}

            if base_index is None:
                self.logger.error("Base index from SPY not available.")
            else:
                for symbol in other_symbols:
                    if symbol in downloaded_data:
                        clean_symbol = symbol.replace('^', '').replace('=F', 'F').replace('=X', 'X').replace('=', '')
                        symbol_data = downloaded_data[symbol]

                        if (isinstance(symbol_data, pd.DataFrame) and
                            'Close' in symbol_data.columns and
                            not (symbol_data['Close'].isnull().all()).item()):

                            market_data_closes[f"{clean_symbol}Close"] = symbol_data['Close'].squeeze().reindex(base_index, method='ffill')

                if market_data_closes:
                    market_data_raw_df = pd.DataFrame(market_data_closes, index=base_index)
                    market_data_raw_df = market_data_raw_df.ffill().dropna(how='all')

                    if not market_data_raw_df.empty:
                        if not isinstance(market_data_raw_df.index, pd.DatetimeIndex):
                            market_data_raw_df.index = pd.to_datetime(market_data_raw_df.index)

                        for col in market_data_raw_df.select_dtypes(include=np.number).columns:
                            market_data_raw_df[col] = market_data_raw_df[col].round(2)

                        self.logger.debug(f"Market data shape: {market_data_raw_df.shape}")
                    else:
                        self.logger.warning("Market data DataFrame empty after processing.")
                else:
                    self.logger.warning("No data processed for market symbols.")

        except Exception as e:
            self.logger.error(f"Market data processing error: {e}", exc_info=True)

        # Save data
        spy_df_out = spy_data_raw_df.copy() if spy_data_raw_df is not None else None
        market_df_out = market_data_raw_df.copy() if market_data_raw_df is not None else None

        try:
            if spy_df_out is not None:
                spy_df_out.to_csv(self.config.RAW_DATA_FILE, index=True)
                self.logger.info(f"Saved SPY+VIX raw data to: {self.config.RAW_DATA_FILE}")
            else:
                self.logger.error("No SPY/VIX data available for saving.")
                return None, None

            if market_df_out is not None and not market_df_out.empty:
                market_df_out.to_csv(self.config.MARKET_RAW_FILE, index=True)
                self.logger.info(f"Saved Market raw data to: {self.config.MARKET_RAW_FILE}")
            else:
                self.logger.warning("No market data to save.")

        except Exception as e:
            self.logger.error(f"Failed to save raw data files: {e}", exc_info=True)
            return None, None

        return spy_df_out, market_df_out


# ============================================================================
# FEATURE ENGINEERING SERVICE
# ============================================================================

class FeatureEngineeringService:
    """Custom technical indicators and feature engineering methods"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def ichimoku_cloud(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        senkou_shift: int = 26,
        chikou_shift: int = 26
    ) -> pd.DataFrame:
        """Calculates Ichimoku Cloud components"""

        self.logger.debug("Calculating Ichimoku Cloud...")

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(senkou_shift)

        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(senkou_shift)

        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-chikou_shift)

        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }, index=close.index)

    def psar(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        initial_af: float = 0.02,
        max_af: float = 0.2,
        af_step: float = 0.02
    ) -> pd.DataFrame:
        """Calculates Parabolic SAR (Stop and Reverse)"""

        self.logger.debug("Calculating Parabolic SAR...")

        original_index = high.index
        length = len(close)

        if length < 2:
            nan_series = pd.Series([np.nan]*length, index=original_index)
            zero_series = pd.Series([0]*length, index=original_index)
            return pd.DataFrame({
                'psar': nan_series,
                'psar_up': zero_series,
                'psar_down': zero_series
            })

        # Combine and drop NaNs
        combined = pd.DataFrame({'high': high, 'low': low}).dropna()

        if len(combined) < 2:
            nan_series = pd.Series([np.nan]*length, index=original_index)
            zero_series = pd.Series([0]*length, index=original_index)
            return pd.DataFrame({
                'psar': nan_series,
                'psar_up': zero_series,
                'psar_down': zero_series
            })

        # Convert to numpy for efficiency
        high_np = combined['high'].values
        low_np = combined['low'].values
        valid_index = combined.index
        n_valid = len(combined)

        psar_np = np.full(n_valid, np.nan)
        bull_trend_np = np.full(n_valid, True)
        af_np = np.full(n_valid, initial_af)
        ep_np = np.full(n_valid, np.nan)

        # Initial conditions
        psar_np[0] = low_np[0]
        ep_np[0] = high_np[0]

        # PSAR calculation loop
        for i in range(1, n_valid):
            prev_psar = psar_np[i-1]
            prev_af = af_np[i-1]
            prev_ep = ep_np[i-1]
            prev_bull = bull_trend_np[i-1]
            current_psar = prev_psar

            if prev_bull:
                psar_candidate = prev_psar + prev_af * (prev_ep - prev_psar)
                psar_candidate = min(psar_candidate, low_np[i-1])
                if i > 1:
                    psar_candidate = min(psar_candidate, low_np[i-2])

                if low_np[i] < psar_candidate:
                    bull_trend_np[i] = False
                    current_psar = max(prev_ep, high_np[i])
                    if i > 0:
                        current_psar = max(current_psar, high_np[i-1])
                    ep_np[i] = low_np[i]
                    af_np[i] = initial_af
                else:
                    bull_trend_np[i] = True
                    current_psar = psar_candidate
                    if high_np[i] > prev_ep:
                        ep_np[i] = high_np[i]
                        af_np[i] = min(prev_af + af_step, max_af)
                    else:
                        ep_np[i] = prev_ep
                        af_np[i] = prev_af
            else:
                psar_candidate = prev_psar - prev_af * (prev_psar - prev_ep)
                psar_candidate = max(psar_candidate, high_np[i-1])
                if i > 1:
                    psar_candidate = max(psar_candidate, high_np[i-2])

                if high_np[i] > psar_candidate:
                    bull_trend_np[i] = True
                    current_psar = min(prev_ep, low_np[i])
                    if i > 0:
                        current_psar = min(current_psar, low_np[i-1])
                    ep_np[i] = high_np[i]
                    af_np[i] = initial_af
                else:
                    bull_trend_np[i] = False
                    current_psar = psar_candidate
                    if low_np[i] < prev_ep:
                        ep_np[i] = low_np[i]
                        af_np[i] = min(prev_af + af_step, max_af)
                    else:
                        ep_np[i] = prev_ep
                        af_np[i] = prev_af

            psar_np[i] = current_psar

        # Create final Series
        psar_series = pd.Series(psar_np, index=valid_index).reindex(original_index)
        psar_up_np = np.where(bull_trend_np, psar_np, 0.0)
        psar_down_np = np.where(~bull_trend_np, psar_np, 0.0)
        psar_up_series = pd.Series(psar_up_np, index=valid_index).reindex(original_index, fill_value=0.0)
        psar_down_series = pd.Series(psar_down_np, index=valid_index).reindex(original_index, fill_value=0.0)

        return pd.DataFrame({
            'psar': psar_series,
            'psar_up': psar_up_series,
            'psar_down': psar_down_series
        })

    def lamplighter(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        open_: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Placeholder for the Lamplighter indicator calculation"""

        self.logger.warning("Using placeholder Lamplighter logic.")
        typical_price = (high + low + close) / 3
        lamplighter_values = typical_price * volume

        return pd.Series(lamplighter_values, index=close.index, name='lamplighter')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ehlers_super_smoother(
    prices: pd.Series,
    cutoff_period: float = 4.4,
    alpha_factor: float = 0.147
) -> pd.Series:
    """
    Implements Ehlers Super Smoother with fixed parameters

    Args:
        prices: Pandas Series of price data
        cutoff_period: The cutoff period for the low-pass filter (4.4)
        alpha_factor: Fixed alpha value (0.147)

    Returns:
        Pandas Series of smoothed prices
    """
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    n = len(prices)
    smoothed_values = []
    smoothed_values.append(prices.iloc[0])

    # Calculate filter coefficients
    a1 = np.exp(-1.414 * np.pi / cutoff_period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / cutoff_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    alpha = alpha_factor

    # Initialize filter components
    filt1 = prices.iloc[0]
    filt2 = prices.iloc[0]

    for i in range(1, n):
        filt1 = c1 * prices.iloc[i] + c2 * filt1 + c3 * filt2
        filt2 = filt1
        smoothed_values.append(filt1)

    smoothed = pd.Series(smoothed_values, index=prices.index)
    return smoothed


def generate_lags(
    df: pd.DataFrame,
    cols_to_lag: List[str],
    max_lag: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """Generates lags 0 through max_lag for specified columns"""

    if not cols_to_lag or max_lag < 0 or df is None or df.empty:
        logger.warning("Skipping lag generation: Invalid input.")
        return pd.DataFrame(index=df.index if df is not None else None)

    all_lags_list = []
    valid_cols_to_lag = [col for col in cols_to_lag if col in df.columns]

    if not valid_cols_to_lag:
        logger.warning(f"None of the requested columns to lag exist: {cols_to_lag}")
        return pd.DataFrame(index=df.index)

    df_subset = df[valid_cols_to_lag].copy()

    for i in range(0, max_lag + 1):
        lagged_subset = df_subset.shift(i)
        lagged_subset = lagged_subset.add_suffix(f'_lag{i}')
        all_lags_list.append(lagged_subset)

    if all_lags_list:
        lagged_features_wide = pd.concat(all_lags_list, axis=1)
        return lagged_features_wide
    else:
        return pd.DataFrame(index=df.index)


def download_etf_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    round_digits: int,
    download_delay: float,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """Downloads OHLCV data for specified ETF symbols"""

    logger.info(f"Downloading ETF data for {symbols} from {start_date} to {end_date}")

    if not symbols:
        logger.warning("No ETF symbols provided.")
        return None

    all_etf_data_dict = {}
    failed_symbols = []
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    for symbol in symbols:
        try:
            logger.debug(f"Downloading ETF: {symbol}")
            data_single = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                timeout=90
            )

            if data_single.empty:
                logger.warning(f"No data returned for ETF: {symbol}")
                failed_symbols.append(symbol)
            elif not all(col in data_single.columns for col in required_cols):
                logger.warning(f"ETF '{symbol}' missing required columns.")
                failed_symbols.append(symbol)
            elif (data_single['Close'].isnull().all()).item():
                logger.warning(f"ETF '{symbol}' Close column is all NaN.")
                failed_symbols.append(symbol)
            else:
                ohlcv_data = data_single[required_cols].copy()
                ohlcv_data.columns = [f"{symbol}_{col}" for col in ohlcv_data.columns]
                all_etf_data_dict[symbol] = ohlcv_data

            logger.debug(f"Pausing for {download_delay}s...")
            time.sleep(download_delay)

        except Exception as e:
            logger.error(f"Failed to download ETF {symbol}: {e}")
            failed_symbols.append(symbol)
            time.sleep(download_delay)

    if not all_etf_data_dict:
        logger.error("No ETF symbols were successfully downloaded.")
        return None

    if failed_symbols:
        logger.warning(f"Failed ETF symbols: {failed_symbols}")

    # Find common index
    try:
        first_df = next(iter(all_etf_data_dict.values()))
        common_index = pd.to_datetime(first_df.index)

        if not isinstance(common_index, pd.DatetimeIndex):
            raise ValueError("Could not convert to DatetimeIndex.")

        for df in all_etf_data_dict.values():
            current_index = pd.to_datetime(df.index)
            if isinstance(current_index, pd.DatetimeIndex):
                common_index = common_index.intersection(current_index)

    except Exception as idx_err:
        logger.error(f"Error processing indices: {idx_err}")
        return None

    if common_index.empty:
        logger.warning("No common dates found across ETFs.")
        return None

    logger.debug(f"Found {len(common_index)} common dates for ETFs.")

    # Reindex and combine
    reindexed_dfs = []
    for symbol, df in all_etf_data_dict.items():
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            reindexed_dfs.append(df.reindex(common_index))
        except Exception as reindex_err:
            logger.warning(f"Could not reindex ETF {symbol}: {reindex_err}")

    if not reindexed_dfs:
        logger.error("No ETF data could be reindexed.")
        return None

    etf_df = pd.concat(reindexed_dfs, axis=1)
    logger.debug(f"Combined ETF DataFrame shape: {etf_df.shape}")

    # Clean data
    if not isinstance(etf_df.index, pd.DatetimeIndex):
        etf_df.index = pd.to_datetime(etf_df.index)

    etf_df = etf_df.ffill()

    processed_symbols = list(all_etf_data_dict.keys())
    close_cols = [f"{s}_Close" for s in processed_symbols if f"{s}_Close" in etf_df.columns]

    if close_cols:
        initial_rows = len(etf_df)
        etf_df.dropna(subset=close_cols, how='any', inplace=True)
        if len(etf_df) < initial_rows:
            logger.debug(f"Dropped {initial_rows - len(etf_df)} rows due to NaN in ETF Close columns.")

    if etf_df.empty:
        logger.warning("ETF DataFrame empty after cleaning.")
        return None

    # Round numeric columns
    try:
        num_cols = etf_df.select_dtypes(include=np.number).columns
        etf_df[num_cols] = etf_df[num_cols].round(round_digits)
        logger.debug(f"Rounded ETF data to {round_digits} decimals.")
    except Exception as round_err:
        logger.error(f"Failed to round ETF data: {round_err}")

    logger.info(f"Successfully processed ETF data. Shape: {etf_df.shape}")
    return etf_df


# ============================================================================
# Continue with processor implementations in next message...
# ============================================================================
