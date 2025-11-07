#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
                        Data Processors Module
==============================================================================

Contains all processor implementations for the data pipeline:
- Technical Analyzer
- Cycle Processor
- HMM Processor
- Mandelbrot Processor
"""

import logging
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional

from hmmlearn import hmm
from scipy.stats import lognorm
from numpy.linalg import matrix_power
from scipy.optimize import minimize
from scipy.signal import find_peaks, detrend


# ============================================================================
# TECHNICAL ANALYZER PROCESSOR
# ============================================================================

def run_technical_analyzer(
    input_df: pd.DataFrame,
    feature_eng_service,
    logger: logging.Logger,
    mfi_period: int = 23,
    willr_long_period: int = 252,
    willr_medium_period: int = 50,
    stochrsi_period: int = 14,
    macd_fast_period: int = 12,
    macd_slow_period: int = 26,
    macd_signal_period: int = 9,
    adosc_fast_period: int = 3,
    adosc_slow_period: int = 10,
    di_period: int = 23,
    mama_fastlimit: float = 0.5,
    mama_slowlimit: float = 0.05,
    roc_trend_period: int = 5,
    round_digits: Optional[int] = 4
) -> pd.DataFrame:
    """Calculates technical indicators for the input DataFrame"""

    logger.info("Running Technical Analyzer...")

    if not isinstance(input_df.index, pd.DatetimeIndex):
        logger.error("Input DataFrame must have DatetimeIndex.")
        return input_df.copy()

    indicators_df = input_df.copy()

    try:
        required_columns = ['SPY_High', 'SPY_Low', 'SPY_Close', 'SPY_Volume']
        missing_required = [col for col in required_columns if col not in indicators_df.columns]

        if missing_required:
            logger.error(f"Missing required columns: {missing_required}")
            return input_df.copy()

        # Base calculations
        indicators_df['wcl'] = (
            indicators_df['SPY_High'] +
            indicators_df['SPY_Low'] +
            indicators_df['SPY_Close']
        ) / 3

        # TA-Lib indicators
        logger.debug("Calculating TA-Lib indicators...")
        with np.errstate(divide='ignore', invalid='ignore'):
            # Hilbert Transform
            indicators_df['sine'], indicators_df['lead_sine'] = talib.HT_SINE(indicators_df['wcl'])
            indicators_df['cycle'] = talib.HT_DCPERIOD(indicators_df['wcl'])
            indicators_df['trendmode'] = talib.HT_TRENDMODE(indicators_df['wcl'])
            indicators_df['trendline'] = talib.HT_TRENDLINE(indicators_df['wcl'])
            indicators_df['roc_trend'] = talib.ROC(indicators_df['trendline'], timeperiod=roc_trend_period)

            # MAMA
            indicators_df['fama'], indicators_df['mama'] = talib.MAMA(
                indicators_df['wcl'],
                fastlimit=mama_fastlimit,
                slowlimit=mama_slowlimit
            )
            indicators_df['fama_mama'] = (
                (indicators_df['fama'] - indicators_df['mama']) /
                indicators_df['mama'].replace(0, np.nan)
            )

            # Volume
            indicators_df['obv'] = talib.OBV(indicators_df['wcl'], indicators_df['SPY_Volume'])
            indicators_df['ChaikinAD'] = talib.ADOSC(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                indicators_df['SPY_Volume'],
                fastperiod=adosc_fast_period,
                slowperiod=adosc_slow_period
            )
            indicators_df['MFI'] = talib.MFI(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                indicators_df['SPY_Volume'],
                timeperiod=mfi_period
            )

            # Momentum/Oscillator
            indicators_df['longrun'] = talib.WILLR(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                timeperiod=willr_long_period
            )
            indicators_df['mediumrun'] = talib.WILLR(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                timeperiod=willr_medium_period
            )

            indicators_df['StochRSI_K'], indicators_df['StochRSI_D'] = talib.STOCHRSI(
                indicators_df['SPY_Close'],
                timeperiod=stochrsi_period
            )

            indicators_df['macd'], indicators_df['macdsignal'], indicators_df['macdhist'] = talib.MACD(
                indicators_df['SPY_Close'],
                fastperiod=macd_fast_period,
                slowperiod=macd_slow_period,
                signalperiod=macd_signal_period
            )

            # Directional Movement Index
            indicators_df['minus_DI'] = talib.MINUS_DI(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                timeperiod=di_period
            )
            indicators_df['plus_DI'] = talib.PLUS_DI(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                timeperiod=di_period
            )

            # Stochastic Oscillator
            indicators_df['slowk'], indicators_df['slowd'] = talib.STOCH(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )

        # Optional indicators
        if 'TLTClose' in indicators_df.columns and 'IEFClose' in indicators_df.columns:
            indicators_df['dfir'] = indicators_df['TLTClose'] - indicators_df['IEFClose']
        else:
            logger.debug("TLTClose or IEFClose not found, skipping 'dfir'.")

        # Custom Feature Engineering indicators
        logger.debug("Calculating custom indicators (Ichimoku, PSAR, Lamplighter)...")

        # Ichimoku
        ichimoku_dict = feature_eng_service.ichimoku_cloud(
            indicators_df['SPY_High'],
            indicators_df['SPY_Low'],
            indicators_df['SPY_Close']
        )
        for key, value_series in ichimoku_dict.items():
            if isinstance(value_series, pd.Series):
                indicators_df[key] = value_series
            else:
                logger.warning(f"Ichimoku component '{key}' not a Series.")

        if 'tenkan_sen' in indicators_df and 'kijun_sen' in indicators_df:
            indicators_df['delta_tenkan_kiju'] = indicators_df['tenkan_sen'] - indicators_df['kijun_sen']
        else:
            logger.warning("Missing Ichimoku components for delta_tenkan_kiju.")

        if 'senkou_span_a' in indicators_df and 'senkou_span_b' in indicators_df:
            indicators_df['delta_spanA_spanB'] = indicators_df['senkou_span_a'] - indicators_df['senkou_span_b']
        else:
            logger.warning("Missing Ichimoku components for delta_spanA_spanB.")

        # PSAR
        psar = feature_eng_service.psar(
            indicators_df['SPY_High'],
            indicators_df['SPY_Low'],
            indicators_df['SPY_Close']
        )
        indicators_df['psar'] = psar['psar']
        indicators_df['psar_up'] = psar['psar_up'].replace(0, np.nan)
        indicators_df['psar_down'] = psar['psar_down'].replace(0, np.nan)

        with np.errstate(divide='ignore', invalid='ignore'):
            indicators_df['psar1'] = indicators_df['psar_up'] / indicators_df['SPY_Close'].replace(0, np.nan)
            indicators_df['psar2'] = indicators_df['psar_down'] / indicators_df['SPY_Close'].replace(0, np.nan)

        # Lamplighter
        if 'SPY_Open' in indicators_df.columns:
            indicators_df['lamplighter'] = feature_eng_service.lamplighter(
                indicators_df['SPY_High'],
                indicators_df['SPY_Low'],
                indicators_df['SPY_Close'],
                indicators_df['SPY_Open'],
                indicators_df['SPY_Volume']
            )
        else:
            logger.debug("SPY_Open not found, skipping 'lamplighter'.")

        # Final cleanup
        indicators_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optional rounding
        if round_digits is not None:
            try:
                num_cols = indicators_df.select_dtypes(include=np.number).columns
                indicators_df[num_cols] = indicators_df[num_cols].round(round_digits)
                logger.debug(f"Rounded TA features to {round_digits} decimals.")
            except Exception as round_err:
                logger.error(f"Failed to round TA features: {round_err}")

        logger.info("Technical indicator calculation finished.")
        return indicators_df

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        return input_df.copy()


# ============================================================================
# CYCLE PROCESSOR
# ============================================================================

def detect_dominant_cycles_fft(
    price_data: pd.Series,
    logger: logging.Logger,
    short_range: Tuple[int, int] = (5, 30),
    medium_range: Tuple[int, int] = (31, 365),
    long_min: int = 366,
    n_short: int = 4,
    n_medium: int = 3,
    n_long: int = 2,
    min_amplitude_ratio: float = 0.05
) -> Dict[str, List[int]]:
    """Detects dominant cycle periods using FFT amplitude spectrum peaks"""

    logger.info("Starting FFT Cycle Detection...")

    n_samples = len(price_data)
    min_required_length = max(short_range[1], medium_range[1], long_min) * 2

    if n_samples < min_required_length:
        logger.debug(f"Data length ({n_samples}) might be short for reliable detection.")

    try:
        valid_data = price_data.dropna().values.astype(float)
        detrended_data = detrend(valid_data) if len(valid_data) >= 2 else valid_data
        logger.debug("Applied linear detrending.")
    except ValueError as e:
        logger.warning(f"Could not detrend data: {e}. Using original data.")
        detrended_data = price_data.fillna(0).values

    if len(detrended_data) < 2:
        logger.error("Not enough data points for FFT.")
        return {'short': [4, 8, 16], 'medium': [], 'long': []}

    # Perform FFT
    fft_values = np.fft.fft(detrended_data)
    fft_freq = np.fft.fftfreq(len(detrended_data))[1:len(detrended_data)//2]
    fft_amplitude = np.abs(fft_values)[1:len(detrended_data)//2]

    if len(fft_freq) == 0:
        logger.warning("Not enough data points for FFT spectrum.")
        return {'short': [4, 8, 16], 'medium': [], 'long': []}

    # Find peaks
    max_amplitude = np.max(fft_amplitude) if len(fft_amplitude) > 0 else 0
    min_height = max_amplitude * min_amplitude_ratio if max_amplitude > 0 else 0

    peak_indices, properties = find_peaks(fft_amplitude, height=min_height)

    selected_cycles = {'short': [], 'medium': [], 'long': []}

    if len(peak_indices) > 0:
        peak_amplitudes = properties['peak_heights']
        valid_freq_mask = fft_freq[peak_indices] != 0
        valid_peak_indices = peak_indices[valid_freq_mask]
        valid_peak_amplitudes = peak_amplitudes[valid_freq_mask]

        if len(valid_peak_indices) > 0:
            peak_periods = 1 / fft_freq[valid_peak_indices]
            detected_peaks = sorted(
                zip(peak_periods, valid_peak_amplitudes),
                key=lambda x: x[1],
                reverse=True
            )

            cycle_counts = {'short': 0, 'medium': 0, 'long': 0}
            max_counts = {'short': n_short, 'medium': n_medium, 'long': n_long}

            logger.debug(f"Found {len(detected_peaks)} significant peaks.")

            for period, amplitude in detected_peaks:
                period_int = int(round(period))
                category = None

                if period_int <= 1:
                    continue

                if short_range[0] <= period_int <= short_range[1]:
                    category = 'short'
                elif medium_range[0] <= period_int <= medium_range[1]:
                    category = 'medium'
                elif period_int >= long_min:
                    category = 'long'

                if (category and
                    cycle_counts[category] < max_counts[category] and
                    period_int not in selected_cycles[category]):
                    selected_cycles[category].append(period_int)
                    cycle_counts[category] += 1

                if all(cycle_counts[cat] >= max_counts[cat] for cat in max_counts):
                    break

            for category in selected_cycles:
                selected_cycles[category].sort()
        else:
            logger.warning("No valid peaks found after filtering zero frequency.")
    else:
        logger.warning("No significant peaks found in FFT amplitude spectrum.")

    # Fallback for short cycles
    if not selected_cycles['short']:
        selected_cycles['short'] = [4, 8, 16]
        logger.info(f"FFT found no short cycles. Using defaults: {selected_cycles['short']}")

    logger.info("FFT Cycle Detection Finished.")

    if not any(p for cat in selected_cycles for p in selected_cycles[cat]):
        logger.warning("FFT analysis did not yield any usable cycles.")

    return selected_cycles


def align_to_minimums(
    data: np.ndarray,
    period: int,
    window_length: int,
    logger: logging.Logger
) -> float:
    """Finds the optimal phase shift for a cosine wave to align its minimums"""

    if window_length <= 2 or period <= 1:
        logger.debug(f"Window/period too short ({window_length}/{period}). Phase=0.")
        return 0.0

    frequency = 2 * np.pi / period
    t = np.arange(window_length)

    # Find minima in price data
    min_distance_samples = max(1, int(period / 4))
    price_minima_indices, _ = find_peaks(-data, distance=min_distance_samples)

    if len(price_minima_indices) == 0:
        logger.debug(f"No distinct minima found for period {period}. Trying percentile.")
        threshold = np.percentile(data, 10)
        price_minima_indices = np.where(data <= threshold)[0]

        if len(price_minima_indices) == 0 or len(price_minima_indices) == window_length:
            price_minima_indices = np.array([np.argmin(data)])

    if len(price_minima_indices) == 0:
        logger.warning(f"Could not determine price minima for period {period}. Phase=0.")
        return 0.0

    # Error function for optimization
    def error_function(phase_param: np.ndarray) -> float:
        phase = phase_param[0]
        cosine_wave = np.cos(frequency * t + phase)
        cosine_minima_indices = np.where(cosine_wave < -0.95)[0]

        if len(cosine_minima_indices) == 0:
            return float(window_length * len(price_minima_indices))

        total_distance = 0.0
        for price_min_idx in price_minima_indices:
            distances = np.abs(price_min_idx - cosine_minima_indices)
            min_dist = np.min(distances) if len(distances) > 0 else window_length
            total_distance += min_dist

        avg_distance = total_distance / len(price_minima_indices) if len(price_minima_indices) > 0 else float(window_length)
        return avg_distance

    # Run optimization
    initial_phase = [0.0]
    bounds = [(-np.pi, np.pi)]
    optimal_phase = initial_phase[0]

    try:
        result = minimize(error_function, initial_phase, method='Powell', bounds=bounds)
        if not result.success:
            logger.warning(f"Phase optimization failed for period {period}. Using Phase=0.")
        else:
            optimal_phase = result.x[0]
            logger.debug(f"Phase optimization success for period {period}. Phase: {optimal_phase:.4f}")
    except Exception as e:
        logger.error(f"Error during phase optimization for period {period}: {e}")

    optimal_phase = (optimal_phase + np.pi) % (2 * np.pi) - np.pi
    return optimal_phase


def generate_minimum_aligned_cosine_waves(
    price_data: pd.Series,
    cycles: Dict[str, List[int]],
    lookback_window: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """Generates cosine waves phase-aligned to price minimums"""

    results = {}
    data_length = len(price_data)
    actual_lookback = min(lookback_window, data_length)

    if actual_lookback <= 2:
        logger.warning(f"Not enough data ({actual_lookback}) for phase alignment.")
        return pd.DataFrame(index=price_data.index)

    # Extract recent data
    recent_data_series = price_data.iloc[-actual_lookback:]
    recent_data = recent_data_series.dropna().values.astype(float)

    if len(recent_data) <= 2:
        logger.warning("Not enough valid data in lookback window.")
        return pd.DataFrame(index=price_data.index)

    if len(recent_data) < actual_lookback * 0.5:
        logger.debug(f"High NaN count in lookback window ({actual_lookback - len(recent_data)} NaNs).")

    t_full = np.arange(data_length)
    composite_cosine_sum = np.zeros(data_length)
    num_waves_summed = 0

    logger.info(f"Generating aligned cosine waves using {len(recent_data)}-period lookback...")

    for category, periods in cycles.items():
        for period in periods:
            if not isinstance(period, (int, float)) or period <= 1:
                logger.warning(f"Skipping invalid cycle period '{period}' in '{category}'.")
                continue

            logger.debug(f"Aligning phase for {category} cycle, period={period}...")
            optimal_phase = align_to_minimums(recent_data, period, len(recent_data), logger)

            frequency = 2 * np.pi / period
            cosine_wave = np.cos(frequency * t_full + optimal_phase)

            base_name = f'pure_cycle_{category}_{period}'
            results[f'{base_name}_cos'] = cosine_wave
            logger.debug(f"Generated wave: {base_name}_cos phase {optimal_phase:.4f}")

            composite_cosine_sum += cosine_wave
            num_waves_summed += 1

    if num_waves_summed > 0:
        results['composite_cosine_sum'] = composite_cosine_sum
        logger.info(f"Calculated composite sum from {num_waves_summed} waves.")
    else:
        logger.warning("No valid cosine waves generated for composite sum.")

    result_df = pd.DataFrame(results, index=price_data.index)
    return result_df


def run_cycle_processor(
    input_df: pd.DataFrame,
    config,
    logger: logging.Logger,
    round_digits: Optional[int] = 4
) -> pd.DataFrame:
    """Main function to run cycle detection and feature generation"""

    logger.info("Running Cycle Processor...")

    temp_df = input_df.copy()

    # Ensure DatetimeIndex
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        if config.DATE_COLUMN in temp_df.columns:
            try:
                temp_df[config.DATE_COLUMN] = pd.to_datetime(temp_df[config.DATE_COLUMN])
                temp_df = temp_df.set_index(config.DATE_COLUMN).sort_index()
                logger.debug(f"Set '{config.DATE_COLUMN}' as index.")
            except Exception as e:
                logger.error(f"Failed to set DatetimeIndex: {e}")
                return pd.DataFrame()
        else:
            logger.error("Input lacks DatetimeIndex or Date column.")
            return pd.DataFrame()

    if config.CYCLE_PRICE_COLUMN not in temp_df.columns:
        logger.error(f"Price column '{config.CYCLE_PRICE_COLUMN}' not found.")
        return pd.DataFrame()

    price_data_series = temp_df[config.CYCLE_PRICE_COLUMN]
    price_data_fft = price_data_series.dropna()

    if price_data_fft.empty:
        logger.error("Price column has no valid data for FFT.")
        return pd.DataFrame()

    # Use static cycles as per original code
    detected_cycles = {'short': [4, 8, 16], 'medium': [174, 180], 'long': [419, 839]}
    logger.info(f"Using STATIC cycles: {detected_cycles}")

    # Generate aligned cosine waves
    try:
        aligned_cycles_df = generate_minimum_aligned_cosine_waves(
            price_data=price_data_series,
            cycles=detected_cycles,
            lookback_window=config.CYCLE_LOOKBACK_WINDOW,
            logger=logger
        )

        if aligned_cycles_df.empty:
            logger.warning("Aligned cycle generation resulted in empty DataFrame.")
            return pd.DataFrame(index=temp_df.index)

        # Optional rounding
        if round_digits is not None:
            try:
                aligned_cycles_df = aligned_cycles_df.round(round_digits)
                logger.debug(f"Rounded cycle features to {round_digits} decimals.")
            except Exception as round_err:
                logger.error(f"Failed to round cycle features: {round_err}")

        logger.info(f"Successfully generated {len(aligned_cycles_df.columns)} cycle features.")
        return aligned_cycles_df

    except Exception as e:
        logger.error(f"Error during aligned cycle wave generation: {e}", exc_info=True)
        return pd.DataFrame()


# ============================================================================
# HMM PROCESSOR
# ============================================================================

class LogNormalHMM(hmm.GaussianHMM):
    """Hidden Markov Model with Log-Normal emission distributions"""

    _lognormal_params = frozenset(['shape_', 'loc_', 'scale_'])

    def __init__(
        self,
        n_components=1,
        shape_prior=1.0,
        loc_prior=0.0,
        scale_prior=1.0,
        startprob_prior=1.0,
        transmat_prior=1.0,
        algorithm="viterbi",
        random_state=None,
        n_iter=100,
        tol=1e-3,
        verbose=False,
        params="stmc",
        init_params="stmc"
    ):
        super().__init__(
            n_components=n_components,
            covariance_type='diag',
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params
        )
        self.shape_ = np.full(n_components, shape_prior, dtype=float)
        self.loc_ = np.full(n_components, loc_prior, dtype=float)
        self.scale_ = np.full(n_components, scale_prior, dtype=float)

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = 1
        return {"s": nc, "t": nc * nc, "m": nc * nf, "c": nc * nf}

    def _init(self, X, lengths=None):
        super()._init(X, lengths=lengths)
        logger = logging.getLogger()

        if any(p in self.init_params for p in ['m', 'c']):
            try:
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X_flat_positive = X[X > 0].flatten()

                if len(X_flat_positive) == 0:
                    logger.warning("No positive data points for lognormal init.")
                    return

                global_shape, global_loc, global_scale = lognorm.fit(X_flat_positive, floc=0)
                rng = np.random.RandomState(self.random_state if isinstance(self.random_state, int) else None)

                self.shape_ = np.maximum(0.01, global_shape * rng.uniform(0.8, 1.2, self.n_components))
                self.loc_ = np.zeros(self.n_components)
                self.scale_ = np.maximum(0.01, global_scale * rng.uniform(0.8, 1.2, self.n_components))

                logger.debug("Initialized lognormal parameters using global fit.")
            except Exception as e:
                logger.warning(f"Initial log-normal parameter fit failed: {e}. Using priors.")

    def _check(self):
        super()._check()
        try:
            for param_name in ['shape_', 'loc_', 'scale_']:
                param = getattr(self, param_name)
                if not isinstance(param, np.ndarray) or param.shape != (self.n_components,):
                    raise ValueError(f"{param_name} shape mismatch")

            if np.any(self.shape_ <= 0):
                raise ValueError("shape_ must be positive")
            if np.any(self.scale_ <= 0):
                raise ValueError("scale_ must be positive")
        except AttributeError as e:
            raise ValueError(f"Parameter check failed: Missing attribute {e}")

    def _compute_log_likelihood(self, X):
        """Computes log probability of observations under each state's emission dist"""
        logger = logging.getLogger()

        temp_X = X
        if temp_X.ndim == 1:
            temp_X = temp_X.reshape(-1, 1)

        if temp_X.shape[0] == 0:
            logger.debug("Input X is empty in _compute_log_likelihood.")
            return np.zeros((0, self.n_components))

        X_flat = temp_X[:, 0]
        log_likelihoods = np.full((len(X_flat), self.n_components), -np.inf)

        for i in range(self.n_components):
            shape, loc, scale = self.shape_[i], self.loc_[i], self.scale_[i]

            if shape > 0 and scale > 0:
                try:
                    valid_mask = X_flat > 0
                    log_pdf_i = np.full_like(X_flat, -np.inf)

                    if np.any(valid_mask):
                        log_pdf_i[valid_mask] = lognorm.logpdf(X_flat[valid_mask], s=shape, loc=loc, scale=scale)

                    log_pdf_i[np.isneginf(log_pdf_i)] = -1e10
                    log_pdf_i[np.isnan(log_pdf_i)] = -1e10
                    log_likelihoods[:, i] = log_pdf_i
                except ValueError as ve:
                    logger.warning(f"ValueError in lognorm.logpdf for state {i}: {ve}")
                    pass

        return log_likelihoods

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        self._m_step_emission(stats['obs'], stats['post'])

    def _m_step_emission(self, X, responsibilities):
        logger = logging.getLogger()

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_flat = X[:, 0]

        if responsibilities.ndim != 2 or responsibilities.shape[1] != self.n_components:
            logger.warning("Responsibilities shape mismatch.")
            return

        for i in range(self.n_components):
            weights = responsibilities[:, i]
            total_weight = np.sum(weights)

            if total_weight > 1e-6:
                try:
                    positive_mask = (X_flat > 0) & (weights > 1e-10)
                    X_positive = X_flat[positive_mask]
                    weights_positive = weights[positive_mask]

                    if len(X_positive) < 2 or np.sum(weights_positive) < 1e-6:
                        logger.debug(f"Skipping M-step state {i}: Not enough weighted positive data.")
                        continue

                    log_X = np.log(X_positive)
                    weighted_log_mean = np.average(log_X, weights=weights_positive)
                    weighted_log_variance = max(
                        1e-10,
                        np.average((log_X - weighted_log_mean)**2, weights=weights_positive)
                    )
                    weighted_log_std = np.sqrt(weighted_log_variance)

                    self.shape_[i] = np.maximum(1e-6, weighted_log_std)
                    self.scale_[i] = np.maximum(1e-6, np.exp(weighted_log_mean))
                    self.loc_[i] = 0.0
                except Exception as e:
                    logger.warning(f"M-step state {i} emission failed: {e}. Keeping old params.")

    def _generate_sample_from_state(self, state, random_state=None):
        logger = logging.getLogger()
        rs = random_state if random_state is not None else np.random.RandomState()
        shape = max(1e-6, self.shape_[state])
        loc = self.loc_[state]
        scale = max(1e-6, self.scale_[state])

        try:
            sample = lognorm.rvs(s=shape, loc=loc, scale=scale, size=1, random_state=rs)[0]
        except Exception as e:
            logger.error(f"Error generating sample state {state}: {e}")
            sample = -1

        if not np.isfinite(sample) or sample <= 0:
            logger.debug(f"Generated invalid sample ({sample}) state {state}. Falling back.")
            sample = lognorm.median(s=shape, loc=loc, scale=scale) if scale > 0 and shape > 0 else 1.0

        return np.array([sample])

    def _get_emission_params(self):
        return (self.shape_, self.loc_, self.scale_)

    def _set_emission_params(self, params):
        self.shape_, self.loc_, self.scale_ = params
        self.shape_ = np.asarray(self.shape_)
        self.loc_ = np.asarray(self.loc_)
        self.scale_ = np.asarray(self.scale_)

    def _needs_lognormal_emission_update(self):
        return any(p in self.params for p in ['m', 'c'])


def train_hmm_model(
    ModelClass,
    data,
    model_name: str,
    n_states: int,
    n_iter: int,
    random_state: int,
    logger: logging.Logger,
    **kwargs
):
    """Helper function to train an HMM model"""

    logger.info(f"Training {model_name} HMM with {n_states} states...")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    params_str = ('ste', 'ste') if ModelClass == hmm.CategoricalHMM else ('stmc', 'stmc')
    params_str, init_params_str = params_str

    if not np.all(np.isfinite(data)):
        logger.warning(f"{np.sum(~np.isfinite(data))} non-finite values in {model_name} data. Replacing with 0.")
        data = np.nan_to_num(data)

    if len(data) == 0:
        logger.error(f"Cannot train {model_name} HMM: Data is empty.")
        return None

    model = ModelClass(
        n_components=n_states,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
        verbose=False,
        params=params_str,
        init_params=init_params_str,
        **kwargs
    )

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.fit(data)

        converged = model.monitor_.converged if hasattr(model, 'monitor_') and model.monitor_ is not None else 'Unknown'
        logger.info(f"{model_name} HMM Training Converged: {converged}")

        if converged is False:
            logger.warning(f"{model_name} HMM did not converge within {n_iter} iterations.")

    except Exception as e:
        logger.error(f"CRITICAL ERROR during {model_name} HMM fitting: {e}", exc_info=True)
        return None

    return model


def generate_hmm_predictions(
    processed_index: pd.Index,
    model,
    data_processed,
    model_name_prefix: str,
    n_states: int,
    n_forward_days: int,
    round_decimals: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """Calculates historical forward state probabilities"""

    logger.info(f"Generating Historical Forward Predictions for {model_name_prefix}...")

    if model is None:
        logger.warning(f"Skipping {model_name_prefix} predictions: Model is None.")
        return pd.DataFrame(index=processed_index)

    output_df = pd.DataFrame(index=processed_index)

    if data_processed.ndim == 1:
        data_processed = data_processed.reshape(-1, 1)

    try:
        state_probs_hist = model.predict_proba(data_processed)
        transmat = model.transmat_

        # Validation and NaN handling
        if np.isnan(state_probs_hist).any():
            logger.warning(f"{np.sum(np.isnan(state_probs_hist).any(axis=1))} rows with NaNs in predict_proba output.")
            state_probs_hist = np.nan_to_num(state_probs_hist, nan=1.0/n_states)
            row_sums = state_probs_hist.sum(axis=1, keepdims=True)
            safe_row_sums = np.where(np.abs(row_sums) > 1e-9, row_sums, 1.0)
            state_probs_hist = np.divide(
                state_probs_hist,
                safe_row_sums,
                out=np.full_like(state_probs_hist, 1.0/n_states),
                where=safe_row_sums!=0
            )

        if transmat is None or np.isnan(transmat).any() or not np.allclose(transmat.sum(axis=1), 1.0):
            logger.error(f"ERROR: Transition matrix invalid for {model_name_prefix}.")
            return pd.DataFrame(index=processed_index)

        # Prepare output columns
        feature_cols = []
        for d in range(1, n_forward_days + 1):
            for s in range(n_states):
                col_name = f'{model_name_prefix}_HMM_D{d}_State{s}_Prob'
                output_df[col_name] = np.nan
                feature_cols.append(col_name)

        # Propagate probabilities
        logger.info(f"Propagating probabilities for {len(processed_index)} time steps...")
        transmat_powers = [matrix_power(transmat, d) for d in range(1, n_forward_days + 1)]
        results_array = np.full((len(processed_index), len(feature_cols)), np.nan)

        for t in range(len(processed_index)):
            current_probs = state_probs_hist[t]
            if np.isnan(current_probs).any():
                continue

            col_idx = 0
            for d_idx, d in enumerate(range(1, n_forward_days + 1)):
                transmat_d = transmat_powers[d_idx]
                pred_probs_d = current_probs @ transmat_d
                pred_sum = pred_probs_d.sum()

                if not np.isfinite(pred_sum) or pred_sum < 1e-9:
                    pred_probs_d = np.full(n_states, 1.0 / n_states)
                else:
                    pred_probs_d /= pred_sum

                for s in range(n_states):
                    results_array[t, col_idx] = np.round(pred_probs_d[s], round_decimals)
                    col_idx += 1

        output_df[feature_cols] = results_array
        logger.info(f"Finished generating {model_name_prefix} predictions.")

    except Exception as e:
        logger.error(f"CRITICAL ERROR DURING {model_name_prefix} PREDICTION: {e}", exc_info=True)
        return pd.DataFrame(index=processed_index)

    return output_df


# Regime mapping
REGIME_MAPPING = {
    (0, 0): 2, (0, 1): 3, (0, 2): 8, (0, 3): 9,
    (1, 0): 1, (1, 1): 4, (1, 2): 5, (1, 3): 10,
    (2, 0): 0, (2, 1): 6, (2, 2): 7, (2, 3): 11,
}
DEFAULT_REGIME_ID = -1


def get_regime_id(vix_state_idx, rsi_state_idx):
    """Determines regime ID using REGIME_MAPPING"""
    logger = logging.getLogger()

    if pd.isna(vix_state_idx) or pd.isna(rsi_state_idx):
        return DEFAULT_REGIME_ID

    try:
        state_tuple = (int(vix_state_idx), int(rsi_state_idx))
    except (ValueError, TypeError):
        logger.debug(f"Could not convert states VIX='{vix_state_idx}', RSI='{rsi_state_idx}' to int tuple.")
        return DEFAULT_REGIME_ID

    return REGIME_MAPPING.get(state_tuple, DEFAULT_REGIME_ID)


def run_hmm_processor(
    input_df: pd.DataFrame,
    config,
    logger: logging.Logger
) -> pd.DataFrame:
    """Main function to run HMM processing"""

    logger.info("Running HMM Processor...")

    if not isinstance(input_df.index, pd.DatetimeIndex):
        logger.error("Input DataFrame must have a DatetimeIndex.")
        return pd.DataFrame()

    df_tech_original_index = input_df.index
    hmm_df = input_df.copy()

    # Data preparation
    logger.debug("Preparing data for HMMs...")

    required_cols = [config.HMM_VIX_COLUMN, config.HMM_STOCH_K_COLUMN]

    if not all(col in hmm_df.columns for col in required_cols):
        logger.error(f"Missing required HMM columns: {required_cols}")
        return pd.DataFrame()

    hmm_df.rename(columns={
        config.HMM_VIX_COLUMN: "VIX",
        config.HMM_STOCH_K_COLUMN: 'StochRSI'
    }, inplace=True)

    # Discretize StochRSI
    bins = [-np.inf, 0.25, 0.50, 0.75, np.inf]
    labels = [0, 1, 2, 3]
    hmm_df['StochRSI_Encoded'] = pd.cut(hmm_df['StochRSI'], bins=bins, labels=labels, right=False)

    initial_rows = len(hmm_df)
    critical_hmm_cols = ['VIX', 'StochRSI_Encoded']
    hmm_df.dropna(subset=critical_hmm_cols, inplace=True)

    if initial_rows > len(hmm_df):
        logger.debug(f"Dropped {initial_rows - len(hmm_df)} rows due to NaNs in HMM critical columns.")

    if hmm_df.empty:
        logger.error("HMM DataFrame empty after NaN drop.")
        return pd.DataFrame()

    try:
        hmm_df['StochRSI_Encoded'] = (
            hmm_df['StochRSI_Encoded'].cat.codes
            if pd.api.types.is_categorical_dtype(hmm_df['StochRSI_Encoded'])
            else hmm_df['StochRSI_Encoded'].astype(int)
        )

        if (hmm_df['StochRSI_Encoded'] == -1).any():
            logger.warning("StochRSI_Encoded contains -1 after conversion.")
    except Exception as e:
        logger.error(f"Failed to convert StochRSI_Encoded to int: {e}")
        return pd.DataFrame()

    vix_data_processed = hmm_df[['VIX']].values
    rsi_data_processed = hmm_df[['StochRSI_Encoded']].values
    processed_index = hmm_df.index

    # Train models
    if config.HMM_VIX_N_STATES != 3 or config.HMM_RSI_N_STATES != 4:
        logger.warning(f"State counts (VIX={config.HMM_VIX_N_STATES}, RSI={config.HMM_RSI_N_STATES}) mismatch REGIME_MAPPING (3, 4).")

    vix_hmm_model = train_hmm_model(
        ModelClass=LogNormalHMM,
        data=vix_data_processed,
        model_name="VIX",
        n_states=config.HMM_VIX_N_STATES,
        n_iter=config.HMM_VIX_N_ITER,
        random_state=config.RANDOM_SEED,
        logger=logger
    )

    rsi_hmm_model = train_hmm_model(
        ModelClass=hmm.CategoricalHMM,
        data=rsi_data_processed,
        model_name="RSI",
        n_states=config.HMM_RSI_N_STATES,
        n_iter=config.HMM_RSI_N_ITER,
        random_state=config.RANDOM_SEED,
        logger=logger
    )

    if vix_hmm_model is None or rsi_hmm_model is None:
        logger.error("HMM model training failed.")
        return pd.DataFrame()

    # Generate predictions
    vix_predictions = generate_hmm_predictions(
        processed_index,
        vix_hmm_model,
        vix_data_processed,
        'VIX',
        config.HMM_VIX_N_STATES,
        config.HMM_N_FORWARD_DAYS,
        config.HMM_ROUND_DECIMALS,
        logger
    )

    rsi_predictions = generate_hmm_predictions(
        processed_index,
        rsi_hmm_model,
        rsi_data_processed,
        'RSI',
        config.HMM_RSI_N_STATES,
        config.HMM_N_FORWARD_DAYS,
        config.HMM_ROUND_DECIMALS,
        logger
    )

    if vix_predictions.empty and rsi_predictions.empty:
        logger.error("HMM prediction generation failed.")
        return pd.DataFrame(index=df_tech_original_index)

    # Combine predictions & assign regime
    hmm_output_processed = pd.merge(vix_predictions, rsi_predictions, left_index=True, right_index=True, how='outer')

    if not hmm_output_processed.empty:
        logger.info("Assigning combined market state regimes...")

        for d in range(1, config.HMM_N_FORWARD_DAYS + 1):
            regime_col_name = f'Regime_ID_D{d}'
            vix_prob_cols = [f'VIX_HMM_D{d}_State{s}_Prob' for s in range(config.HMM_VIX_N_STATES)]
            rsi_prob_cols = [f'RSI_HMM_D{d}_State{s}_Prob' for s in range(config.HMM_RSI_N_STATES)]

            if not all(col in hmm_output_processed.columns for col in vix_prob_cols + rsi_prob_cols):
                logger.warning(f"Missing probability columns for Day {d}. Assigning default Regime ID.")
                hmm_output_processed[regime_col_name] = DEFAULT_REGIME_ID
                continue

            try:
                most_likely_vix_state = hmm_output_processed[vix_prob_cols].idxmax(axis=1).str.extract(r'State(\d+)').astype(float).iloc[:, 0]
                most_likely_rsi_state = hmm_output_processed[rsi_prob_cols].idxmax(axis=1).str.extract(r'State(\d+)').astype(float).iloc[:, 0]
                hmm_output_processed[regime_col_name] = np.vectorize(get_regime_id)(
                    most_likely_vix_state,
                    most_likely_rsi_state
                ).astype(int)
            except Exception as e:
                logger.error(f"Error calculating Regime ID Day {d}: {e}. Assigning default.")
                hmm_output_processed[regime_col_name] = DEFAULT_REGIME_ID

        logger.info("Finished assigning market state regimes.")
    else:
        logger.warning("Merged HMM predictions empty. Cannot assign regimes.")

    # Reindex and reorder
    logger.debug("Reindexing final HMM results to original index...")
    final_output = hmm_output_processed.reindex(df_tech_original_index)
    logger.debug(f"Final HMM output shape: {final_output.shape}")

    if not final_output.empty:
        try:
            all_cols = final_output.columns.tolist()
            regime_cols = sorted([f'Regime_ID_D{d}' for d in range(1, config.HMM_N_FORWARD_DAYS + 1) if f'Regime_ID_D{d}' in all_cols])
            prob_cols = sorted([col for col in all_cols if col not in regime_cols])
            new_col_order = regime_cols + prob_cols
            final_output = final_output[new_col_order]
        except Exception as e:
            logger.warning(f"Could not reorder HMM columns: {e}.")

    logger.info("HMM Processor finished.")
    return final_output


# ============================================================================
# MANDELBROT PROCESSOR
# ============================================================================

def calculate_mandelbrot_coordinates(
    price_series: pd.Series,
    volume_series: pd.Series,
    max_iter: int,
    logger: logging.Logger
) -> Tuple[pd.Series, pd.Series]:
    """Calculates Mandelbrot-inspired X and Y coordinates"""

    logger.info("Calculating Mandelbrot coordinates...")

    combined = pd.DataFrame({'price': price_series, 'volume': volume_series}).dropna()

    if combined.empty:
        logger.warning("Mandelbrot: Input empty after NaN drop.")
        original_index = price_series.index.union(volume_series.index)
        return pd.Series(dtype=float, index=original_index), pd.Series(dtype=float, index=original_index)

    price_clean = combined['price']
    volume_clean = combined['volume']
    output_index = combined.index

    # Normalize [0, 1]
    price_range = price_clean.max() - price_clean.min()
    volume_range = volume_clean.max() - volume_clean.min()

    price_norm = (
        (price_clean - price_clean.min()) / price_range
        if price_range > 1e-9
        else np.zeros_like(price_clean)
    )
    volume_norm = (
        (volume_clean - volume_clean.min()) / volume_range
        if volume_range > 1e-9
        else np.zeros_like(volume_clean)
    )

    price_norm_np = price_norm.values
    volume_norm_np = volume_norm.values

    mandelbrot_x_vals = np.zeros(len(combined))
    mandelbrot_y_vals = np.zeros(len(combined))

    # Mandelbrot calculation loop
    for i in range(len(combined)):
        c_real = -1.5 + 2.0 * price_norm_np[i]
        c_imag = -1.0 + 2.0 * volume_norm_np[i]

        z_real, z_imag = 0.0, 0.0
        iteration = 0

        while (z_real*z_real + z_imag*z_imag < 4.0) and (iteration < max_iter):
            z_real_temp = z_real*z_real - z_imag*z_imag + c_real
            z_imag = 2*z_real*z_imag + c_imag
            z_real = z_real_temp
            iteration += 1

        escape_ratio = iteration / max_iter
        mandelbrot_x_vals[i] = z_real * escape_ratio
        mandelbrot_y_vals[i] = z_imag * escape_ratio

    logger.info("Mandelbrot coordinates calculated.")

    mandelbrot_x_series = pd.Series(mandelbrot_x_vals, index=output_index)
    mandelbrot_y_series = pd.Series(mandelbrot_y_vals, index=output_index)

    return mandelbrot_x_series, mandelbrot_y_series


def run_mandelbrot_processor(
    input_df: pd.DataFrame,
    config,
    logger: logging.Logger
) -> pd.DataFrame:
    """Orchestrates the Mandelbrot indicator calculation process"""

    logger.info("Running Mandelbrot Processor...")

    if not isinstance(input_df.index, pd.DatetimeIndex):
        logger.error("Input DataFrame must have a DatetimeIndex.")
        return pd.DataFrame()

    mandel_df = input_df

    required_cols = [config.MANDELBROT_PRICE_COLUMN, config.MANDELBROT_VOLUME_COLUMN]

    if not all(col in mandel_df.columns for col in required_cols):
        logger.error(f"Missing required Mandelbrot columns: {required_cols}")
        return pd.DataFrame()

    # Calculate coordinates
    try:
        mandelbrot_x, mandelbrot_y = calculate_mandelbrot_coordinates(
            mandel_df[config.MANDELBROT_PRICE_COLUMN],
            mandel_df[config.MANDELBROT_VOLUME_COLUMN],
            config.MANDELBROT_MAX_ITER,
            logger
        )
    except Exception as e:
        logger.error(f"Error during Mandelbrot calculation: {e}", exc_info=True)
        return pd.DataFrame()

    # Prepare output
    output_df = pd.DataFrame({
        'Mandelbrot_X': mandelbrot_x,
        'Mandelbrot_Y': mandelbrot_y
    })

    logger.debug(f"Mandelbrot processing finished. Shape: {output_df.shape}")

    final_output = output_df.reindex(input_df.index)

    # Optional rounding
    if config.MANDELBROT_ROUND_DECIMALS is not None and not final_output.empty:
        try:
            final_output = final_output.round(config.MANDELBROT_ROUND_DECIMALS)
            logger.debug(f"Rounded Mandelbrot features to {config.MANDELBROT_ROUND_DECIMALS} decimals.")
        except Exception as round_err:
            logger.error(f"Failed to round Mandelbrot features: {round_err}")

    logger.info("Mandelbrot Processor finished.")
    return final_output
