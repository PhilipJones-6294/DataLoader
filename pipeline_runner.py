#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
                    Main Pipeline Runner
==============================================================================

Main execution logic for the data pipeline.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import talib

from data_pipeline import (
    Config,
    setup_logging,
    setup_directories,
    DataAcquisitionService,
    FeatureEngineeringService,
    ehlers_super_smoother,
    generate_lags,
    download_etf_data
)

from processors import (
    run_technical_analyzer,
    run_cycle_processor,
    run_hmm_processor,
    run_mandelbrot_processor
)


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_pipeline(config: Config, logger: logging.Logger):
    """Executes the full data processing pipeline"""

    logger.info("=" * 80)
    logger.info("Starting Data Pipeline Execution on Nvidia DGX")
    logger.info("=" * 80)

    run_start_time = pd.Timestamp.now()

    # Initialize DataFrames
    df_raw_spy_vix = None
    df_market_raw = None
    df_processed_unlagged = None
    df_final_lagged = None
    df_etf_final = None

    try:
        # --- Stage 0: Setup Directories ---
        logger.info("Pipeline Stage 0: Setting up directories...")
        setup_directories(config.CORE_DIRECTORIES)

        # --- Stage 1: Data Acquisition ---
        logger.info("Pipeline Stage 1: Running Data Acquisition...")
        try:
            acquisition_service = DataAcquisitionService(config, logger)
            df_raw_spy_vix, df_market_raw = acquisition_service.download_market_data(include_futures=True)

            if df_raw_spy_vix is None or df_raw_spy_vix.empty:
                logger.critical("Data acquisition failed: Raw SPY/VIX data not generated.")
                return

            logger.info(f"Raw data acquisition successful. SPY/VIX shape: {df_raw_spy_vix.shape}")

            if df_market_raw is not None and not df_market_raw.empty:
                logger.info(f"Market Raw shape: {df_market_raw.shape}")
            else:
                logger.info("Market Raw data is empty or was not generated.")

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during Data Acquisition: {e}", exc_info=True)
            return

        # --- Stage 1.5: Download ETF Data ---
        logger.info("Pipeline Stage 1.5: Downloading ETF Data (SPXS, SPXL)...")
        if df_raw_spy_vix is not None and not df_raw_spy_vix.empty:
            start_date_str = df_raw_spy_vix.index.min().strftime('%Y-%m-%d')
            end_date_dt = df_raw_spy_vix.index.max() + pd.Timedelta(days=1)
            end_date_str = end_date_dt.strftime('%Y-%m-%d')

            try:
                df_etf_raw = download_etf_data(
                    config.ETF_SYMBOLS,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    round_digits=2,
                    download_delay=config.DOWNLOAD_DELAY,
                    logger=logger
                )

                if df_etf_raw is not None and not df_etf_raw.empty:
                    common_index = df_raw_spy_vix.index.intersection(df_etf_raw.index)
                    df_etf_final = df_etf_raw.loc[common_index]

                    if not df_etf_final.empty:
                        df_etf_final.to_csv(config.ETF_DATA_FILE_PATH, index=True)
                        logger.info(f"ETF data saved to: {config.ETF_DATA_FILE_PATH}")
                        logger.info(f"ETF data shape: {df_etf_final.shape}")
                    else:
                        logger.warning("ETF data empty after aligning index.")
                else:
                    logger.warning(f"Failed to download ETF data.")

            except Exception as etf_err:
                logger.error(f"Error during ETF data processing: {etf_err}", exc_info=True)
        else:
            logger.warning("Skipping ETF data download because main data is missing.")

        # --- Stages 2-5: Feature Processing ---
        processed_base = None
        macd_hist_df = pd.DataFrame()

        try:
            # --- Stage 2: Calculate Target and MOE ---
            logger.info("Pipeline Stage 2: Calculating Target and MOE...")

            if not all(col in df_raw_spy_vix.columns for col in ['SPY_High', 'SPY_Low', 'SPY_Close']):
                raise ValueError("Missing required SPY columns for Target calculation.")

            # Calculate Weighted Close Price (WCL)
            wcl = ehlers_super_smoother(
                talib.WCLPRICE(
                    df_raw_spy_vix['SPY_High'],
                    df_raw_spy_vix['SPY_Low'],
                    df_raw_spy_vix['SPY_Close']
                )
            )

            # Calculate future return for Target
            future_spy_close = df_raw_spy_vix['SPY_Close'].shift(-config.TARGET_LOOKAHEAD_DAYS)

            # Calculate Target percentage return
            target = np.where(
                (wcl != 0) & (~wcl.isna()) & (~future_spy_close.isna()),
                (future_spy_close - wcl) / wcl * 100,
                np.nan
            )
            target_series = pd.Series(
                target,
                index=df_raw_spy_vix.index,
                name=config.TARGET_COLUMN
            ).round(config.TARGET_ROUND_DECIMALS)

            # Calculate MOE
            conditions = [
                target_series > config.TARGET_MOE_THRESHOLD,
                target_series < -config.TARGET_MOE_THRESHOLD
            ]
            outcomes = [1, -1]
            default_outcome = 0

            moe_series = pd.Series(
                np.select(conditions, outcomes, default=default_outcome),
                index=target_series.index,
                name=config.MOE_COLUMN,
                dtype=np.int8
            )

            # Combine
            df_for_processors = pd.concat([
                target_series,
                moe_series,
                df_raw_spy_vix,
                wcl.rename('wcl')
            ], axis=1)

            logger.info(f"Target and MOE calculated. Shape: {df_for_processors.shape}")

            # --- Stage 3: Run Feature Processors ---
            logger.info("Pipeline Stage 3: Running Feature Processors...")

            # Run Technical Analyzer
            feature_eng = FeatureEngineeringService(logger)
            df_tech = run_technical_analyzer(
                df_for_processors.copy(),
                feature_eng,
                logger,
                round_digits=None
            )

            if df_tech is None or df_tech.empty:
                raise ValueError("Technical Analyzer failed or returned empty.")

            logger.info(f"TA finished. Shape: {df_tech.shape}")

            # Run Cycle Processor
            df_cycles = run_cycle_processor(df_tech.copy(), config, logger, round_digits=config.CYCLE_ROUND_DECIMALS)

            if df_cycles is None:
                raise ValueError("Cycle Processor failed.")

            processed_after_cycle = pd.merge(
                df_tech,
                df_cycles,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_cycle')
            )
            logger.info(f"Cycle Processor finished. Shape: {processed_after_cycle.shape}")

            # Run HMM Processor
            if config.HMM_STOCH_K_COLUMN not in processed_after_cycle.columns:
                logger.warning(f"HMM required column '{config.HMM_STOCH_K_COLUMN}' not found after TA.")

            df_hmm = run_hmm_processor(processed_after_cycle.copy(), config, logger)

            if df_hmm is None:
                raise ValueError("HMM Processor failed.")

            processed_after_hmm = pd.merge(
                processed_after_cycle,
                df_hmm,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_hmm')
            )
            logger.info(f"HMM Processor finished. Shape: {processed_after_hmm.shape}")

            # Run Mandelbrot Processor
            df_mandelbrot = run_mandelbrot_processor(processed_after_hmm.copy(), config, logger)

            if df_mandelbrot is None:
                raise ValueError("Mandelbrot Processor failed.")

            processed_base = pd.merge(
                processed_after_hmm,
                df_mandelbrot,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_mandel')
            )
            logger.info(f"Mandelbrot Processor finished. Shape: {processed_base.shape}")

            if processed_base.empty:
                raise ValueError("Processing resulted in an empty DataFrame.")

            # --- Stage 4: Process Market Data (MACD) ---
            logger.info("Pipeline Stage 4: Calculating Unlagged MACD Histograms...")

            all_macd_hist = {}
            macd_cols_processed = 0

            if df_market_raw is not None and not df_market_raw.empty:
                min_periods_needed = config.MACD_SLOW_PERIOD + config.MACD_SIGNAL_PERIOD - 1

                if not isinstance(df_market_raw.index, pd.DatetimeIndex):
                    if config.DATE_COLUMN in df_market_raw.columns:
                        df_market_raw = df_market_raw.set_index(
                            pd.to_datetime(df_market_raw[config.DATE_COLUMN])
                        ).drop(columns=[config.DATE_COLUMN])
                    else:
                        logger.warning("Market raw data lacks DatetimeIndex, cannot calculate MACD.")
                        df_market_raw = None

                if df_market_raw is not None:
                    for col in df_market_raw.columns:
                        if col == config.DATE_COLUMN:
                            continue

                        price_series = df_market_raw[col].dropna().astype(float)

                        if len(price_series) < min_periods_needed:
                            continue

                        try:
                            macd, signal, hist = talib.MACD(
                                price_series,
                                fastperiod=config.MACD_FAST_PERIOD,
                                slowperiod=config.MACD_SLOW_PERIOD,
                                signalperiod=config.MACD_SIGNAL_PERIOD
                            )

                            prefix = col.replace('Close', '').replace('^', '').replace('=F', 'F').replace('=X', 'X').replace('=','')
                            hist_col_name = f'{prefix}_Hist'
                            all_macd_hist[hist_col_name] = hist
                            macd_cols_processed += 1

                        except Exception as macd_err:
                            logger.error(f"Error calculating MACD for '{col}': {macd_err}")

                    if all_macd_hist:
                        macd_hist_df = pd.concat(all_macd_hist, axis=1).sort_index()
                        logger.info(f"MACD histograms calculated for {macd_cols_processed} symbols. Shape: {macd_hist_df.shape}")
                    else:
                        logger.warning("No MACD histograms were calculated.")
            else:
                logger.warning("Market raw data empty, skipping MACD calculation.")

            # Align MACD with processed_base
            if not macd_hist_df.empty:
                common_macd_index = processed_base.index.intersection(macd_hist_df.index)
                macd_hist_df = macd_hist_df.loc[common_macd_index]
                processed_base = processed_base.loc[common_macd_index]
                logger.info(f"Aligned indices ({len(common_macd_index)} rows).")

            # --- Stage 5: Save Processed (Unlagged) File ---
            logger.info("Pipeline Stage 5: Saving Processed (Unlagged) Data...")

            if not macd_hist_df.empty:
                df_processed_unlagged = pd.merge(
                    processed_base,
                    macd_hist_df,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
            else:
                df_processed_unlagged = processed_base.copy()

            # Drop intermediate columns
            cols_to_drop_proc = [col for col in config.FEATURE_DROP_COLS if col in df_processed_unlagged.columns]
            if cols_to_drop_proc:
                df_processed_unlagged = df_processed_unlagged.drop(columns=cols_to_drop_proc)
                logger.info(f"Dropped intermediate columns: {cols_to_drop_proc}")

            # Save
            if not df_processed_unlagged.empty:
                df_processed_unlagged.to_csv(config.PROCESSED_DATA_FILE_PATH, index=True)
                logger.info(f"Processed (unlagged) data saved to: {config.PROCESSED_DATA_FILE_PATH}")
                logger.info(f"Processed data shape: {df_processed_unlagged.shape}")
            else:
                logger.error("Processed DataFrame empty. Skipping save.")
                raise ValueError("Processed data became empty.")

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during Stages 2-5: {e}", exc_info=True)
            return

        # --- Stages 6 & 7: Lag Generation and Final Merge ---
        logger.info("Pipeline Stages 6 & 7: Generating Lags and Merging...")

        try:
            # Generate lags for MACD
            lagged_macd_hist_wide = pd.DataFrame(index=df_processed_unlagged.index)

            if not macd_hist_df.empty:
                lagged_macd_hist_wide = generate_lags(
                    macd_hist_df,
                    macd_hist_df.columns.tolist(),
                    config.PIPELINE_MAX_LAG,
                    logger
                )

                if not lagged_macd_hist_wide.empty:
                    logger.info(f"Lagged MACD features generated. Shape: {lagged_macd_hist_wide.shape}")
                else:
                    logger.warning("Lagged MACD generation resulted in empty DataFrame.")

            # Separate unlagged columns
            cols_not_to_lag_exist = [
                col for col in config.PIPELINE_COLS_NOT_TO_LAG
                if col in df_processed_unlagged.columns
            ]

            missing_unlagged_cols = set(config.PIPELINE_COLS_NOT_TO_LAG) - set(cols_not_to_lag_exist)
            if missing_unlagged_cols:
                logger.warning(f"Specified unlagged columns missing: {missing_unlagged_cols}.")

            df_unlagged_final_cols = (
                df_processed_unlagged[cols_not_to_lag_exist].copy()
                if cols_not_to_lag_exist
                else pd.DataFrame(index=df_processed_unlagged.index)
            )

            # Identify base columns to lag
            base_cols_to_lag_names = df_processed_unlagged.columns.difference(cols_not_to_lag_exist).tolist()
            base_cols_to_lag_names = [
                col for col in base_cols_to_lag_names
                if col not in config.FEATURE_DROP_COLS
            ]

            lagged_base_features_wide = pd.DataFrame(index=df_processed_unlagged.index)

            if base_cols_to_lag_names:
                lagged_base_features_wide = generate_lags(
                    df_processed_unlagged,
                    base_cols_to_lag_names,
                    config.PIPELINE_MAX_LAG,
                    logger
                )

                if not lagged_base_features_wide.empty:
                    logger.info(f"Lagged base features generated. Shape: {lagged_base_features_wide.shape}")
                else:
                    logger.warning("Lagged base feature generation resulted in empty DataFrame.")
            else:
                logger.warning("No base columns identified for lagging.")

            # Final merge
            df_final_lagged = df_unlagged_final_cols

            if not lagged_base_features_wide.empty:
                df_final_lagged = pd.merge(
                    df_final_lagged,
                    lagged_base_features_wide,
                    left_index=True,
                    right_index=True,
                    how='outer'
                )
            else:
                logger.warning("Lagged base features empty, skipping merge.")

            if not lagged_macd_hist_wide.empty:
                df_final_lagged = pd.merge(
                    df_final_lagged,
                    lagged_macd_hist_wide,
                    left_index=True,
                    right_index=True,
                    how='outer'
                )
            else:
                logger.warning("Lagged MACD features empty, skipping merge.")

            if df_final_lagged.empty:
                logger.critical("Final lagged DataFrame empty after merging.")
                return

            logger.info(f"Final merge complete. Shape before cleaning: {df_final_lagged.shape}")

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during lag generation: {e}", exc_info=True)
            return

        # --- Stage 8: Clean and Save Final Data ---
        logger.info("Pipeline Stage 8: Cleaning and Saving Final Data...")

        try:
            initial_total_rows = len(df_final_lagged)

            # Drop initial rows
            rows_to_drop = max(config.PIPELINE_DROP_INITIAL_ROWS, config.PIPELINE_MAX_LAG)

            if initial_total_rows > rows_to_drop:
                df_final_lagged = df_final_lagged.iloc[rows_to_drop:]
                logger.info(f"Dropped {rows_to_drop} initial rows. Remaining: {len(df_final_lagged)}")
            else:
                logger.warning(f"Final DataFrame has <= {rows_to_drop} rows.")

            # Fill NaNs (except Target)
            if config.TARGET_COLUMN in df_final_lagged.columns:
                cols_to_fill = df_final_lagged.columns.difference([config.TARGET_COLUMN])

                if not cols_to_fill.empty:
                    df_final_lagged[cols_to_fill] = df_final_lagged[cols_to_fill].ffill().bfill()
                    nan_count_others = df_final_lagged[cols_to_fill].isnull().sum().sum()

                    if nan_count_others > 0:
                        logger.warning(f"{nan_count_others} NaNs remain in non-Target columns.")

                # Note: NOT dropping NaN Target rows to preserve all data
                nan_count_target = df_final_lagged[config.TARGET_COLUMN].isnull().sum()
                if nan_count_target > 0:
                    logger.info(f"{nan_count_target} NaNs found in '{config.TARGET_COLUMN}' (expected due to lookahead).")

            else:
                logger.warning(f"'{config.TARGET_COLUMN}' not found. Applying global fill.")
                df_final_lagged = df_final_lagged.ffill().bfill().dropna()

                if df_final_lagged.isnull().sum().sum() > 0:
                    logger.warning("NaNs remain after global fill/dropna.")

            # Final rounding
            if config.PIPELINE_ROUND_DECIMALS is not None:
                try:
                    numeric_cols = df_final_lagged.select_dtypes(include=np.number).columns

                    cols_to_exclude_rounding = [config.TARGET_COLUMN] + [
                        col for col in df_final_lagged.columns
                        if pd.api.types.is_integer_dtype(df_final_lagged[col])
                    ]

                    numeric_cols_to_round = numeric_cols.difference(cols_to_exclude_rounding)

                    if not numeric_cols_to_round.empty:
                        df_final_lagged[numeric_cols_to_round] = df_final_lagged[numeric_cols_to_round].round(
                            config.PIPELINE_ROUND_DECIMALS
                        )
                        logger.info(f"Applied rounding ({config.PIPELINE_ROUND_DECIMALS} decimals).")

                except Exception as e:
                    logger.error(f"Error during final rounding: {e}")

            # Drop any remaining intermediate columns
            final_cols_to_drop = [
                col for col in config.FEATURE_DROP_COLS
                if col in df_final_lagged.columns
            ]

            if final_cols_to_drop:
                df_final_lagged = df_final_lagged.drop(columns=final_cols_to_drop)
                logger.info(f"Dropped final columns: {final_cols_to_drop}")

            # Save final data
            if not df_final_lagged.empty:
                df_final_lagged.to_csv(config.FINAL_DATA_FILE_PATH, index=True)
                logger.info("=" * 80)
                logger.info("Final (lagged) data saved successfully")
                logger.info("=" * 80)
                logger.info(f"Saved to: {config.FINAL_DATA_FILE_PATH}")
                logger.info(f"Final Data Shape: {df_final_lagged.shape}")
                logger.info(f"Final Data Sample (last 5 rows):")
                logger.info(f"\n{df_final_lagged.tail()}")
            else:
                logger.critical("Final lagged DataFrame empty. No file written.")

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during final cleaning/saving: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"UNHANDLED CRITICAL ERROR in Pipeline: {e}", exc_info=True)

    finally:
        # Log final status
        run_end_time = pd.Timestamp.now()
        duration = run_end_time - run_start_time

        final_main_exists = os.path.exists(config.FINAL_DATA_FILE_PATH)
        final_etf_exists = os.path.exists(config.ETF_DATA_FILE_PATH)

        logger.info("=" * 80)
        logger.info("Pipeline Run Summary")
        logger.info("=" * 80)

        if final_main_exists and df_final_lagged is not None and not df_final_lagged.empty:
            logger.info("Main Pipeline: SUCCESS")
            logger.info(f"Main Output: {config.FINAL_DATA_FILE_PATH}")
        else:
            logger.error("Main Pipeline: FAILED")
            if not final_main_exists:
                logger.error(f"Main output file NOT found: {config.FINAL_DATA_FILE_PATH}")
            elif df_final_lagged is None:
                logger.error("Main final DataFrame is None.")
            else:
                logger.error("Main final DataFrame was empty.")

        if final_etf_exists and df_etf_final is not None and not df_etf_final.empty:
            logger.info("ETF Data Processing: SUCCESS")
            logger.info(f"ETF Output: {config.ETF_DATA_FILE_PATH}")
        elif df_etf_final is None and not final_etf_exists:
            logger.warning("ETF data processing was skipped or failed.")
        else:
            logger.error("ETF Data Processing: FAILED")

        logger.info(f"Total execution time: {duration}")
        logger.info("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the pipeline"""

    import argparse

    parser = argparse.ArgumentParser(
        description='Data Pipeline for SPY/VIX Market Data Processing'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Base output directory (default: /data/afterford/output or DATA_PIPELINE_OUTPUT_DIR env var)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Create configuration
    config = Config(base_output_dir=args.output_dir)

    # Setup logging
    logger = setup_logging(config.CORE_DIRECTORIES['logs'], log_level=args.log_level)

    logger.info("Data Pipeline Starting...")
    logger.info(f"Output Directory: {config.BASE_OUTPUT_DIR}")
    logger.info(f"Log Level: {args.log_level}")

    # Run pipeline
    try:
        run_pipeline(config, logger)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Pipeline execution complete.")


if __name__ == '__main__':
    main()
