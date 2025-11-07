# Data Pipeline for Nvidia DGX

Refactored data processing pipeline for SPY/VIX market data analysis, optimized for Nvidia DGX systems.

## Overview

This pipeline performs comprehensive market data acquisition and feature engineering:

- **Data Acquisition**: Downloads SPY, VIX, and other market data via yfinance
- **Technical Analysis**: Calculates 40+ technical indicators using TA-Lib
- **Cycle Detection**: FFT-based cycle analysis with phase-aligned cosine waves
- **HMM Analysis**: Hidden Markov Models for VIX and Stochastic RSI regime detection
- **Mandelbrot Features**: Fractal-inspired coordinate calculations
- **Feature Engineering**: Lag generation, target calculation, and data cleaning

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: Minimum 16GB recommended
- **Storage**: ~10GB for data and logs
- **Network**: Internet connection for data downloads

## Installation

### 1. Install System Dependencies

First, install TA-Lib C library and development tools:

```bash
# Update package list
sudo apt-get update

# Install TA-Lib dependencies
sudo apt-get install -y build-essential wget

# Download and install TA-Lib
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig

# Clean up
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

### 2. Create Python Virtual Environment

```bash
# Navigate to project directory
cd /path/to/DataLoader

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test TA-Lib import
python -c "import talib; print('TA-Lib version:', talib.__version__)"

# Test other imports
python -c "import pandas, numpy, yfinance, hmmlearn; print('All imports successful')"
```

## Configuration

### Environment Variables

You can set the output directory via environment variable:

```bash
export DATA_PIPELINE_OUTPUT_DIR=/data/afterford/output
```

### Directory Structure

The pipeline creates the following directory structure:

```
$DATA_PIPELINE_OUTPUT_DIR/
├── data/
│   ├── raw/              # Raw downloaded data
│   │   ├── data_raw.csv  # SPY/VIX raw data
│   │   └── market_raw.csv # Other market data
│   ├── processed/        # Processed unlagged data
│   │   └── processed_data.csv
│   └── data.csv          # Final lagged dataset
│   └── data_etf.csv      # ETF data (SPXS, SPXL)
├── logs/                 # Pipeline execution logs
├── models/               # (Reserved for future use)
├── visualizations/       # (Reserved for future use)
└── temp/                 # Temporary files
```

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run pipeline with defaults
python pipeline_runner.py
```

### Advanced Usage

```bash
# Specify custom output directory
python pipeline_runner.py --output-dir /custom/path/output

# Enable debug logging
python pipeline_runner.py --log-level DEBUG

# Both options
python pipeline_runner.py --output-dir /custom/path --log-level INFO
```

### Command-Line Options

- `--output-dir PATH`: Base output directory (default: `/data/afterford/output`)
- `--log-level LEVEL`: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

## Pipeline Stages

### Stage 0: Directory Setup
Creates all necessary output directories.

### Stage 1: Data Acquisition
- Downloads SPY and VIX data (1993-present)
- Downloads market indicators (DJI, RUT, FTSE, N225, bonds, commodities, etc.)
- Processes and saves raw data files

### Stage 1.5: ETF Data Download
- Downloads SPXS and SPXL ETF data
- Aligns with main dataset dates

### Stage 2: Target & MOE Calculation
- Calculates target variable (5-day forward return)
- Calculates Movement Outcome Encoding (MOE: -1, 0, 1)
- Applies Ehlers Super Smoother to weighted close price

### Stage 3: Feature Processing
- **Technical Analyzer**: 40+ TA-Lib indicators (MACD, RSI, Stochastic, etc.)
- **Cycle Processor**: Phase-aligned cosine waves for market cycles
- **HMM Processor**: Hidden Markov Models for regime detection
- **Mandelbrot Processor**: Fractal-inspired coordinate features

### Stage 4: Market MACD Calculation
- Calculates MACD histograms for all market symbols
- Unlagged features for use in lag generation

### Stage 5: Save Processed Data
- Merges all features
- Saves intermediate processed dataset (unlagged)

### Stages 6-7: Lag Generation
- Generates lags 0-15 for all calculated features
- Preserves SPY OHLCV, Target, and MOE in original timeframe

### Stage 8: Final Cleaning & Save
- Drops initial rows to remove NaN artifacts
- Forward/backward fills missing values
- Rounds numeric features
- Saves final lagged dataset

## Output Files

### Main Outputs

1. **data.csv** - Final lagged dataset with all features
   - All calculated indicators with 0-15 lags
   - SPY OHLCV columns (unlagged)
   - Target and MOE columns
   - ~1000+ columns depending on market data availability

2. **data_etf.csv** - ETF OHLCV data
   - SPXS and SPXL data
   - Aligned dates with main dataset

### Intermediate Files

3. **processed_data.csv** - Unlagged features before lag generation
4. **data_raw.csv** - Raw SPY/VIX data
5. **market_raw.csv** - Raw market indicator data

### Logs

- Located in `$OUTPUT_DIR/logs/`
- Named with timestamp: `pipeline_YYYYMMDD_HHMMSS.log`
- Contains detailed execution information, warnings, and errors

## Features Generated

### Technical Indicators
- Hilbert Transform (sine, lead_sine, cycle, trendmode, trendline)
- MAMA/FAMA
- Volume indicators (OBV, Chaikin A/D, MFI)
- Momentum indicators (Williams %R, Stochastic RSI, MACD)
- Directional Movement Index (+DI, -DI)
- Ichimoku Cloud (tenkan_sen, kijun_sen, senkou spans, chikou_span)
- Parabolic SAR

### Cycle Features
- Phase-aligned cosine waves for short (4, 8, 16), medium (174, 180), and long (419, 839) cycles
- Composite cosine sum

### HMM Features
- VIX regime probabilities (3 states) for 1-5 days forward
- Stochastic RSI regime probabilities (4 states) for 1-5 days forward
- Combined regime IDs (12 total regimes)

### Mandelbrot Features
- Mandelbrot_X and Mandelbrot_Y coordinates

### Market Features
- MACD histograms for all downloaded market symbols

## Configuration Parameters

Key parameters can be modified in `data_pipeline.py` in the `Config` class:

### Target Calculation
- `TARGET_LOOKAHEAD_DAYS`: 5 (days ahead for target calculation)
- `TARGET_MOE_THRESHOLD`: 5.0 (percentage threshold for MOE)

### Lag Generation
- `PIPELINE_MAX_LAG`: 15 (maximum number of lags)

### HMM Settings
- `HMM_VIX_N_STATES`: 3
- `HMM_RSI_N_STATES`: 4
- `HMM_N_FORWARD_DAYS`: 5
- `HMM_VIX_N_ITER`: 150 (training iterations)

### Cycle Detection
- `CYCLE_LOOKBACK_WINDOW`: 2520 (trading days for phase alignment)

### Data Cleaning
- `PIPELINE_DROP_INITIAL_ROWS`: 100 (rows to drop at start)
- `PIPELINE_ROUND_DECIMALS`: 4

## Troubleshooting

### TA-Lib Import Error

If you get `ImportError: libtalib.so.0: cannot open shared object file`:

```bash
sudo ldconfig
# Or specify library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Rate Limit Errors from yfinance

The pipeline includes automatic delays between downloads (default 5 seconds). If you still encounter rate limits:

1. Increase `DOWNLOAD_DELAY` in `Config` class
2. Run during off-peak hours
3. Consider splitting symbol downloads

### Memory Issues

For large datasets:

1. Increase available RAM
2. Reduce `CYCLE_LOOKBACK_WINDOW`
3. Reduce `PIPELINE_MAX_LAG`
4. Process symbols in batches

### Missing Data Warnings

Some market symbols may fail to download:
- Check symbol availability on Yahoo Finance
- Verify internet connection
- Review logs for specific error messages

## Performance Considerations

### Execution Time

Typical execution time on DGX system:
- Data download: 5-15 minutes (depends on network and rate limiting)
- Feature processing: 5-10 minutes
- Total: ~15-30 minutes for 20+ years of data

### Optimization Tips

1. **Parallel Processing**: The pipeline runs sequentially; consider parallelizing independent processors
2. **Caching**: Raw data files are saved and can be reused
3. **Incremental Updates**: Modify to only download new data after initial run
4. **GPU Acceleration**: Some operations (especially HMM training) could benefit from GPU

## Module Structure

```
DataLoader/
├── data_pipeline.py       # Core classes and utilities
├── processors.py          # All feature processors
├── pipeline_runner.py     # Main execution logic
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Contributing

When modifying the pipeline:

1. **Config Changes**: Update the `Config` class in `data_pipeline.py`
2. **New Processors**: Add to `processors.py` following existing patterns
3. **Pipeline Logic**: Modify `pipeline_runner.py`
4. **Testing**: Test with small date ranges first
5. **Logging**: Use appropriate log levels (DEBUG for development, INFO for production)

## Known Limitations

1. **Data Availability**: Limited by Yahoo Finance data quality and availability
2. **HMM Convergence**: Models may not always converge within iteration limits
3. **Memory Usage**: Large feature sets require significant RAM
4. **Sequential Processing**: No multi-threading currently implemented

## Future Enhancements

- [ ] Multi-threaded/GPU-accelerated processing
- [ ] Incremental data updates
- [ ] Additional data sources beyond yfinance
- [ ] Model persistence for HMMs
- [ ] Visualization generation
- [ ] Real-time data streaming
- [ ] Configurable processor selection

## License

[Specify your license]

## Contact

[Your contact information]

## References

- TA-Lib: https://ta-lib.org/
- yfinance: https://github.com/ranaroussi/yfinance
- hmmlearn: https://hmmlearn.readthedocs.io/

---

**Original Colab Notebook**: Afterford_Load_Data.ipynb
**Refactored**: 2025-11-07
**Target Platform**: Nvidia DGX Systems
