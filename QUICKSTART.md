# Quick Start Guide

Get the pipeline running in 5 minutes!

## Prerequisites

- Linux system (Ubuntu 20.04+ recommended)
- Python 3.8+
- Internet connection
- ~10GB free disk space

## Installation

```bash
# 1. Clone or navigate to the repository
cd /path/to/DataLoader

# 2. Run the automated setup script
chmod +x setup.sh
./setup.sh

# This will:
# - Install TA-Lib system library
# - Create Python virtual environment
# - Install all dependencies
# - Verify installation
```

## Run the Pipeline

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run with default settings
python pipeline_runner.py

# OR specify custom output directory
python pipeline_runner.py --output-dir /your/custom/path

# OR enable debug logging
python pipeline_runner.py --log-level DEBUG
```

## What Happens?

The pipeline will:

1. **Download Data** (~5-10 min)
   - SPY and VIX from 1993-present
   - Market indicators (DJI, bonds, commodities, etc.)
   - ETF data (SPXS, SPXL)

2. **Process Features** (~5-10 min)
   - Calculate 40+ technical indicators
   - Detect market cycles
   - Train HMM models
   - Generate Mandelbrot coordinates
   - Create lagged features (0-15 lags)

3. **Save Output**
   - Final data: `output/data.csv` (~1000+ columns)
   - ETF data: `output/data_etf.csv`
   - Logs: `output/logs/pipeline_TIMESTAMP.log`

## Expected Output Location

Default: `/data/afterford/output/`

```
output/
├── data.csv              ← Main output (lagged features)
├── data_etf.csv          ← ETF OHLCV data
├── data/
│   ├── raw/              ← Raw downloads
│   └── processed/        ← Intermediate data
└── logs/                 ← Execution logs
```

## Verify Success

```bash
# Check if main output exists
ls -lh /data/afterford/output/data.csv

# View last few rows
tail /data/afterford/output/data.csv

# Check log for errors
tail -50 /data/afterford/output/logs/pipeline_*.log
```

## Common Issues

### TA-Lib Import Error

```bash
# Fix library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# Or
sudo ldconfig
```

### Permission Denied

```bash
# Create output directory with permissions
sudo mkdir -p /data/afterford/output
sudo chmod 775 /data/afterford/output
sudo chown $USER:$USER /data/afterford/output
```

### Rate Limit Errors

The pipeline includes automatic delays. If you still get rate limited:
- Wait 15 minutes and try again
- Run during off-peak hours
- The downloaded data is cached in `data/raw/` - you can reprocess without re-downloading

## Next Steps

- **Read README.md** for detailed documentation
- **Customize Config** in `data_pipeline.py` (see Config class)
- **Schedule Regular Runs** using cron or systemd timer
- **Integrate with Models** using the output CSV

## Command Reference

```bash
# Help
python pipeline_runner.py --help

# Custom output directory
python pipeline_runner.py --output-dir /custom/path

# Debug logging
python pipeline_runner.py --log-level DEBUG

# Set via environment variable
export DATA_PIPELINE_OUTPUT_DIR=/custom/path
python pipeline_runner.py
```

## Performance

**Typical Execution Time:**
- Small dataset (1 year): ~2-5 minutes
- Full dataset (20+ years): ~15-30 minutes

**Resource Usage:**
- RAM: 4-8GB typical, 16GB recommended
- Disk: ~1-5GB for outputs
- Network: ~100-500MB download

## Getting Help

1. Check logs: `output/logs/pipeline_*.log`
2. Read README.md for troubleshooting
3. Review error messages in console output
4. Verify all dependencies installed correctly

## File Structure

```
DataLoader/
├── data_pipeline.py       # Core utilities and classes
├── processors.py          # Feature processors
├── pipeline_runner.py     # Main execution (RUN THIS)
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── README.md             # Full documentation
└── QUICKSTART.md         # This file
```

---

**Ready to run?** Just execute:

```bash
./setup.sh && source venv/bin/activate && python pipeline_runner.py
```

That's it! 🚀
