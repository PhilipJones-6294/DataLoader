#!/bin/bash
# Setup script for Data Pipeline on Nvidia DGX

set -e  # Exit on error

echo "=========================================="
echo "Data Pipeline Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root for system dependencies
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Note: You may need sudo privileges for system dependencies.${NC}"
    echo ""
fi

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}✓ Python $python_version found${NC}"
else
    echo -e "${RED}✗ Python $required_version or higher is required. Found: $python_version${NC}"
    exit 1
fi

# Step 2: Install TA-Lib system dependency
echo ""
echo "Step 2: Checking TA-Lib library..."
if ldconfig -p | grep -q libta-lib; then
    echo -e "${GREEN}✓ TA-Lib library already installed${NC}"
else
    echo -e "${YELLOW}TA-Lib not found. Attempting to install...${NC}"

    # Check if we have sudo
    if command -v sudo &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y build-essential wget

        cd /tmp
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr
        make
        sudo make install
        sudo ldconfig

        cd ..
        rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
        cd -

        echo -e "${GREEN}✓ TA-Lib installed successfully${NC}"
    else
        echo -e "${RED}✗ sudo not available. Please install TA-Lib manually.${NC}"
        echo "See README.md for installation instructions."
        exit 1
    fi
fi

# Step 3: Create virtual environment
echo ""
echo "Step 3: Setting up Python virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Step 4: Activate virtual environment and install dependencies
echo ""
echo "Step 4: Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Step 5: Verify installation
echo ""
echo "Step 5: Verifying installation..."

verification_failed=0

# Test TA-Lib
if python -c "import talib" 2>/dev/null; then
    echo -e "${GREEN}✓ TA-Lib Python wrapper${NC}"
else
    echo -e "${RED}✗ TA-Lib Python wrapper${NC}"
    verification_failed=1
fi

# Test other imports
for module in pandas numpy yfinance hmmlearn scipy; do
    if python -c "import $module" 2>/dev/null; then
        echo -e "${GREEN}✓ $module${NC}"
    else
        echo -e "${RED}✗ $module${NC}"
        verification_failed=1
    fi
done

# Step 6: Create default output directory
echo ""
echo "Step 6: Setting up output directories..."
default_output_dir="/data/afterford/output"

read -p "Create default output directory at $default_output_dir? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$default_output_dir"/{data/{raw,processed},logs,models,visualizations,temp}
    echo -e "${GREEN}✓ Output directories created at $default_output_dir${NC}"

    # Set permissions if needed
    if [ "$EUID" -eq 0 ]; then
        chmod -R 775 "$default_output_dir"
        echo -e "${GREEN}✓ Permissions set${NC}"
    fi
else
    echo -e "${YELLOW}Skipped. You can create directories later or specify a custom path when running the pipeline.${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="

if [ $verification_failed -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "To run the pipeline:"
    echo "  1. Activate the virtual environment:"
    echo "     source venv/bin/activate"
    echo ""
    echo "  2. Run the pipeline:"
    echo "     python pipeline_runner.py"
    echo ""
    echo "  3. For custom output directory:"
    echo "     python pipeline_runner.py --output-dir /your/custom/path"
    echo ""
    echo "For more options, see:"
    echo "  python pipeline_runner.py --help"
    echo ""
    echo "For detailed documentation, see README.md"
else
    echo -e "${RED}✗ Some checks failed. Please review the errors above.${NC}"
    echo "See README.md for troubleshooting steps."
    exit 1
fi

echo "=========================================="
