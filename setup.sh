#!/usr/bin/env bash
set -e

#
# IRIS One-Click Setup Script with uv
#
# This script automates the entire setup process for IRIS:
# 1. Checks and installs uv (if needed)
# 2. Creates/updates virtual environment
# 3. Installs all dependencies
# 4. Verifies installation
#
# Usage:
#   ./setup.sh                  # Standard setup
#   ./setup.sh --gpu            # Setup with GPU dependencies
#   ./setup.sh --dev            # Setup with dev tools
#   ./setup.sh --all            # Setup with all optional dependencies
#   ./setup.sh --reinstall      # Force reinstall everything
#

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${CYAN}${BOLD}$1${NC}"
    echo "========================================"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
INSTALL_GPU=false
INSTALL_DEV=false
INSTALL_ALL=false
FORCE_REINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            INSTALL_GPU=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        -h|--help)
            echo "IRIS One-Click Setup Script with uv"
            echo ""
            echo "Usage:"
            echo "  ./setup.sh [options]"
            echo ""
            echo "Options:"
            echo "  --gpu           Install GPU dependencies (torch, torchvision)"
            echo "  --dev           Install development tools (pytest, black, ruff, mypy)"
            echo "  --all           Install all optional dependencies"
            echo "  --reinstall     Force reinstall virtual environment"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./setup.sh"
            echo "  ./setup.sh --gpu"
            echo "  ./setup.sh --dev --gpu"
            echo "  ./setup.sh --all"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clear screen for cleaner output
clear

print_header "IRIS - Intelligent Research Information System"
echo -e "${BOLD}One-Click Setup with uv${NC}\n"
echo "This script will:"
echo "  1. Check/install uv package manager"
echo "  2. Create/update Python virtual environment"
echo "  3. Install all project dependencies"
echo "  4. Verify installation"
echo ""

# Check if user wants to continue
read -p "Continue? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_info "Setup cancelled"
    exit 0
fi

# ============================================
# Step 1: Check uv installation
# ============================================
print_header "Step 1: Checking uv Installation"

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    print_info "uv is already installed: $UV_VERSION"

    if [ "$FORCE_REINSTALL" = true ]; then
        print_warning "Force reinstall requested, updating uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        print_info "uv updated successfully"
    fi
else
    print_info "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        print_info "uv installed successfully"
    else
        print_error "Failed to install uv"
        print_info "Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

# ============================================
# Step 2: Create/Update Virtual Environment
# ============================================
print_header "Step 2: Setting Up Virtual Environment"

if [ -d ".venv" ] && [ "$FORCE_REINSTALL" = true ]; then
    print_warning "Removing existing .venv directory..."
    rm -rf .venv
fi

if [ -d ".venv" ]; then
    print_info "Virtual environment already exists at .venv/"
    print_info "Updating dependencies..."
else
    print_info "Creating new virtual environment..."
fi

# Sync dependencies (uv will create venv if it doesn't exist)
EXTRA_ARGS=""
if [ "$INSTALL_GPU" = true ] || [ "$INSTALL_ALL" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --extra gpu"
fi
if [ "$INSTALL_DEV" = true ] || [ "$INSTALL_ALL" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --extra dev"
fi

uv sync $EXTRA_ARGS

if [ -d ".venv" ]; then
    print_info "Virtual environment ready"
else
    print_error "Failed to create virtual environment"
    exit 1
fi

# ============================================
# Step 3: Install vLLM (separate due to special install)
# ============================================
print_header "Step 3: Installing vLLM"

print_info "Installing vLLM for fast LLM inference..."
uv pip install "vllm>=0.6.0" --system

print_info "vLLM installed successfully"

# ============================================
# Step 4: Download NLTK data
# ============================================
print_header "Step 4: Downloading NLTK Data"

print_info "Downloading NLTK sentence tokenizer data..."
NLTK_DOWNLOAD=$(.venv/bin/python -c "
import nltk
import sys
try:
    nltk.data.find('tokenizers/punkt')
    print('punkt already downloaded')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        print('punkt downloaded successfully')
    except Exception as e:
        print(f'punkt download failed: {e}', file=sys.stderr)
        sys.exit(1)

try:
    nltk.data.find('tokenizers/punkt_tab')
    print('punkt_tab already downloaded')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
        print('punkt_tab downloaded successfully')
    except Exception as e:
        print(f'punkt_tab download failed: {e}', file=sys.stderr)
        sys.exit(1)

print('All NLTK data downloaded successfully')
" 2>&1)

if [ $? -eq 0 ]; then
    print_info "NLTK data ready"
    echo "$NLTK_DOWNLOAD" | grep -v "already downloaded" | head -5
else
    print_error "Failed to download NLTK data"
    echo "$NLTK_DOWNLOAD"
    print_warning "NLTK data will be downloaded on first use"
fi

# ============================================
# Step 5: Verify Installation
# ============================================
print_header "Step 5: Verifying Installation"

print_info "Python version:"
.venv/bin/python --version

print_info "Installed packages:"
.venv/bin/pip list | grep -E "(arxiv|pymilvus|fastapi|vllm|nltk)" || true

# Test imports (including NLTK)
print_info "Testing Python imports..."
TEST_IMPORTS=$(.venv/bin/python -c "
import sys
failed = []
packages = ['arxiv', 'yaml', 'fitz', 'chonkie', 'pymilvus', 'openai', 'fastapi', 'uvicorn', 'nltk']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        failed.append(pkg)
        print(f'  ✗ {pkg} - FAILED')
if failed:
    sys.exit(1)
" 2>&1)

if [ $? -eq 0 ]; then
    print_info "All core packages imported successfully"
else
    print_error "Some packages failed to import"
    echo "$TEST_IMPORTS"
    exit 1
fi

# ============================================
# Step 6: Create necessary directories
# ============================================
print_header "Step 6: Creating Directory Structure"

mkdir -p logs
print_info "Created logs directory"

print_info "Directory structure:"
tree -L 2 -d 2>/dev/null || find . -maxdepth 2 -type d | grep -v ".git" | grep -v "__pycache__" | head -20

# ============================================
# Step 7: Check prerequisites
# ============================================
print_header "Step 7: Checking Prerequisites"

# Check Python version
PYTHON_VERSION=$(.venv/bin/python --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# Check Docker (for Milvus)
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_info "Docker: $DOCKER_VERSION"
else
    print_warning "Docker not found. Required for Milvus vector database."
    print_info "Install Docker: https://docs.docker.com/get-docker/"
fi

# Check GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || true
else
    print_warning "No NVIDIA GPU detected. CPU-only mode will be used."
fi

# ============================================
# Step 8: Configuration check
# ============================================
print_header "Step 8: Configuration"

if [ -f "configs/config.yaml" ]; then
    print_info "Configuration file found: configs/config.yaml"
    print_warning "Please review and update configs/config.yaml with your settings:"
    print_warning "  - Model paths"
    print_warning "  - Database paths"
    print_warning "  - Email settings (optional)"
else
    print_error "Configuration file not found: configs/config.yaml"
    print_info "Please create a configuration file before running IRIS"
fi

# ============================================
# Summary
# ============================================
print_header "Setup Complete!"

echo -e "${GREEN}${BOLD}IRIS has been successfully installed!${NC}\n"

echo "Next steps:"
echo ""
echo "1. Configure IRIS:"
echo "   nano configs/config.yaml"
echo ""
echo "2. Start Milvus (vector database):"
echo "   bash build_milvus.sh"
echo ""
echo "3. Run IRIS update cycle:"
echo "   ./start_IRIS.sh"
echo ""
echo "4. Or use the query interface:"
echo "   .venv/bin/python scripts/iris_query.py --interactive"
echo ""
echo "5. Or start the web interface:"
echo "   .venv/bin/python scripts/run_web.py"
echo ""

echo "Available commands:"
echo "  ./start_IRIS.sh              # Run single update cycle"
echo "  ./start_IRIS.sh --scheduler  # Start scheduler mode"
echo "  .venv/bin/python scripts/iris_query.py --help"
echo "  .venv/bin/python scripts/run_web.py"
echo ""

if [ "$INSTALL_GPU" = false ] && [ "$INSTALL_ALL" = false ]; then
    echo -e "${YELLOW}Note: GPU dependencies were not installed.${NC}"
    echo "      To enable GPU support, run: ./setup.sh --gpu"
    echo ""
fi

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Note: Docker is not installed.${NC}"
    echo "      Install Docker to use Milvus vector database."
    echo "      https://docs.docker.com/get-docker/"
    echo ""
fi

print_info "For full documentation, see README.md or IRIS_Tutorial.md"
