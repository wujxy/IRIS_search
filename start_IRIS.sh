#!/bin/bash
#
# IRIS Startup Script
# Starts the IRIS literature monitoring service.
#
# Usage:
#   ./start_IRIS.sh              # Run single update cycle
#   ./start_IRIS.sh --daemon    # Run in daemon mode with periodic updates
#   ./start_IRIS.sh --help      # Show help
#

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python environment
PYTHON_ENV=".venv/bin/python"

# Default options
DAEMON_MODE=false
INTERVAL_HOURS=2
CONFIG_FILE="${SCRIPT_DIR}/configs/config.yaml"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        --interval)
            INTERVAL_HOURS=$2
            shift 2
            ;;
        --config)
            CONFIG_FILE=$2
            shift 2
            ;;
        -h|--help)
            echo "IRIS Startup Script"
            echo ""
            echo "Usage:"
            echo "  ./start_IRIS.sh [options]"
            echo ""
            echo "Options:"
            echo "  --daemon              Run in daemon mode with periodic updates"
            echo "  --interval HOURS       Update interval in hours (default: 2)"
            echo "  --config PATH         Path to config file (default: configs/config.yaml)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./start_IRIS.sh"
            echo "  ./start_IRIS.sh --daemon --interval 4"
            echo "  ./start_IRIS.sh --config custom_config.yaml"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python environment exists
if [ ! -f "$PYTHON_ENV" ]; then
    print_error "Python environment not found: $PYTHON_ENV"
    exit 1
fi

print_info "Starting IRIS literature monitoring service..."
print_info "Python environment: $PYTHON_ENV"
print_info "Configuration file: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Function to run single update cycle
run_update_cycle() {
    print_info "Running update cycle..."
    $PYTHON_ENV scripts/run_update_cycle.py --config "$CONFIG_FILE" --log-level DEBUG
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        print_info "Update cycle completed successfully"
    else
        print_error "Update cycle failed with exit code: $exit_code"
    fi

    return $exit_code
}

# Daemon mode: run update cycles periodically
if [ "$DAEMON_MODE" = true ]; then
    print_info "Running in daemon mode..."
    print_info "Update interval: $INTERVAL_HOURS hour(s)"
    print_info "Press Ctrl+C to stop"

    while true; do
        echo ""
        echo "========================================"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting update cycle"
        echo "========================================"

        run_update_cycle

        # Wait for next update
        sleep_seconds=$((INTERVAL_HOURS * 3600))
        print_info "Next update in $INTERVAL_HOURS hour(s) (sleeping for $sleep_seconds seconds)..."

        # Sleep in smaller chunks to handle Ctrl+C properly
        for ((i=0; i<$sleep_seconds; i+=60)); do
            sleep 60
        done
    done
else
    # Single update cycle mode
    run_update_cycle
fi
