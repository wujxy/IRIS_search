#!/usr/bin/env bash
#
# IRIS Quick Start Script (uv version)
#
# Usage:
#   ./start.sh                    # Run single update cycle
#   ./start.sh daemon             # Run in daemon mode
#   ./start.sh query              # Interactive query mode
#   ./start.sh web                # Start web interface
#   ./start.sh --config <file>    # Specify config file
#   ./start.sh help               # Show help
#

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo -e "\n${CYAN}${BOLD}$1${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv environment exists
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found!"
    print_info "Please run ./setup.sh first to set up IRIS."
    exit 1
fi

# Use uv to run Python
PYTHON="uv run --no-sync python"

# Default config file
CONFIG_FILE="configs/config.yaml"

# Parse arguments to extract --config option
ARGS=("$@")
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

print_info "Using config: $CONFIG_FILE"

# Parse command
case "${ARGS[0]:-update}" in
    update)
        print_header "Starting IRIS Update Cycle"
        $PYTHON scripts/run_update_cycle.py --config "$CONFIG_FILE" "${ARGS[@]:1}"
        ;;
    daemon)
        print_header "Starting IRIS Daemon Mode"
        print_info "Press Ctrl+C to stop"
        $PYTHON scripts/run_update_cycle.py --config "$CONFIG_FILE" --daemon "${ARGS[@]:1}"
        ;;
    query)
        print_header "IRIS Interactive Query Mode"
        $PYTHON scripts/iris_query.py --config "$CONFIG_FILE" --interactive
        ;;
    web)
        print_header "Starting IRIS Web Interface"
        print_info "Web server will be available at: http://127.0.0.1:8000"
        $PYTHON scripts/run_web.py --config "$CONFIG_FILE"
        ;;
    help|--help|-h)
        echo "IRIS Quick Start Script"
        echo ""
        echo "Usage: ./start.sh [options] [command]"
        echo ""
        echo "Options:"
        echo "  --config, -c <file>  Specify config file (default: configs/config.yaml)"
        echo ""
        echo "Commands:"
        echo "  update     Run single update cycle (default)"
        echo "  daemon     Run in daemon mode with periodic updates"
        echo "  query      Start interactive query mode"
        echo "  web        Start web interface"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./start.sh"
        echo "  ./start.sh --config configs/config_ex.yaml"
        echo "  ./start.sh daemon --interval 2"
        echo "  ./start.sh query"
        echo "  ./start.sh web"
        ;;
    *)
        print_error "Unknown command: ${ARGS[0]}"
        print_info "Use './start.sh help' for usage information"
        exit 1
        ;;
esac
