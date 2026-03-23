#!/usr/bin/env bash
#
# IRIS Quick Start Script (uv version)
#
# Usage:
#   ./start.sh                    # Run single update cycle
#   ./start.sh scheduler start    # Start scheduler
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
    scheduler)
        # Handle scheduler subcommands
        case "${ARGS[1]:-start}" in
            start)
                print_header "Starting IRIS Scheduler"
                print_info "Scheduler mode with persistent task state"
                print_info "Press Ctrl+C to stop"
                $PYTHON scripts/run_update_cycle.py --config "$CONFIG_FILE" --scheduler "${ARGS[@]:2}"
                ;;
            status)
                print_header "Scheduler Status"
                $PYTHON -c "
from scheduler.task_service import TaskService
from scheduler.scheduler_orchestrator import SchedulerOrchestrator
from src.config import get_config

config = get_config().config
service = TaskService(config['storage']['paper_db_path'])
stats = service.get_task_stats()
print(f'Total tasks: {stats[\"total\"]}')
for status, count in stats.items():
    if status != 'total':
        print(f'  {status}: {count}')
" 2>/dev/null || print_error "Scheduler module not available"
                ;;
            tasks)
                print_header "Recent Tasks"
                $PYTHON -c "
from scheduler.task_service import TaskService
from src.config import get_config

config = get_config().config
service = TaskService(config['storage']['paper_db_path'])
tasks = service.get_recent_tasks(limit=10)
print(f'\\nRecent {len(tasks)} tasks:')
print(f'{'Task ID':<36} {'Type':<15} {'Status':<12} {'Scheduled':<20}')
print('-' * 90)
for task in tasks:
    print(f'{task.task_id:<36} {task.task_type:<15} {task.status:<12} {task.scheduled_time.strftime(\"%Y-%m-%d %H:%M:%S\"):<20}')
" 2>/dev/null || print_error "Scheduler module not available"
                ;;
            cancel)
                if [ -z "${ARGS[2]}" ]; then
                    print_error "Usage: ./start.sh scheduler cancel <task_id>"
                    exit 1
                fi
                print_header "Cancelling Task: ${ARGS[2]}"
                $PYTHON -c "
from scheduler.task_service import TaskService
from src.config import get_config
import sys

config = get_config().config
service = TaskService(config['storage']['paper_db_path'])
task_id = '${ARGS[2]}'
success = service.cancel_task(task_id)
if success:
    print('Task cancelled successfully')
    sys.exit(0)
else:
    print('Failed to cancel task')
    sys.exit(1)
" 2>/dev/null || print_error "Scheduler module not available"
                ;;
            run-now)
                print_header "Triggering Immediate Update"
                $PYTHON -c "
from scheduler.scheduler_orchestrator import SchedulerOrchestrator
from src.config import get_config
import sys

config = get_config().config
orchestrator = SchedulerOrchestrator(config)
success = orchestrator.run_single_cycle()
if success:
    print('Update cycle triggered')
    sys.exit(0)
else:
    print('Failed to trigger update')
    sys.exit(1)
" 2>/dev/null || print_error "Scheduler module not available"
                ;;
            *)
                print_error "Unknown scheduler command: ${ARGS[1]}"
                print_info "Available commands: start, status, tasks, cancel <task_id>, run-now"
                exit 1
                ;;
        esac
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
        echo "  update               Run single update cycle (default)"
        echo "  scheduler start      Start scheduler with persistent state (recommended)"
        echo "  scheduler status     Show scheduler status"
        echo "  scheduler tasks      List recent tasks"
        echo "  scheduler cancel <id> Cancel a task"
        echo "  scheduler run-now    Trigger immediate update"
        echo "  query                Start interactive query mode"
        echo "  web                  Start web interface"
        echo "  help                 Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./start.sh"
        echo "  ./start.sh --config configs/config_ex.yaml"
        echo "  ./start.sh scheduler start"
        echo "  ./start.sh scheduler tasks"
        echo "  ./start.sh query"
        echo "  ./start.sh web"
        ;;
    *)
        print_error "Unknown command: ${ARGS[0]}"
        print_info "Use './start.sh help' for usage information"
        exit 1
        ;;
esac
