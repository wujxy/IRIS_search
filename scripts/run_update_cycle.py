#!/usr/bin/env python3
"""
IRIS Update Cycle Orchestrator
Main script to run a complete IRIS update cycle.

Usage:
    python scripts/run_update_cycle.py
    python scripts/run_update_cycle.py --config configs/config.yaml
    python scripts/run_update_cycle.py --scheduler  # New scheduler mode
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

from src.config import get_config
from src.common import setup_logging
from core.orchestrator import UpdateOrchestrator
from scheduler.scheduler_orchestrator import SchedulerOrchestrator

logger = logging.getLogger(__name__)




def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IRIS Update Cycle Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_update_cycle.py
  python run_update_cycle.py --config configs/config.yaml
  python run_update_cycle.py --log-level DEBUG
  python run_update_cycle.py --scheduler --interval 2
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (overrides config)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Hours between update cycles (default: from config)"
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Run with APScheduler-based scheduler (persistent state)"
    )

    args = parser.parse_args()

    # Load configuration
    config_loader = get_config(args.config)
    config = config_loader.config

    # Override log level if specified
    if args.log_level:
        config["logging"]["level"] = args.log_level

    # Setup logging - create log file path from log_dir
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"iris_update_{timestamp}.log"

    setup_logging(
        level=config["logging"]["level"],
        log_file=str(log_file),
        console=True
    )

    # Run orchestrator
    if args.scheduler:
        # APScheduler-based mode
        interval = args.interval or config.get("scheduler", {}).get(
            "default_interval_hours",
            config.get("update", {}).get("interval_hours", 24)
        )
        orchestrator = SchedulerOrchestrator(config)
        try:
            orchestrator.start_scheduler(interval_hours=interval)
            # Keep running until interrupted
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down scheduler...")
            orchestrator.stop_scheduler()
            logger.info("Scheduler stopped")
    else:
        # Single update cycle (default)
        orchestrator = UpdateOrchestrator(config)
        success = orchestrator.run_cycle()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()