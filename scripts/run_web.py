#!/usr/bin/env python3
"""
IRIS Web Server Startup Script
Starts the FastAPI web server for literature browsing.

Usage:
    python run_web.py
    python run_web.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

import uvicorn
import logging
from src.config import get_config
from src.common import setup_logging
from services.deploy_service import DeployService

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def check_and_start_models(deploy_server):
    """
    Check and start Milvus, embedding, and QA models if needed.
    Web service will start regardless of status (graceful degradation).
    """
    import time

    # 1. Check and start Milvus container first
    logger.info("Checking Milvus container...")
    if not deploy_server.milvus_control.search():
        logger.warning("Milvus container not running, attempting to start...")
        if deploy_server.milvus_control.start():
            logger.info("Milvus container started successfully")
            # Wait for Milvus to be ready
            logger.info("Waiting for Milvus to be ready...")
            time.sleep(10)
        else:
            logger.error("Failed to start Milvus container")
    else:
        logger.info("Milvus container already running")

    # 2. Check embedding model (port 65503)
    logger.info("Checking embedding model...")
    if not deploy_server.index_vllm.search(timeout=10):
        logger.warning("Embedding model not running, attempting to start...")
        if deploy_server.index_vllm.start():
            if deploy_server.index_vllm.search(timeout=60):
                logger.info("Embedding model started successfully")
            else:
                logger.error("Embedding model failed to become ready")
        else:
            logger.error("Failed to start embedding model")
    else:
        logger.info("Embedding model already running")

    # Check QA model (port 65504)
    logger.info("Checking QA model...")
    if not deploy_server.qa_vllm.search(timeout=10):
        logger.warning("QA model not running, attempting to start...")
        if deploy_server.qa_vllm.start():
            if deploy_server.qa_vllm.search(timeout=60):
                logger.info("QA model started successfully")
            else:
                logger.error("QA model failed to become ready")
        else:
            logger.error("Failed to start QA model")
    else:
        logger.info("QA model already running")

    # Continue to start web service regardless of infrastructure status
    logger.info("Web service will start regardless of infrastructure status")


def main():
    """Start the IRIS web server."""
    parser = argparse.ArgumentParser(
        description="IRIS Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web.py
  python run_web.py --config configs/config.yaml
  python run_web.py --config configs/config_ex.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/config.yaml)"
    )

    args = parser.parse_args()

    # Load configuration
    config_loader = get_config(args.config)
    config = config_loader.config
    web_config = config_loader.web or {}

    host = web_config.get('host', '127.0.0.1')
    port = web_config.get('port', 8000)
    reload = web_config.get('reload', True)
    log_level = web_config.get('log_level', 'info')

    # Check and start models before web service (similar to run_update_cycle.py line 262)
    logger.info("Checking model availability before starting web service...")
    deploy_server = DeployService(config)
    check_and_start_models(deploy_server)

    logger.info(f"Starting IRIS Web Server on {host}:{port}")

    # Run the server
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower()
    )


if __name__ == '__main__':
    main()
