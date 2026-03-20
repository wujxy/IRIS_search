#!/usr/bin/env python3
"""
IRIS Web Server Startup Script
Starts the FastAPI web server for literature browsing.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

import uvicorn
import logging
from utils.helpers import load_config, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Start the IRIS web server."""
    # Load configuration
    config = load_config()
    web_config = config.get('web', {
        'host': '127.0.0.1',
        'port': 8000,
        'reload': True,
        'log_level': 'info'
    })

    host = web_config.get('host', '127.0.0.1')
    port = web_config.get('port', 8000)
    reload = web_config.get('reload', True)
    log_level = web_config.get('log_level', 'info')

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
