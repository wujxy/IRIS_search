"""
Dependency injection for IRIS Web module.
Provides FastAPI dependencies for services.
"""

from fastapi import Depends
from services.paper_service import PaperService
from utils.helpers import load_config


def get_paper_service() -> PaperService:
    """Get PaperService instance with configured database path."""
    config = load_config()
    db_path = config['storage']['paper_db_path']
    return PaperService(db_path)


def get_web_config() -> dict:
    """Get web configuration from config.yaml."""
    config = load_config()
    return config.get('web', {
        'host': '127.0.0.1',
        'port': 8000,
        'reload': True,
        'log_level': 'info',
        'pagination': {
            'default_per_page': 10,
            'max_per_page': 100
        }
    })
