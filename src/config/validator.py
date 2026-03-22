"""
IRIS Configuration Validator

Validates configuration values before use.
"""

import logging
import socket
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Configuration validator for IRIS."""

    def __init__(self):
        """Initialize validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate all configuration sections.

        Args:
            config: Configuration dictionary

        Returns:
            True if validation passes (no errors)
        """
        self.errors = []
        self.warnings = []

        # Validate each section
        self._validate_models(config.get('models', {}))
        self._validate_storage(config.get('storage', {}))
        self._validate_milvus(config.get('milvus', {}))
        self._validate_embedding(config.get('embedding', {}))
        self._validate_qa(config.get('qa', {}))
        self._validate_arxiv(config.get('arxiv', {}))
        self._validate_web(config.get('web', {}))

        # Log results
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Config validation warning: {warning}")

        if self.errors:
            for error in self.errors:
                logger.error(f"Config validation error: {error}")
            return False

        logger.info("Configuration validation passed")
        return True

    def _validate_models(self, models: Dict[str, Any]):
        """Validate models configuration."""
        required_paths = ['embedding_model_path', 'reranker_model_path', 'llm_model_path']

        for path_key in required_paths:
            path = models.get(path_key)
            if path:
                if not Path(path).exists():
                    self.errors.append(f"Model path not found: {path_key} = {path}")
                else:
                    logger.debug(f"Model path OK: {path_key}")
            else:
                self.warnings.append(f"Model path not specified: {path_key}")

    def _validate_storage(self, storage: Dict[str, Any]):
        """Validate storage configuration."""
        if 'database_root' in storage:
            db_root = Path(storage['database_root'])
            if not db_root.exists():
                self.warnings.append(
                    f"Database root does not exist: {db_root}\n"
                    f"It will be created on first run."
                )

        if 'paper_db_path' in storage:
            db_path = Path(storage['paper_db_path'])
            db_dir = db_path.parent
            if not db_dir.exists():
                self.warnings.append(
                    f"Database directory does not exist: {db_dir}\n"
                    f"It will be created on first run."
                )

    def _validate_milvus(self, milvus: Dict[str, Any]):
        """Validate Milvus configuration."""
        if milvus.get('enabled', True):
            # Check if port is available
            host = milvus.get('uri', 'http://localhost:29901')
            # Extract port if present
            if ':29901' in host:
                port = 29901
                if self._is_port_in_use(port):
                    self.warnings.append(
                        f"Milvus port {port} appears to be in use.\n"
                        f"Ensure Milvus Docker container is running."
                    )

    def _validate_embedding(self, embedding: Dict[str, Any]):
        """Validate embedding configuration."""
        required_fields = ['base_url', 'model_name']
        for field in required_fields:
            if not embedding.get(field):
                self.warnings.append(f"Embedding {field} not specified")

    def _validate_qa(self, qa: Dict[str, Any]):
        """Validate QA configuration."""
        required_fields = ['base_url', 'model_name']
        for field in required_fields:
            if not qa.get(field):
                self.warnings.append(f"QA {field} not specified")

    def _validate_arxiv(self, arxiv: Dict[str, Any]):
        """Validate arXiv configuration."""
        if not arxiv.get('keywords'):
            self.warnings.append("No arXiv search keywords configured")

        max_results = arxiv.get('max_results_per_keyword', 0)
        if max_results <= 0:
            self.warnings.append("max_results_per_keyword should be positive")

    def _validate_web(self, web: Dict[str, Any]):
        """Validate web configuration."""
        port = web.get('port', 8000)
        if self._is_port_in_use(port):
            self.warnings.append(
                f"Web port {port} is in use.\n"
                f"Choose a different port or stop the conflicting service."
            )

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex(('localhost', port)) == 0
        except Exception:
            return False
