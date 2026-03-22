"""
IRIS Utility Functions
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def expand_env_vars(config):
    """
    Expand environment variables in configuration.

    Replaces ${VAR_NAME} patterns with values from environment variables.
    Useful for API keys and sensitive configuration.

    Args:
        config: Configuration dict, list, or string

    Returns:
        Configuration with environment variables expanded
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        def replace_env(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        return re.sub(r'\$\{([^}]+)\}', replace_env, config)
    else:
        return config


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary containing configuration with environment variables expanded
    """
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Expand environment variables (e.g., ${API_KEY} -> actual value)
    config = expand_env_vars(config)

    return config


def load_questions(question_set_path: str = None) -> List[str]:
    """
    Load standard question set for paper summarization.

    Args:
        question_set_path: Path to questions.txt file

    Returns:
        List of questions
    """
    if question_set_path is None:
        question_set_path = Path(__file__).parent.parent / "configs" / "questions.txt"

    with open(question_set_path, 'r', encoding='utf-8') as f:
        questions = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

    return questions


def setup_logging(
    log_dir: str = "./logs",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Specific log file name (optional)

    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"iris_{timestamp}.log"

    log_path = log_dir / log_file

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger


def create_update_folder(database_root: str) -> Path:
    """
    Create a new timestamped update folder in the database.

    Args:
        database_root: Path to the IRIS database root

    Returns:
        Path to the newly created update folder
    """
    database_root = Path(database_root)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    update_folder = database_root / f"update_{timestamp}"

    # Create subdirectories
    update_folder.mkdir(parents=True, exist_ok=True)
    (update_folder / "pdfs").mkdir(exist_ok=True)
    (update_folder / "logs").mkdir(exist_ok=True)

    return update_folder


def save_metadata(papers: List[Dict[str, Any]], metadata_path: Path):
    """
    Save paper metadata to JSON file.

    Args:
        papers: List of paper metadata dictionaries
        metadata_path: Path to save metadata.json
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_papers": len(papers),
        "new_papers": [p for p in papers if p.get("status") == "new"],
        "duplicate_papers": [p for p in papers if p.get("status") == "duplicate"],
        "review_papers": [p for p in papers if p.get("status") == "review"],
        "download_failed_papers": [p for p in papers if p.get("status") == "download_failed"],
        "all_papers": papers
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_all_paper_entries(database_root: str) -> set:
    """
    Load all existing paper entry_ids from the database.

    Args:
        database_root: Path to the IRIS database root

    Returns:
        Set of all existing entry_ids
    """
    database_root = Path(database_root)
    entry_ids = set()

    # Iterate through all update folders
    for update_folder in database_root.glob("update_*"):
        metadata_file = update_folder / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for paper in data.get("all_papers", []):
                        entry_id = paper.get("entry_id")
                        if entry_id:
                            entry_ids.add(entry_id)
            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupted metadata files
                continue

    return entry_ids


def is_review_paper(title: str, summary: str, review_keywords: List[str]) -> bool:
    """
    Check if a paper is a review/survey/overview paper.

    Args:
        title: Paper title
        summary: Paper abstract
        review_keywords: Keywords that indicate review papers

    Returns:
        True if paper is a review paper
    """
    text = (title + " " + summary).lower()

    for keyword in review_keywords:
        if keyword.lower() in text:
            return True

    return False


def paper_to_dict(arxiv_result) -> Dict[str, Any]:
    """
    Convert arXiv search result to dictionary format.

    Args:
        arxiv_result: arxiv.Result object

    Returns:
        Dictionary containing paper metadata
    """
    return {
        "entry_id": arxiv_result.entry_id,
        "updated": arxiv_result.updated.isoformat() if arxiv_result.updated else None,
        "published": arxiv_result.published.isoformat() if arxiv_result.published else None,
        "title": arxiv_result.title,
        "authors": [str(author) for author in arxiv_result.authors],
        "summary": arxiv_result.summary.replace('\n', ' ').strip(),
        "comment": arxiv_result.comment,
        "journal_ref": arxiv_result.journal_ref,
        "doi": arxiv_result.doi,
        "primary_category": arxiv_result.primary_category,
        "categories": arxiv_result.categories,
        "pdf_url": arxiv_result.pdf_url,
        "status": "new"  # Will be updated by the orchestrator
    }


def save_update_log(update_folder: Path, content: str):
    """
    Save update log to markdown file.

    Args:
        update_folder: Path to the update folder
        content: Log content as string
    """
    log_path = update_folder / "update_log.md"

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(content)


def save_summary_log(update_folder: Path, summaries: str):
    """
    Save paper summaries to markdown file.

    Args:
        update_folder: Path to the update folder
        summaries: Summaries as markdown string
    """
    log_path = update_folder / "logs" / "summary_log.md"

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(summaries)


def save_knowledge_log(update_folder: Path, knowledge: str):
    """
    Save extracted knowledge to markdown file.

    Args:
        update_folder: Path to the update folder
        knowledge: Knowledge as markdown string
    """
    log_path = update_folder / "logs" / "knowledge_log.md"

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(knowledge)


def get_latest_update_folder(database_root: str) -> Optional[Path]:
    """
    Get the most recent update folder from the database.

    Args:
        database_root: Path to the IRIS database root

    Returns:
        Path to the latest update folder, or None if no updates exist
    """
    database_root = Path(database_root)
    update_folders = sorted(database_root.glob("update_*"), reverse=True)

    if update_folders:
        return update_folders[0]

    return None


def get_master_chunks_path(database_root: str) -> Path:
    """
    Get path to master chunks.jsonl file.

    Args:
        database_root: Path to IRIS database root

    Returns:
        Path to master chunks.jsonl
    """
    database_root = Path(database_root)
    master_storage = database_root / "master_index_storage"
    master_storage.mkdir(parents=True, exist_ok=True)
    return master_storage / "chunks.jsonl"
