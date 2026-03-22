"""
IRIS Path Utilities

Unified path handling for the IRIS project.
Sets up project paths and provides common path operations.
"""

import sys
from pathlib import Path


def setup_project_paths() -> Path:
    """
    Set up project Python paths.

    Adds project root and src directory to sys.path for imports.
    Should be called once at application startup.

    Returns:
        Path to project root directory
    """
    # Get project root (3 levels up from this file: src/common/path_utils.py)
    project_root = Path(__file__).parent.parent.parent.parent

    # Convert to absolute path
    project_root = project_root.resolve()

    # Add to sys.path if not already present
    project_root_str = str(project_root)
    src_str = str(project_root / "src")

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    if src_str not in sys.path:
        sys.path.insert(1, src_str)

    return project_root


# Auto-setup on import
PROJECT_ROOT = setup_project_paths()


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def get_src_dir() -> Path:
    """Get the src directory."""
    return PROJECT_ROOT / "src"


def get_config_dir() -> Path:
    """Get the configs directory."""
    return PROJECT_ROOT / "configs"


def get_data_dir() -> Path:
    """Get the data directory."""
    return PROJECT_ROOT / "data"


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path, relative_to: Path | None = None) -> Path:
    """
    Resolve a path to absolute path.

    Args:
        path: Path to resolve (can be relative or absolute)
        relative_to: Base path for relative paths (defaults to project root)

    Returns:
        Resolved absolute Path
    """
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    base = relative_to or PROJECT_ROOT
    return (base / path).resolve()
