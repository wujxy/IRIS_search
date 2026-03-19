#!/usr/bin/env python3
"""
Migrate IRIS metadata.json files to SQLite database.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src to path for new structure
sys.path.insert(1, str(Path(__file__).parent.parent / "src"))

from utils.helpers import load_config, setup_logging
from services.paper_service import PaperService


logger = logging.getLogger(__name__)


def migrate_database(database_root: str, db_path: str):
    """
    Migrate all metadata.json files to SQLite.

    Args:
        database_root: Path to IRIS database root
        db_path: Path to SQLite database file
    """
    database_root = Path(database_root)

    # Initialize paper service
    paper_service = PaperService(db_path)

    migrated_count = 0
    skipped_count = 0
    error_count = 0
    total_papers_processed = 0

    print(f"Migrating papers from: {database_root}")
    print(f"To database: {db_path}\n")

    # Process each update folder
    update_folders = sorted(database_root.glob("update_*"))

    if not update_folders:
        print("No update folders found.")
        return

    for update_folder in update_folders:
        metadata_file = update_folder / "metadata.json"

        if not metadata_file.exists():
            print(f"Skipping {update_folder.name}: no metadata.json")
            skipped_count += 1
            continue

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get update folder name
            folder_name = update_folder.name

            # Process all papers
            all_papers = data.get("all_papers", [])

            for paper in all_papers:
                total_papers_processed += 1

                # Extract paper_id from entry_id
                entry_id = paper.get('entry_id', '')
                paper_id = entry_id.split('/')[-1].split('v')[0] if entry_id else ''

                # Prepare paper data
                paper_data = {
                    'entry_id': entry_id,
                    'paper_id': paper_id,
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', []),
                    'published': paper.get('published'),
                    'updated': paper.get('updated'),
                    'summary': paper.get('summary', ''),
                    'comment': paper.get('comment'),
                    'journal_ref': paper.get('journal_ref'),
                    'doi': paper.get('doi'),
                    'primary_category': paper.get('primary_category'),
                    'categories': paper.get('categories', []),
                    'status': paper.get('status'),
                    'download_status': paper.get('download_status'),
                    'pdf_url': paper.get('pdf_url'),
                    'pdf_path': paper.get('pdf_path'),
                    'update_folder': folder_name
                }

                # Add to database
                if paper_service.add_paper(paper_data):
                    migrated_count += 1
                else:
                    # Likely duplicate (entry_id unique constraint)
                    skipped_count += 1

            print(f"  Processed {folder_name}: {len(all_papers)} papers")

        except json.JSONDecodeError as e:
            print(f"Error reading {metadata_file}: {e}")
            error_count += 1
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            error_count += 1

    # Get final stats
    stats = paper_service.get_stats()

    print(f"\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total papers in database: {stats['total']}")
    print(f"  - New: {stats.get('new', 0)}")
    print(f"  - Duplicate: {stats.get('duplicate', 0)}")
    print(f"  - Review: {stats.get('review', 0)}")
    print(f"\nMigration:")
    print(f"  - Papers processed: {total_papers_processed}")
    print(f"  - Migrated: {migrated_count}")
    print(f"  - Skipped (duplicate): {skipped_count}")
    print(f"  - Errors: {error_count}")


def dry_run(database_root: str):
    """
    Show what would be migrated without actually migrating.

    Args:
        database_root: Path to IRIS database root
    """
    database_root = Path(database_root)

    print(f"DRY RUN - Showing what would be migrated from: {database_root}\n")

    update_folders = sorted(database_root.glob("update_*"))

    if not update_folders:
        print("No update folders found.")
        return

    total_papers = 0
    for update_folder in update_folders:
        metadata_file = update_folder / "metadata.json"

        if not metadata_file.exists():
            continue

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_papers = data.get("all_papers", [])
                total_papers += len(all_papers)
                print(f"{update_folder.name}: {len(all_papers)} papers")
        except:
            pass

    print(f"\nTotal papers to migrate: {total_papers}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate IRIS metadata to SQLite database"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to IRIS database (overrides config)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database file (overrides config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actual migration"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        log_dir=config["logging"]["log_dir"],
        level=args.log_level
    )

    # Get paths
    database_root = args.database or config["storage"]["database_root"]
    db_path = args.db_path or config["storage"].get("paper_db_path",
        str(Path(database_root) / "iris_papers.db"))

    if args.dry_run:
        dry_run(database_root)
    else:
        migrate_database(database_root, db_path)


if __name__ == "__main__":
    main()
