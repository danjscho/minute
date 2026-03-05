#!/usr/bin/env python3
"""
Setup script for SNOMED CT data.

This script:
1. Extracts SNOMED CT RF2 release archive
2. Parses concepts, descriptions, and relationships
3. Filters to relevant hierarchies (Finding, Procedure, Body Structure)
4. Builds and saves concept database

Usage:
    python scripts/setup_snomed.py --rf2-archive /path/to/SnomedCT_*.zip

Prerequisites:
    - SNOMED CT RF2 release archive (requires license from SNOMED International or NHS Digital)
    - Sufficient disk space (~5GB for extracted files, ~100MB for concept database)
"""

import argparse
import json
import logging
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.settings import get_settings
from worker.snomed.data.concept_db import ConceptDatabase
from worker.snomed.data.rf2_parser import RF2Parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_rf2_archive(archive_path: Path, output_dir: Path) -> Path:
    """
    Extract RF2 archive to output directory.

    Args:
        archive_path: Path to ZIP archive
        output_dir: Directory to extract to

    Returns:
        Path to extracted directory
    """
    logger.info("Extracting %s to %s", archive_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)

    # Find the extracted directory (usually named like SnomedCT_...)
    extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if extracted_dirs:
        return extracted_dirs[0]
    return output_dir


# SNOMED CT version date format length (YYYYMMDD)
VERSION_DATE_LENGTH = 8


def _extract_version_from_part(part: str) -> str | None:
    """Extract version date from a string part if it matches expected format."""
    if part.startswith("20") and len(part) >= VERSION_DATE_LENGTH:
        return part[:VERSION_DATE_LENGTH]
    return None


def detect_snomed_version(rf2_path: Path) -> str:
    """
    Try to detect SNOMED CT version from directory/file names.

    Args:
        rf2_path: Path to RF2 data

    Returns:
        Version string or "unknown"
    """
    # Look for version in directory name (e.g., SnomedCT_UKClinicalRF2_PRODUCTION_20240101T000001Z)
    name = rf2_path.name
    for part in name.split("_"):
        version = _extract_version_from_part(part)
        if version:
            return version

    # Look in release information files
    for pattern in ["sct2_Concept_*.txt", "sct2_Description_*.txt"]:
        for f in rf2_path.rglob(pattern):
            # Version often in filename like sct2_Concept_Snapshot_GB1000000_20240101.txt
            parts = f.stem.split("_")
            for part in parts:
                version = _extract_version_from_part(part)
                if version:
                    return version

    return "unknown"


def write_manifest(output_dir: Path, stats: dict, version: str, source: str) -> None:
    """
    Write manifest file with setup information.

    Args:
        output_dir: Output directory
        stats: Database statistics
        version: SNOMED CT version
        source: Source description
    """
    manifest = {
        "setup_date": datetime.now(tz=UTC).isoformat(),
        "snomed_version": version,
        "source": source,
        "statistics": stats,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote manifest to %s", manifest_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup SNOMED CT data from RF2 release archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rf2-archive",
        type=Path,
        help="Path to SNOMED CT RF2 release archive (ZIP file)",
    )
    parser.add_argument(
        "--rf2-path",
        type=Path,
        help="Path to already-extracted RF2 directory (alternative to --rf2-archive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for concept database (default: from settings)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip archive extraction (use if RF2 files already extracted)",
    )
    parser.add_argument(
        "--filter-hierarchies",
        action="store_true",
        default=True,
        help="Filter to clinical hierarchies (Finding, Procedure, Body Structure)",
    )
    parser.add_argument(
        "--no-filter-hierarchies",
        action="store_false",
        dest="filter_hierarchies",
        help="Include all SNOMED CT concepts (larger database)",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int | None:
    """Validate command line arguments. Returns error code or None if valid."""
    if not args.rf2_archive and not args.rf2_path:
        parser.error("Either --rf2-archive or --rf2-path must be specified")

    if args.rf2_archive and not args.rf2_archive.exists():
        logger.error("RF2 archive not found: %s", args.rf2_archive)
        return 1

    if args.rf2_path and not args.rf2_path.exists():
        logger.error("RF2 path not found: %s", args.rf2_path)
        return 1

    return None


def resolve_rf2_path(args: argparse.Namespace, output_dir: Path) -> Path | None:
    """Resolve the RF2 data path from arguments. Returns None on error."""
    if args.rf2_archive and not args.skip_extraction:
        return extract_rf2_archive(args.rf2_archive, output_dir / "rf2")
    elif args.rf2_path:
        return args.rf2_path
    else:
        # Assume extraction already done
        rf2_path = output_dir / "rf2"
        if not rf2_path.exists():
            logger.error("RF2 directory not found: %s", rf2_path)
            return None
        return rf2_path


def log_statistics(stats: dict) -> None:
    """Log database statistics."""
    logger.info("Database statistics:")
    logger.info("  Total concepts: %d", stats["total_concepts"])
    logger.info("  Indexed terms: %d", stats["indexed_terms"])
    logger.info("  Concepts by semantic tag:")
    for tag, count in sorted(stats["concepts_by_semantic_tag"].items()):
        logger.info("    %s: %d", tag, count)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Create parser for error handling (parse_args uses it internally)
    parser = argparse.ArgumentParser()
    error = validate_args(args, parser)
    if error is not None:
        return error

    # Get settings
    settings = get_settings()
    output_dir = args.output_dir or Path(settings.SNOMED_CT_DATA_PATH)

    # Determine RF2 data path
    rf2_path = resolve_rf2_path(args, output_dir)
    if rf2_path is None:
        return 1

    # Detect version
    version = detect_snomed_version(rf2_path)
    source = f"SNOMED CT (parsed from {rf2_path.name})"
    logger.info("Detected SNOMED CT version: %s", version)

    # Parse concepts
    logger.info("Parsing SNOMED CT RF2 files...")
    rf2_parser = RF2Parser(rf2_path)
    concepts = rf2_parser.parse_concepts()

    if not concepts:
        logger.error("No concepts parsed from RF2 files")
        return 1

    logger.info("Parsed %d total concepts", len(concepts))

    # Filter to relevant hierarchies
    if args.filter_hierarchies:
        logger.info("Filtering to clinical hierarchies...")
        concepts = rf2_parser.filter_by_hierarchies(concepts)
        concepts = rf2_parser.filter_by_semantic_tags(concepts)

    # Build concept database
    logger.info("Building concept database...")
    concept_db = ConceptDatabase(concepts)
    concept_db.set_metadata(version=version, source=source)

    # Save database
    db_path = output_dir / "concept_db.json"
    concept_db.save(db_path)

    # Log statistics and write manifest
    stats = concept_db.stats()
    log_statistics(stats)
    write_manifest(output_dir, stats, version, source)

    logger.info("Setup complete. Concept database saved to %s", db_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
