#!/usr/bin/env python3
"""
Download pre-built SNOMED CT embedding index.

This script downloads pre-built FAISS indices for SNOMED CT concepts.

NOTE: Pre-built indices contain embeddings of SNOMED CT terms which are
subject to SNOMED International licensing terms. Users must have a valid
SNOMED CT license to use these indices.

Usage:
    python scripts/download_snomed_index.py --list    # List available indices
    python scripts/download_snomed_index.py --version 20240101

For organizations without internet access or with specific compliance
requirements, indices can be built locally using:
    python scripts/setup_snomed.py --rf2-archive /path/to/snomed.zip
    python scripts/build_snomed_index.py
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Index registry (placeholder - in production this would be a remote manifest)
# Due to SNOMED CT licensing, we cannot host pre-built indices publicly
INDEX_REGISTRY: dict[str, dict] = {
    # Example entries - actual URLs would require SNOMED license verification
    # "20240101_uk": {
    #     "version": "20240101",
    #     "edition": "UK Clinical",
    #     "index_url": "https://example.com/snomed/index_20240101_uk.faiss",
    #     "ids_url": "https://example.com/snomed/index_20240101_uk.ids.json",
    #     "concepts_url": "https://example.com/snomed/concepts_20240101_uk.json",
    #     "checksum": "sha256:abc123...",
    #     "n_concepts": 200000,
    #     "size_mb": 150,
    # }
}


def list_available_indices() -> None:
    """List available pre-built indices."""
    if not INDEX_REGISTRY:
        print("No pre-built indices available.")  # noqa: T201
        print()  # noqa: T201
        print("Due to SNOMED CT licensing requirements, pre-built indices cannot")  # noqa: T201
        print("be distributed publicly. Please build indices locally:")  # noqa: T201
        print()  # noqa: T201
        print("  1. Obtain SNOMED CT RF2 release from:")  # noqa: T201
        print("     - NHS Digital (UK): https://isd.digital.nhs.uk/trud")  # noqa: T201
        print("     - SNOMED International: https://www.snomed.org/")  # noqa: T201
        print()  # noqa: T201
        print("  2. Run the setup and build scripts:")  # noqa: T201
        print("     python scripts/setup_snomed.py --rf2-archive /path/to/snomed.zip")  # noqa: T201
        print("     python scripts/build_snomed_index.py")  # noqa: T201
        return

    print("Available pre-built indices:")  # noqa: T201
    print()  # noqa: T201
    for key, info in INDEX_REGISTRY.items():
        print(f"  {key}:")  # noqa: T201
        print(f"    Version: {info['version']}")  # noqa: T201
        print(f"    Edition: {info['edition']}")  # noqa: T201
        print(f"    Concepts: {info['n_concepts']:,}")  # noqa: T201
        print(f"    Size: {info['size_mb']} MB")  # noqa: T201
        print()  # noqa: T201


def verify_checksum(filepath: Path, expected: str) -> bool:
    """Verify file checksum."""
    algo, expected_hash = expected.split(":", 1)
    if algo != "sha256":
        logger.warning("Unknown checksum algorithm: %s", algo)
        return True  # Skip verification

    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()
    if actual_hash != expected_hash:
        logger.error("Checksum mismatch: expected %s, got %s", expected_hash, actual_hash)
        return False
    return True


def download_with_progress(url: str, filepath: Path) -> None:
    """Download file with progress reporting."""
    logger.info("Downloading %s", url)

    def progress_hook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            print(f"\r  Progress: {percent}%", end="", flush=True)  # noqa: T201

    urlretrieve(url, filepath, reporthook=progress_hook)  # noqa: S310
    print()  # newline after progress  # noqa: T201


def download_index(version: str, output_dir: Path) -> bool:
    """
    Download pre-built index.

    Args:
        version: Index version key
        output_dir: Output directory

    Returns:
        True if successful
    """
    if version not in INDEX_REGISTRY:
        logger.error("Unknown index version: %s", version)
        logger.error("Available versions: %s", ", ".join(INDEX_REGISTRY.keys()) or "none")
        return False

    info = INDEX_REGISTRY[version]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download FAISS index
        index_path = output_dir / "faiss.index"
        download_with_progress(info["index_url"], index_path)

        # Download concept IDs
        ids_path = output_dir / "faiss.index.ids.json"
        download_with_progress(info["ids_url"], ids_path)

        # Download concepts database (optional)
        if "concepts_url" in info:
            concepts_path = output_dir.parent / "concept_db.json"
            download_with_progress(info["concepts_url"], concepts_path)

        # Verify checksum
        if "checksum" in info and not verify_checksum(index_path, info["checksum"]):
            logger.error("Checksum verification failed")
            return False

        # Write download manifest
        manifest = {
            "downloaded_from": info["index_url"],
            "version": info["version"],
            "edition": info["edition"],
            "n_concepts": info["n_concepts"],
        }
        manifest_path = output_dir / "download_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Successfully downloaded index to %s", output_dir)
        return True

    except Exception as e:
        logger.exception("Download failed: %s", e)
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download pre-built SNOMED CT embedding index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available pre-built indices",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Index version to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: from settings)",
    )

    args = parser.parse_args()

    if args.list:
        list_available_indices()
        return 0

    if not args.version:
        parser.error("--version is required (or use --list to see available versions)")

    # Get output directory
    settings = get_settings()
    output_dir = args.output_dir or Path(settings.SNOMED_FAISS_INDEX_PATH).parent

    # Download
    success = download_index(args.version, output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
