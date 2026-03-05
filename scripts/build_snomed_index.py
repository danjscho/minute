#!/usr/bin/env python3
"""
Build SNOMED CT embedding index.

This script:
1. Loads the concept database created by setup_snomed.py
2. Generates embeddings for all concepts using SapBERT
3. Builds a FAISS index for similarity search
4. Saves the index to disk

Usage:
    python scripts/build_snomed_index.py

Prerequisites:
    - SNOMED CT concept database (run setup_snomed.py first)
    - GPU recommended for faster embedding generation
    - ~4GB GPU memory for SapBERT model

Expected runtime: 2-4 hours on RTX 3080 for ~200k concepts
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Time format thresholds (in seconds)
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    elif seconds < SECONDS_PER_HOUR:
        return f"{seconds / SECONDS_PER_MINUTE:.1f}m"
    else:
        return f"{seconds / SECONDS_PER_HOUR:.1f}h"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build SNOMED CT embedding index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--concepts-path",
        type=Path,
        default=None,
        help="Path to concept_db.json (default: from settings)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for FAISS index (default: from settings)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model for embeddings (default: from settings)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cuda, cpu, mps (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="auto",
        choices=["auto", "flat", "ivf"],
        help="FAISS index type (default: auto)",
    )
    parser.add_argument(
        "--use-all-terms",
        action="store_true",
        default=True,
        help="Average embeddings of all terms per concept (default)",
    )
    parser.add_argument(
        "--preferred-term-only",
        action="store_false",
        dest="use_all_terms",
        help="Only use preferred term for embeddings",
    )

    return parser.parse_args()


def save_manifest(
    output_path: Path,
    concepts_path: Path,
    generator: Any,
    args: argparse.Namespace,
    concept_ids: list,
    embeddings: Any,
    embed_time: float,
    index_time: float,
    total_time: float,
) -> Path:
    """Save build manifest to disk."""
    manifest = {
        "build_date": datetime.now(tz=UTC).isoformat(),
        "source_concepts": str(concepts_path),
        "embedding_model": generator.model_name,
        "device": generator.device,
        "batch_size": generator.batch_size,
        "use_all_terms": args.use_all_terms,
        "index_type": args.index_type,
        "n_embeddings": len(concept_ids),
        "embedding_dim": embeddings.shape[1],
        "embed_time_seconds": embed_time,
        "index_time_seconds": index_time,
        "total_time_seconds": total_time,
    }

    manifest_path = output_path.parent / "index_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def log_completion_summary(
    concept_ids: list,
    embed_time: float,
    index_time: float,
    total_time: float,
    output_path: Path,
    manifest_path: Path,
) -> None:
    """Log build completion summary."""
    logger.info("Build complete!")
    logger.info("  Embeddings: %d", len(concept_ids))
    logger.info("  Embedding time: %s", format_time(embed_time))
    logger.info("  Index build time: %s", format_time(index_time))
    logger.info("  Total time: %s", format_time(total_time))
    logger.info("  Index saved to: %s", output_path)
    logger.info("  Manifest saved to: %s", manifest_path)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Get settings
    settings = get_settings()

    # Resolve paths
    concepts_path = args.concepts_path or Path(settings.SNOMED_CT_DATA_PATH) / "concept_db.json"
    output_path = args.output_path or Path(settings.SNOMED_FAISS_INDEX_PATH)

    # Verify input exists
    if not concepts_path.exists():
        logger.error("Concept database not found: %s", concepts_path)
        logger.error("Run setup_snomed.py first to create the concept database.")
        return 1

    # Import here to avoid slow startup if just checking --help
    from worker.snomed.data.concept_db import ConceptDatabase
    from worker.snomed.embeddings.faiss_index import SNOMEDFaissIndex
    from worker.snomed.embeddings.generator import SNOMEDEmbeddingGenerator

    start_time = time.time()

    # Load concept database
    logger.info("Loading concept database from %s", concepts_path)
    concept_db = ConceptDatabase.load(concepts_path)
    stats = concept_db.stats()
    logger.info("Loaded %d concepts", stats["total_concepts"])

    # Initialize embedding generator
    logger.info("Initializing embedding generator...")
    generator = SNOMEDEmbeddingGenerator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Generate embeddings
    logger.info("Generating embeddings (this may take a while)...")
    embed_start = time.time()

    embeddings, concept_ids = generator.embed_concepts(
        concept_db.concepts,
        use_all_terms=args.use_all_terms,
        show_progress=True,
    )

    embed_time = time.time() - embed_start
    logger.info("Generated %d embeddings in %s", len(concept_ids), format_time(embed_time))
    logger.info("Embedding shape: %s", embeddings.shape)

    # Build FAISS index
    logger.info("Building FAISS index...")
    index_start = time.time()

    faiss_index = SNOMEDFaissIndex(output_path)
    faiss_index.build(embeddings, concept_ids, index_type=args.index_type)

    index_time = time.time() - index_start
    logger.info("Built FAISS index in %s", format_time(index_time))

    # Save index
    logger.info("Saving index to %s", output_path)
    faiss_index.save()

    # Save build manifest
    total_time = time.time() - start_time
    manifest_path = save_manifest(
        output_path,
        concepts_path,
        generator,
        args,
        concept_ids,
        embeddings,
        embed_time,
        index_time,
        total_time,
    )

    log_completion_summary(concept_ids, embed_time, index_time, total_time, output_path, manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
