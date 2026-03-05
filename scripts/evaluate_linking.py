"""
Evaluate entity linking performance on annotated data.

Computes Accuracy@1 and Accuracy@K per entity type.

Usage:
    python scripts/evaluate_linking.py --test-data path/to/linking_gold.json
    python scripts/evaluate_linking.py --test-data path/to/linking_gold.json --top-k 5 --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for entity linking."""

    total: int = 0
    correct_at_1: int = 0
    correct_at_k: int = 0

    @property
    def accuracy_at_1(self) -> float:
        return self.correct_at_1 / self.total if self.total > 0 else 0.0

    @property
    def accuracy_at_k(self) -> float:
        return self.correct_at_k / self.total if self.total > 0 else 0.0


@dataclass
class LinkingEvalResult:
    """Complete linking evaluation result."""

    per_type: dict[str, AccuracyMetrics] = field(default_factory=dict)
    overall: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    top_k: int = 5

    def to_dict(self) -> dict:
        result = {
            "overall": {
                "accuracy_at_1": round(self.overall.accuracy_at_1, 4),
                "accuracy_at_k": round(self.overall.accuracy_at_k, 4),
                "total": self.overall.total,
                "correct_at_1": self.overall.correct_at_1,
                "correct_at_k": self.overall.correct_at_k,
                "top_k": self.top_k,
            },
            "per_type": {},
        }
        for entity_type, metrics in sorted(self.per_type.items()):
            result["per_type"][entity_type] = {
                "accuracy_at_1": round(metrics.accuracy_at_1, 4),
                "accuracy_at_k": round(metrics.accuracy_at_k, 4),
                "total": metrics.total,
                "correct_at_1": metrics.correct_at_1,
                "correct_at_k": metrics.correct_at_k,
            }
        return result


def evaluate_linking(
    predictions: list[dict],
    gold: list[dict],
    top_k: int = 5,
) -> LinkingEvalResult:
    """Evaluate linking predictions against gold SNOMED concept IDs.

    Args:
        predictions: List of dicts with keys: entity_text, entity_type,
                     predicted_concept_id, alternative_concept_ids.
        gold: List of dicts with keys: entity_text, entity_type, gold_concept_id.
        top_k: K for Accuracy@K metric.

    Returns:
        LinkingEvalResult with per-type and overall accuracy.
    """
    result = LinkingEvalResult(top_k=top_k)
    type_metrics: dict[str, AccuracyMetrics] = defaultdict(AccuracyMetrics)

    for pred, g in zip(predictions, gold, strict=False):
        entity_type = g.get("entity_type", "unknown")
        gold_id = g["gold_concept_id"]

        type_metrics[entity_type].total += 1
        result.overall.total += 1

        # Check Accuracy@1
        if pred.get("predicted_concept_id") == gold_id:
            type_metrics[entity_type].correct_at_1 += 1
            result.overall.correct_at_1 += 1

        # Check Accuracy@K
        all_ids = [pred.get("predicted_concept_id", ""), *pred.get("alternative_concept_ids", [])]
        if gold_id in all_ids[:top_k]:
            type_metrics[entity_type].correct_at_k += 1
            result.overall.correct_at_k += 1

    result.per_type = dict(type_metrics)
    return result


def _format_table(result: LinkingEvalResult) -> str:
    """Format evaluation result as a readable table."""
    lines = [
        f"\n{'=' * 65}",
        f"  Entity Linking Evaluation (top_k={result.top_k})",
        f"{'=' * 65}",
        f"  {'Type':<25} {'Acc@1':>10} {'Acc@K':>10} {'Total':>8}",
        f"  {'-' * 58}",
    ]

    for entity_type in sorted(result.per_type):
        m = result.per_type[entity_type]
        lines.append(f"  {entity_type:<25} {m.accuracy_at_1:>10.4f} {m.accuracy_at_k:>10.4f} {m.total:>8}")

    lines.append(f"  {'-' * 58}")
    o = result.overall
    lines.append(f"  {'OVERALL':<25} {o.accuracy_at_1:>10.4f} {o.accuracy_at_k:>10.4f} {o.total:>8}")
    lines.append(f"{'=' * 65}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate entity linking on annotated data")
    parser.add_argument("--test-data", type=str, required=True, help="Path to JSON file with gold annotations")
    parser.add_argument("--top-k", type=int, default=5, help="K for Accuracy@K (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results as JSON")
    parser.add_argument("--device", type=str, default="auto", help="Device for inference")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Minimum confidence threshold")
    return parser.parse_args()


def load_test_data(path: str) -> list[dict]:
    """Load annotated test data from JSON.

    Expected format:
    [
        {
            "entity_text": "headache",
            "entity_type": "finding",
            "gold_concept_id": "25064002"
        }
    ]
    """
    test_path = Path(path)
    with test_path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    logger.info("Loading test data from %s", args.test_data)
    test_data = load_test_data(args.test_data)
    logger.info("Loaded %d test samples", len(test_data))

    # Import and initialise linker
    from worker.snomed.linking.sapbert_linker import SapBERTLinkerService
    from worker.snomed.ner.base import Entity

    kwargs: dict = {}
    if args.confidence_threshold is not None:
        kwargs["confidence_threshold"] = args.confidence_threshold

    linker = SapBERTLinkerService(**kwargs)

    # Run linking
    predictions: list[dict] = []
    for i, sample in enumerate(test_data):
        entity = Entity(
            text=sample["entity_text"],
            start=0,
            end=len(sample["entity_text"]),
            entity_type=sample.get("entity_type", ""),
            confidence=1.0,
        )

        result = linker.link_entity(entity, top_k=args.top_k)

        pred = {
            "entity_text": sample["entity_text"],
            "entity_type": sample.get("entity_type", ""),
            "predicted_concept_id": result.top_concept.concept_id if result.top_concept else "",
            "alternative_concept_ids": [c.concept_id for c in result.alternatives],
        }
        predictions.append(pred)

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d samples", i + 1, len(test_data))

    logger.info("Processed all %d samples", len(test_data))

    # Compute metrics
    eval_result = evaluate_linking(predictions, test_data, top_k=args.top_k)
    print(_format_table(eval_result))  # noqa: T201

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "results": eval_result.to_dict(),
            "config": {
                "test_data": args.test_data,
                "top_k": args.top_k,
                "num_samples": len(test_data),
            },
        }
        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
