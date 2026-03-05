"""
Evaluate NER model performance on annotated clinical text.

Computes precision, recall, and F1 score per entity type
using exact-match and partial-match (overlap) evaluation.

Usage:
    python scripts/evaluate_ner.py --test-data path/to/annotations.json
    python scripts/evaluate_ner.py --test-data path/to/annotations.json --model d4data/biomedical-ner-all
    python scripts/evaluate_ner.py --test-data path/to/annotations.json --output results.json
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
class EvalMetrics:
    """Evaluation metrics for a single entity type."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvalResult:
    """Complete evaluation result across all entity types."""

    per_type: dict[str, EvalMetrics] = field(default_factory=dict)
    overall: EvalMetrics = field(default_factory=EvalMetrics)

    def to_dict(self) -> dict:
        result = {
            "overall": {
                "precision": round(self.overall.precision, 4),
                "recall": round(self.overall.recall, 4),
                "f1": round(self.overall.f1, 4),
                "tp": self.overall.tp,
                "fp": self.overall.fp,
                "fn": self.overall.fn,
            },
            "per_type": {},
        }
        for entity_type, metrics in sorted(self.per_type.items()):
            result["per_type"][entity_type] = {
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
            }
        return result


def _spans_overlap(pred_start: int, pred_end: int, gold_start: int, gold_end: int) -> bool:
    """Check if two character spans overlap."""
    return pred_start < gold_end and pred_end > gold_start


def evaluate_exact_match(
    predicted: list[dict],
    gold: list[dict],
) -> EvalResult:
    """Evaluate NER predictions against gold annotations using exact span match.

    Args:
        predicted: List of predicted entities with keys: text, start, end, entity_type.
        gold: List of gold entities with the same keys.

    Returns:
        EvalResult with per-type and overall metrics.
    """
    result = EvalResult()
    type_metrics: dict[str, EvalMetrics] = defaultdict(EvalMetrics)

    matched_gold = set()

    for pred in predicted:
        pred_key = (pred["start"], pred["end"], pred["entity_type"])
        found_match = False

        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            gold_key = (g["start"], g["end"], g["entity_type"])
            if pred_key == gold_key:
                type_metrics[pred["entity_type"]].tp += 1
                result.overall.tp += 1
                matched_gold.add(i)
                found_match = True
                break

        if not found_match:
            type_metrics[pred["entity_type"]].fp += 1
            result.overall.fp += 1

    for i, g in enumerate(gold):
        if i not in matched_gold:
            type_metrics[g["entity_type"]].fn += 1
            result.overall.fn += 1

    result.per_type = dict(type_metrics)
    return result


def evaluate_partial_match(
    predicted: list[dict],
    gold: list[dict],
) -> EvalResult:
    """Evaluate NER predictions using partial span overlap.

    A prediction is a true positive if it overlaps with a gold entity
    of the same type. Each gold entity can only be matched once.

    Args:
        predicted: List of predicted entities.
        gold: List of gold entities.

    Returns:
        EvalResult with per-type and overall metrics.
    """
    result = EvalResult()
    type_metrics: dict[str, EvalMetrics] = defaultdict(EvalMetrics)

    matched_gold = set()

    for pred in predicted:
        found_match = False

        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if pred["entity_type"] != g["entity_type"]:
                continue
            if _spans_overlap(pred["start"], pred["end"], g["start"], g["end"]):
                type_metrics[pred["entity_type"]].tp += 1
                result.overall.tp += 1
                matched_gold.add(i)
                found_match = True
                break

        if not found_match:
            type_metrics[pred["entity_type"]].fp += 1
            result.overall.fp += 1

    for i, g in enumerate(gold):
        if i not in matched_gold:
            type_metrics[g["entity_type"]].fn += 1
            result.overall.fn += 1

    result.per_type = dict(type_metrics)
    return result


def _format_table(result: EvalResult, title: str) -> str:
    """Format evaluation result as a readable table."""
    lines = [f"\n{'=' * 60}", f"  {title}", f"{'=' * 60}"]
    lines.append(f"  {'Type':<25} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    lines.append(f"  {'-' * 73}")

    for entity_type in sorted(result.per_type):
        m = result.per_type[entity_type]
        lines.append(
            f"  {entity_type:<25} {m.precision:>8.4f} {m.recall:>8.4f} {m.f1:>8.4f} {m.tp:>6} {m.fp:>6} {m.fn:>6}"
        )

    lines.append(f"  {'-' * 73}")
    o = result.overall
    lines.append(f"  {'OVERALL':<25} {o.precision:>8.4f} {o.recall:>8.4f} {o.f1:>8.4f} {o.tp:>6} {o.fp:>6} {o.fn:>6}")
    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NER model on annotated clinical text")
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to JSON file with annotated test data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID (default: from settings)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence threshold (default: from settings)",
    )
    return parser.parse_args()


def load_test_data(path: str) -> list[dict]:
    """Load annotated test data from JSON.

    Expected format:
    [
        {
            "text": "Patient has headache and fever.",
            "entities": [
                {"text": "headache", "start": 12, "end": 20, "entity_type": "finding"},
                {"text": "fever", "start": 25, "end": 30, "entity_type": "finding"}
            ]
        }
    ]
    """
    test_path = Path(path)
    with test_path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    # Load test data
    logger.info("Loading test data from %s", args.test_data)
    test_data = load_test_data(args.test_data)
    logger.info("Loaded %d test samples", len(test_data))

    # Import and initialise NER service
    from worker.snomed.ner.biomedbert_ner import BiomedBERTNERService

    kwargs: dict = {"device": args.device}
    if args.model:
        kwargs["model_name"] = args.model
    if args.confidence_threshold is not None:
        kwargs["confidence_threshold"] = args.confidence_threshold

    ner_service = BiomedBERTNERService(**kwargs)

    # Run evaluation
    all_predicted: list[dict] = []
    all_gold: list[dict] = []

    for i, sample in enumerate(test_data):
        text = sample["text"]
        gold_entities = sample["entities"]

        entities = ner_service.extract_entities(text)
        predicted = [{"text": e.text, "start": e.start, "end": e.end, "entity_type": e.entity_type} for e in entities]

        all_predicted.extend(predicted)
        all_gold.extend(gold_entities)

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d samples", i + 1, len(test_data))

    logger.info("Processed all %d samples", len(test_data))

    # Compute metrics
    exact_result = evaluate_exact_match(all_predicted, all_gold)
    partial_result = evaluate_partial_match(all_predicted, all_gold)

    # Print results
    print(_format_table(exact_result, "Exact Match Evaluation"))  # noqa: T201
    print(_format_table(partial_result, "Partial Match (Overlap) Evaluation"))  # noqa: T201

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "exact_match": exact_result.to_dict(),
            "partial_match": partial_result.to_dict(),
            "config": {
                "model": ner_service.model_name,
                "confidence_threshold": ner_service.confidence_threshold,
                "test_data": args.test_data,
                "num_samples": len(test_data),
            },
        }
        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
