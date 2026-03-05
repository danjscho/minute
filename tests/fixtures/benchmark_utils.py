"""Benchmark metric computation helpers for SNOMED CT testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worker.snomed.ner.base import Entity


def compute_ner_metrics(
    predicted: list[Entity],
    gold: list[dict],
    mode: str = "exact",
) -> dict[str, float]:
    """Compute NER precision, recall and F1.

    Args:
        predicted: Entities produced by the NER service.
        gold: Ground-truth entity dicts with "text", "start", "end", "entity_type".
        mode: "exact" for exact span match, "partial" for overlapping span match.

    Returns:
        Dict with "precision", "recall", "f1".
    """
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if mode == "exact":
        pred_set = {(e.start, e.end, e.entity_type) for e in predicted}
        gold_set = {(g["start"], g["end"], g["entity_type"]) for g in gold}
        tp = len(pred_set & gold_set)
    else:
        # Partial: any overlap counts as a match
        tp = 0
        matched_gold: set[int] = set()
        for p in predicted:
            for i, g in enumerate(gold):
                if i in matched_gold:
                    continue
                if p.start < g["end"] and p.end > g["start"] and p.entity_type == g["entity_type"]:
                    tp += 1
                    matched_gold.add(i)
                    break

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_linking_accuracy(
    predictions: list[dict],
    at_k: int = 1,
) -> float:
    """Compute linking accuracy@k.

    Args:
        predictions: List of dicts with "gold_concept_id" and "predicted_concept_ids" (list).
        at_k: Consider a match if gold is in top-k predictions.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if not predictions:
        return 1.0

    correct = 0
    for pred in predictions:
        gold_id = pred["gold_concept_id"]
        predicted_ids = pred["predicted_concept_ids"][:at_k]
        if gold_id in predicted_ids:
            correct += 1

    return correct / len(predictions)
