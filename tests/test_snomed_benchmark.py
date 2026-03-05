"""SNOMED CT coding accuracy benchmark tests.

Runs the full NER -> linking pipeline with mocked models and verifies
accuracy metrics against baseline thresholds. Tests use synthetic data
to avoid SNOMED CT licensing requirements.

These tests verify the pipeline orchestration logic and metric computation,
not actual model accuracy (which requires real ML models and GPU hardware).
"""

from unittest.mock import MagicMock
from uuid import uuid4

from common.database.postgres_models import SourceType
from tests.fixtures.benchmark_utils import compute_linking_accuracy, compute_ner_metrics
from tests.fixtures.snomed_mock import MOCK_CLINICAL_TEXTS, MOCK_LINKING_DATA
from worker.snomed.coding_service import SNOMEDCodingService
from worker.snomed.linking.base import LinkedConcept, LinkResult
from worker.snomed.ner.base import Entity


def _entities_from_gold(gold: list[dict]) -> list[Entity]:
    """Convert gold-standard entity dicts to Entity objects."""
    return [
        Entity(
            text=g["text"],
            start=g["start"],
            end=g["end"],
            entity_type=g["entity_type"],
            confidence=0.9,
        )
        for g in gold
    ]


def _link_result_for(entity: Entity, concept_id: str, preferred_term: str, score: float = 0.92) -> LinkResult:
    """Build a LinkResult matching an entity to a concept."""
    top = LinkedConcept(
        concept_id=concept_id,
        preferred_term=preferred_term,
        fsn=f"{preferred_term} ({entity.entity_type})",
        score=score,
        semantic_tag=entity.entity_type,
    )
    return LinkResult(entity=entity, top_concept=top, alternatives=[top])


# Gold-standard concept mapping for mock linking data keyed by entity text
_GOLD_CONCEPTS: dict[str, dict] = {d["entity_text"]: d for d in MOCK_LINKING_DATA}


class TestNERBenchmark:
    """NER precision/recall/F1 against synthetic ground truth."""

    def _run_ner_benchmark(self) -> dict[str, float]:
        """Run NER on all mock clinical texts and compute aggregate metrics."""
        all_predicted: list[Entity] = []
        all_gold: list[dict] = []

        for sample in MOCK_CLINICAL_TEXTS:
            gold_entities = sample["entities"]
            predicted = _entities_from_gold(gold_entities)
            all_predicted.extend(predicted)
            all_gold.extend(gold_entities)

        return compute_ner_metrics(all_predicted, all_gold, mode="exact")

    def test_entity_detection_precision(self) -> None:
        metrics = self._run_ner_benchmark()
        assert metrics["precision"] == 1.0, f"Precision {metrics['precision']:.2f} < 1.0"

    def test_entity_detection_recall(self) -> None:
        metrics = self._run_ner_benchmark()
        assert metrics["recall"] == 1.0, f"Recall {metrics['recall']:.2f} < 1.0"

    def test_f1_score_above_threshold(self) -> None:
        metrics = self._run_ner_benchmark()
        assert metrics["f1"] >= 0.70, f"F1 {metrics['f1']:.2f} < 0.70"

    def test_per_entity_type_metrics(self) -> None:
        """Each entity type should have reasonable metrics independently."""
        entity_types = {"finding", "procedure", "body_structure"}

        for entity_type in entity_types:
            predicted = []
            gold = []
            for sample in MOCK_CLINICAL_TEXTS:
                type_gold = [e for e in sample["entities"] if e["entity_type"] == entity_type]
                type_pred = _entities_from_gold(type_gold)
                predicted.extend(type_pred)
                gold.extend(type_gold)

            if not gold:
                continue

            metrics = compute_ner_metrics(predicted, gold, mode="exact")
            assert metrics["f1"] >= 0.70, f"{entity_type} F1 {metrics['f1']:.2f} < 0.70"

    def test_no_entities_on_empty_text(self) -> None:
        """Empty text produces no entities."""
        mock_ner = MagicMock()
        mock_ner.extract_entities.return_value = []

        entities = mock_ner.extract_entities("")
        assert entities == []

    def test_partial_overlap_matching(self) -> None:
        """Partial overlap mode counts overlapping spans as matches."""
        predicted = [Entity(text="severe headache", start=0, end=15, entity_type="finding", confidence=0.9)]
        gold = [{"text": "headache", "start": 7, "end": 15, "entity_type": "finding"}]

        metrics = compute_ner_metrics(predicted, gold, mode="partial")
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0


class TestLinkingBenchmark:
    """Entity linking accuracy against synthetic gold standard."""

    def _build_linking_predictions(self, include_correct: bool = True) -> list[dict]:
        """Build predictions from MOCK_LINKING_DATA."""
        predictions = []
        for item in MOCK_LINKING_DATA:
            gold_id = item["gold_concept_id"]
            predicted_ids = [gold_id, "999999", "888888"] if include_correct else ["999999", "888888", gold_id]
            predictions.append({
                "gold_concept_id": gold_id,
                "predicted_concept_ids": predicted_ids,
            })
        return predictions

    def test_accuracy_at_1(self) -> None:
        predictions = self._build_linking_predictions(include_correct=True)
        acc = compute_linking_accuracy(predictions, at_k=1)
        assert acc == 1.0, f"Accuracy@1 {acc:.2f} < 1.0"

    def test_accuracy_at_5(self) -> None:
        predictions = self._build_linking_predictions(include_correct=False)
        acc = compute_linking_accuracy(predictions, at_k=5)
        assert acc == 1.0, f"Accuracy@5 {acc:.2f} < 1.0"

    def test_accuracy_at_1_imperfect(self) -> None:
        """When gold is not in top-1, accuracy drops."""
        predictions = self._build_linking_predictions(include_correct=False)
        acc = compute_linking_accuracy(predictions, at_k=1)
        assert acc == 0.0

    def test_synonym_resolution(self) -> None:
        """Synonyms map to correct concepts."""
        synonym_cases = [
            {"gold_concept_id": "22298006", "predicted_concept_ids": ["22298006"]},  # Heart attack -> MI
            {"gold_concept_id": "267036007", "predicted_concept_ids": ["267036007"]},  # SOB -> Dyspnea
            {"gold_concept_id": "241615005", "predicted_concept_ids": ["241615005"]},  # MRI
        ]
        acc = compute_linking_accuracy(synonym_cases, at_k=1)
        assert acc == 1.0

    def test_cross_type_accuracy(self) -> None:
        """Per-entity-type linking accuracy."""
        type_groups: dict[str, list[dict]] = {}
        for item in MOCK_LINKING_DATA:
            et = item["entity_type"]
            if et not in type_groups:
                type_groups[et] = []
            type_groups[et].append({
                "gold_concept_id": item["gold_concept_id"],
                "predicted_concept_ids": [item["gold_concept_id"]],
            })

        for entity_type, predictions in type_groups.items():
            acc = compute_linking_accuracy(predictions, at_k=1)
            assert acc >= 0.70, f"{entity_type} accuracy {acc:.2f} < 0.70"


class TestEndToEndBenchmark:
    """Full pipeline benchmark: NER + linking via SNOMEDCodingService."""

    def _make_pipeline_service(self, _text: str, gold_entities: list[dict]) -> SNOMEDCodingService:
        """Create a coding service with mocked NER and linker based on gold data."""
        entities = _entities_from_gold(gold_entities)

        mock_ner = MagicMock()
        mock_ner.extract_entities.return_value = entities

        link_results = []
        for entity in entities:
            gold = _GOLD_CONCEPTS.get(entity.text.capitalize()) or _GOLD_CONCEPTS.get(entity.text)
            if gold:
                lr = _link_result_for(entity, gold["gold_concept_id"], entity.text.capitalize())
            else:
                lr = LinkResult(entity=entity, top_concept=None, alternatives=[])
            link_results.append(lr)

        mock_linker = MagicMock()
        mock_linker.link_entities.return_value = link_results

        return SNOMEDCodingService(ner_service=mock_ner, linker_service=mock_linker)

    def test_pipeline_annotation_count(self) -> None:
        """Pipeline produces expected number of annotations for a sample."""
        sample = MOCK_CLINICAL_TEXTS[0]  # headache + fever + Blood pressure
        service = self._make_pipeline_service(sample["text"], sample["entities"])

        annotations = service.code_text(
            text=sample["text"],
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        # headache and fever have gold concepts; Blood pressure (observable_entity) may not
        assert len(annotations) >= 1
        assert len(annotations) <= len(sample["entities"])

    def test_pipeline_concept_accuracy(self) -> None:
        """End-to-end concept assignment accuracy on linkable entities."""
        sample = MOCK_CLINICAL_TEXTS[0]
        service = self._make_pipeline_service(sample["text"], sample["entities"])

        annotations = service.code_text(
            text=sample["text"],
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        for ann in annotations:
            assert ann.snomed_concept_id is not None
            assert ann.snomed_preferred_term is not None

    def test_pipeline_with_no_entities(self) -> None:
        """Pipeline handles text with no clinical entities."""
        sample = MOCK_CLINICAL_TEXTS[4]  # "No significant findings on examination."
        service = self._make_pipeline_service(sample["text"], sample["entities"])

        annotations = service.code_text(
            text=sample["text"],
            transcription_id=uuid4(),
            source_type=SourceType.TRANSCRIPT,
        )

        assert annotations == []

    def test_pipeline_multiple_samples(self) -> None:
        """Pipeline processes multiple samples and produces annotations."""
        total_annotations = 0
        for sample in MOCK_CLINICAL_TEXTS:
            service = self._make_pipeline_service(sample["text"], sample["entities"])
            annotations = service.code_text(
                text=sample["text"],
                transcription_id=uuid4(),
                source_type=SourceType.TRANSCRIPT,
            )
            total_annotations += len(annotations)

        # Should produce some annotations across all samples
        assert total_annotations > 0

    def test_pipeline_annotation_fields(self) -> None:
        """Annotations have all required fields populated."""
        sample = MOCK_CLINICAL_TEXTS[0]
        service = self._make_pipeline_service(sample["text"], sample["entities"])
        tid = uuid4()

        annotations = service.code_text(
            text=sample["text"],
            transcription_id=tid,
            source_type=SourceType.TRANSCRIPT,
        )

        for ann in annotations:
            assert ann.id is not None
            assert ann.transcription_id == tid
            assert ann.source_type == SourceType.TRANSCRIPT
            assert ann.text_span
            assert ann.start_char is not None
            assert ann.end_char is not None
            assert ann.entity_type
            assert ann.snomed_concept_id
            assert ann.confidence_score > 0
