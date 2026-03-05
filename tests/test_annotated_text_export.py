"""Tests for annotated text export (HTML and Markdown)."""

from uuid import uuid4

from common.database.postgres_models import SnomedAnnotation, SourceType
from worker.snomed.export.annotated_text import annotations_to_html, annotations_to_markdown


def _make_annotation(**kwargs) -> SnomedAnnotation:
    """Create a test SnomedAnnotation with sensible defaults."""
    defaults = {
        "id": uuid4(),
        "transcription_id": uuid4(),
        "source_type": SourceType.TRANSCRIPT,
        "text_span": "headache",
        "start_char": 22,
        "end_char": 30,
        "entity_type": "finding",
        "snomed_concept_id": "25064002",
        "snomed_preferred_term": "Headache",
        "snomed_fsn": "Headache (finding)",
        "confidence_score": 0.95,
        "is_verified": False,
    }
    defaults.update(kwargs)
    return SnomedAnnotation(**defaults)


SAMPLE_TEXT = "Patient presents with headache and fever today."


class TestHtmlExport:
    def test_empty_annotations(self) -> None:
        result = annotations_to_html(SAMPLE_TEXT, [])
        assert "<!DOCTYPE html>" in result
        assert "Patient presents with headache" in result
        assert "<mark" not in result

    def test_single_annotation_highlighted(self) -> None:
        ann = _make_annotation(start_char=22, end_char=30, text_span="headache")
        result = annotations_to_html(SAMPLE_TEXT, [ann])
        assert "<mark" in result
        assert 'data-snomed-id="25064002"' in result
        assert 'title="Headache [25064002]"' in result

    def test_entity_type_colour(self) -> None:
        ann = _make_annotation(entity_type="finding")
        result = annotations_to_html(SAMPLE_TEXT, [ann])
        assert "#dbeafe" in result  # blue-100 for finding

    def test_multiple_annotations(self) -> None:
        anns = [
            _make_annotation(start_char=22, end_char=30, text_span="headache"),
            _make_annotation(
                start_char=35,
                end_char=40,
                text_span="fever",
                snomed_concept_id="386661006",
                snomed_preferred_term="Fever",
            ),
        ]
        result = annotations_to_html(SAMPLE_TEXT, anns)
        assert result.count("<mark") == 2

    def test_summary_table_present(self) -> None:
        ann = _make_annotation()
        result = annotations_to_html(SAMPLE_TEXT, [ann])
        assert "Annotation Summary" in result
        assert "<table>" in result
        assert "25064002" in result

    def test_legend_present(self) -> None:
        result = annotations_to_html(SAMPLE_TEXT, [_make_annotation()])
        assert "Legend" in result
        assert "finding" in result

    def test_html_escaping(self) -> None:
        text = "Patient has <3 symptoms & more"
        result = annotations_to_html(text, [])
        assert "&lt;3" in result
        assert "&amp;" in result

    def test_text_before_and_after_annotation(self) -> None:
        ann = _make_annotation(start_char=22, end_char=30)
        result = annotations_to_html(SAMPLE_TEXT, [ann])
        assert "Patient presents with" in result
        assert "and fever today." in result


class TestMarkdownExport:
    def test_empty_annotations(self) -> None:
        result = annotations_to_markdown(SAMPLE_TEXT, [])
        assert SAMPLE_TEXT in result
        assert "**" not in result.split("---")[0]  # no bold in text part

    def test_single_annotation_inline(self) -> None:
        ann = _make_annotation(start_char=22, end_char=30, text_span="headache")
        result = annotations_to_markdown(SAMPLE_TEXT, [ann])
        assert "**headache** [SCTID:25064002]" in result

    def test_multiple_annotations(self) -> None:
        anns = [
            _make_annotation(start_char=22, end_char=30, text_span="headache"),
            _make_annotation(
                start_char=35,
                end_char=40,
                text_span="fever",
                snomed_concept_id="386661006",
            ),
        ]
        result = annotations_to_markdown(SAMPLE_TEXT, anns)
        assert "**headache** [SCTID:25064002]" in result
        assert "**fever** [SCTID:386661006]" in result

    def test_summary_table_present(self) -> None:
        ann = _make_annotation()
        result = annotations_to_markdown(SAMPLE_TEXT, [ann])
        assert "## Annotation Summary" in result
        assert "| Text | Type |" in result
        assert "25064002" in result

    def test_text_preserved_around_annotations(self) -> None:
        ann = _make_annotation(start_char=22, end_char=30)
        result = annotations_to_markdown(SAMPLE_TEXT, [ann])
        lines = result.split("---")[0]
        assert "Patient presents with" in lines
        assert "and fever today." in lines

    def test_annotations_without_offsets_excluded(self) -> None:
        ann = _make_annotation(start_char=None, end_char=None)
        result = annotations_to_markdown(SAMPLE_TEXT, [ann])
        # No inline annotation, but summary table still has it
        assert "**" not in result.split("---")[0]
        assert "25064002" in result
