"""Annotated text export for SNOMED CT annotations.

Generates HTML and Markdown representations of transcription text
with SNOMED concepts highlighted inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.database.postgres_models import SnomedAnnotation

# Entity type to CSS colour mapping (matches frontend SnomedAnnotationCard)
ENTITY_COLOURS: dict[str, str] = {
    "finding": "#dbeafe",  # blue-100
    "procedure": "#dcfce7",  # green-100
    "body_structure": "#ffedd5",  # orange-100
    "disorder": "#fee2e2",  # red-100
    "substance": "#f3e8ff",  # purple-100
    "observable_entity": "#ccfbf1",  # teal-100
}

DEFAULT_COLOUR = "#f3f4f6"  # gray-100


def _sort_annotations(annotations: list[SnomedAnnotation]) -> list[SnomedAnnotation]:
    """Sort annotations by start_char, filtering out those without offsets."""
    return sorted(
        [a for a in annotations if a.start_char is not None and a.end_char is not None],
        key=lambda a: a.start_char,
    )


def annotations_to_html(text: str, annotations: list[SnomedAnnotation]) -> str:
    """Generate a self-contained HTML document with highlighted SNOMED annotations.

    Args:
        text: The full transcript text (dialogue entries joined by space).
        annotations: SNOMED annotations with start_char/end_char offsets.

    Returns:
        Complete HTML document string.
    """
    sorted_anns = _sort_annotations(annotations)
    body_parts: list[str] = []
    cursor = 0

    for ann in sorted_anns:
        start = ann.start_char
        end = ann.end_char

        if start > cursor:
            body_parts.append(_html_escape(text[cursor:start]))

        colour = ENTITY_COLOURS.get(ann.entity_type or "", DEFAULT_COLOUR)
        concept_id = ann.snomed_concept_id or "unknown"
        preferred_term = _html_escape(ann.snomed_preferred_term or "")
        span_text = _html_escape(text[start:end])

        body_parts.append(
            f'<mark style="background-color: {colour}; padding: 1px 3px; border-radius: 3px;" '
            f'data-snomed-id="{concept_id}" '
            f'title="{preferred_term} [{concept_id}]">'
            f"{span_text}</mark>"
        )
        cursor = end

    if cursor < len(text):
        body_parts.append(_html_escape(text[cursor:]))

    annotated_body = "".join(body_parts)

    legend_items = []
    for entity_type, colour in ENTITY_COLOURS.items():
        legend_items.append(
            f'<span style="background-color: {colour}; padding: 2px 8px; border-radius: 3px; '
            f'margin-right: 8px; font-size: 0.85em;">{entity_type.replace("_", " ")}</span>'
        )
    legend = "".join(legend_items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SNOMED CT Annotated Transcript</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 2em auto; line-height: 1.6; color: #333; }}
.legend {{ margin-bottom: 1.5em; padding: 1em; background: #f9fafb; border-radius: 6px; }}
.legend h3 {{ margin: 0 0 0.5em; font-size: 0.9em; color: #6b7280; }}
.content {{ white-space: pre-wrap; }}
.summary {{ margin-top: 2em; border-top: 1px solid #e5e7eb; padding-top: 1em; }}
.summary table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
.summary th, .summary td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }}
.summary th {{ background: #f9fafb; }}
</style>
</head>
<body>
<h1>SNOMED CT Annotated Transcript</h1>
<div class="legend"><h3>Legend</h3>{legend}</div>
<div class="content">{annotated_body}</div>
<div class="summary">
<h2>Annotation Summary</h2>
{_build_html_summary_table(annotations)}
</div>
</body>
</html>"""


def _build_html_summary_table(annotations: list[SnomedAnnotation]) -> str:
    """Build an HTML summary table of all annotations."""
    rows = []
    for ann in annotations:
        rows.append(
            f"<tr>"
            f"<td>{_html_escape(ann.text_span)}</td>"
            f"<td>{_html_escape(ann.entity_type or '')}</td>"
            f"<td>{ann.snomed_concept_id or ''}</td>"
            f"<td>{_html_escape(ann.snomed_preferred_term or '')}</td>"
            f"<td>{ann.confidence_score:.0%}</td>"
            f"<td>{'Yes' if ann.is_verified else 'No'}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Text</th><th>Type</th><th>SNOMED ID</th>"
        "<th>Preferred Term</th><th>Confidence</th><th>Verified</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def annotations_to_markdown(text: str, annotations: list[SnomedAnnotation]) -> str:
    """Generate Markdown text with inline SNOMED codes.

    Annotated spans are rendered as: **span text** [SCTID:concept_id]
    A summary table is appended at the end.

    Args:
        text: The full transcript text (dialogue entries joined by space).
        annotations: SNOMED annotations with start_char/end_char offsets.

    Returns:
        Markdown string.
    """
    sorted_anns = _sort_annotations(annotations)
    parts: list[str] = []
    cursor = 0

    for ann in sorted_anns:
        start = ann.start_char
        end = ann.end_char

        if start > cursor:
            parts.append(text[cursor:start])

        concept_id = ann.snomed_concept_id or "unknown"
        span_text = text[start:end]
        parts.append(f"**{span_text}** [SCTID:{concept_id}]")
        cursor = end

    if cursor < len(text):
        parts.append(text[cursor:])

    annotated_text = "".join(parts)

    summary_lines = [
        "",
        "---",
        "",
        "## Annotation Summary",
        "",
        "| Text | Type | SNOMED ID | Preferred Term | Confidence | Verified |",
        "|------|------|-----------|----------------|------------|----------|",
    ]
    for ann in annotations:
        summary_lines.append(
            f"| {ann.text_span} | {ann.entity_type or ''} | "
            f"{ann.snomed_concept_id or ''} | {ann.snomed_preferred_term or ''} | "
            f"{ann.confidence_score:.0%} | {'Yes' if ann.is_verified else 'No'} |"
        )

    return annotated_text + "\n".join(summary_lines) + "\n"
