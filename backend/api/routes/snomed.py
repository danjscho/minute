"""SNOMED CT annotation API endpoints."""

from __future__ import annotations

import csv
import io
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import col, select

from backend.api.dependencies import SQLSessionDep, UserDep
from common.database.postgres_models import SnomedAnnotation, Transcription
from common.services.queue_services import get_queue_service
from common.services.snomed_search_service import SnomedConceptSearchService
from common.settings import get_settings
from common.types import (
    SnomedAnnotationListResponse,
    SnomedAnnotationResponse,
    SnomedAnnotationVerifyRequest,
    SNOMEDCodingMessageData,
    SnomedConceptSearchResponse,
    SnomedConceptSearchResult,
    TaskType,
    WorkerMessage,
)

settings = get_settings()
logger = logging.getLogger(__name__)

snomed_router = APIRouter(tags=["SNOMED"])
llm_queue_service = get_queue_service(
    settings.QUEUE_SERVICE_NAME, settings.LLM_QUEUE_NAME, settings.LLM_DEADLETTER_QUEUE_NAME
)


async def _verify_transcription_ownership(
    session: SQLSessionDep, transcription_id: UUID, user_id: UUID
) -> Transcription:
    """Verify transcription exists and belongs to the user."""
    transcription = await session.get(Transcription, transcription_id)
    if not transcription or transcription.user_id != user_id:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return transcription


@snomed_router.get(
    "/transcriptions/{transcription_id}/snomed-annotations",
    response_model=SnomedAnnotationListResponse,
)
async def get_snomed_annotations(
    transcription_id: UUID,
    session: SQLSessionDep,
    current_user: UserDep,
) -> SnomedAnnotationListResponse:
    """Get all SNOMED annotations for a transcription."""
    await _verify_transcription_ownership(session, transcription_id, current_user.id)

    statement = (
        select(SnomedAnnotation)
        .where(SnomedAnnotation.transcription_id == transcription_id)
        .order_by(col(SnomedAnnotation.start_char))
    )
    result = await session.exec(statement)
    annotations = result.all()

    return SnomedAnnotationListResponse(
        annotations=[
            SnomedAnnotationResponse(
                id=a.id,
                transcription_id=a.transcription_id,
                source_type=a.source_type,
                source_id=a.source_id,
                text_span=a.text_span,
                start_char=a.start_char,
                end_char=a.end_char,
                entity_type=a.entity_type,
                snomed_concept_id=a.snomed_concept_id,
                snomed_preferred_term=a.snomed_preferred_term,
                snomed_fsn=a.snomed_fsn,
                confidence_score=a.confidence_score,
                is_verified=a.is_verified,
                verified_by=a.verified_by,
                alternative_concepts=a.alternative_concepts,
                created_datetime=a.created_datetime,
                updated_datetime=a.updated_datetime,
            )
            for a in annotations
        ],
        total_count=len(annotations),
    )


@snomed_router.post(
    "/transcriptions/{transcription_id}/snomed-annotations/trigger",
    status_code=202,
)
async def trigger_snomed_coding(
    transcription_id: UUID,
    session: SQLSessionDep,
    current_user: UserDep,
    source_type: str = Query(default="transcript", description="Source to code: transcript or minute"),
) -> dict:
    """Trigger SNOMED CT coding for a transcription.

    Returns 202 Accepted — coding runs asynchronously via the worker queue.
    """
    if not settings.SNOMED_CODING_ENABLED:
        raise HTTPException(status_code=503, detail="SNOMED coding is not enabled")

    if source_type not in ("transcript", "minute"):
        raise HTTPException(status_code=400, detail="source_type must be 'transcript' or 'minute'")

    await _verify_transcription_ownership(session, transcription_id, current_user.id)

    llm_queue_service.publish_message(
        WorkerMessage(
            id=transcription_id,
            type=TaskType.SNOMED_CODING,
            data=SNOMEDCodingMessageData(source_type=source_type),
        )
    )

    return {"message": "SNOMED coding triggered", "transcription_id": str(transcription_id)}


@snomed_router.delete(
    "/snomed-annotations/{annotation_id}",
    status_code=204,
)
async def delete_annotation(
    annotation_id: UUID,
    session: SQLSessionDep,
    current_user: UserDep,
) -> None:
    """Delete a SNOMED annotation (reject)."""
    annotation = await session.get(SnomedAnnotation, annotation_id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    transcription = await session.get(Transcription, annotation.transcription_id)
    if not transcription or transcription.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Annotation not found")

    await session.delete(annotation)
    await session.commit()


@snomed_router.patch(
    "/snomed-annotations/{annotation_id}/verify",
    response_model=SnomedAnnotationResponse,
)
async def verify_annotation(
    annotation_id: UUID,
    request: SnomedAnnotationVerifyRequest,
    session: SQLSessionDep,
    current_user: UserDep,
) -> SnomedAnnotationResponse:
    """Verify or correct a SNOMED annotation."""
    annotation = await session.get(SnomedAnnotation, annotation_id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    # Verify the annotation belongs to a transcription owned by this user
    transcription = await session.get(Transcription, annotation.transcription_id)
    if not transcription or transcription.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Annotation not found")

    annotation.is_verified = request.is_verified
    annotation.verified_by = current_user.id

    # Allow overriding the concept if provided
    if request.snomed_concept_id is not None:
        annotation.snomed_concept_id = request.snomed_concept_id
    if request.snomed_preferred_term is not None:
        annotation.snomed_preferred_term = request.snomed_preferred_term
    if request.snomed_fsn is not None:
        annotation.snomed_fsn = request.snomed_fsn

    session.add(annotation)
    await session.commit()
    await session.refresh(annotation)

    return SnomedAnnotationResponse(
        id=annotation.id,
        transcription_id=annotation.transcription_id,
        source_type=annotation.source_type,
        source_id=annotation.source_id,
        text_span=annotation.text_span,
        start_char=annotation.start_char,
        end_char=annotation.end_char,
        entity_type=annotation.entity_type,
        snomed_concept_id=annotation.snomed_concept_id,
        snomed_preferred_term=annotation.snomed_preferred_term,
        snomed_fsn=annotation.snomed_fsn,
        confidence_score=annotation.confidence_score,
        is_verified=annotation.is_verified,
        verified_by=annotation.verified_by,
        alternative_concepts=annotation.alternative_concepts,
        created_datetime=annotation.created_datetime,
        updated_datetime=annotation.updated_datetime,
    )


@snomed_router.get("/transcriptions/{transcription_id}/snomed-annotations/export", response_model=None)
async def export_snomed_annotations(
    transcription_id: UUID,
    session: SQLSessionDep,
    current_user: UserDep,
    export_format: str = Query(
        "json", alias="format", description="Export format: json, csv, fhir, html, or markdown"
    ),
) -> SnomedAnnotationListResponse | StreamingResponse | JSONResponse:
    """Export SNOMED annotations for a transcription."""
    transcription = await _verify_transcription_ownership(session, transcription_id, current_user.id)

    statement = (
        select(SnomedAnnotation)
        .where(SnomedAnnotation.transcription_id == transcription_id)
        .order_by(col(SnomedAnnotation.start_char))
    )
    result = await session.exec(statement)
    annotations = result.all()

    if export_format == "csv":
        return _annotations_to_csv(annotations)

    if export_format == "fhir":
        from worker.snomed.export.fhir_export import annotations_to_fhir_bundle

        bundle = annotations_to_fhir_bundle(annotations, str(transcription_id))
        return JSONResponse(
            content=bundle,
            media_type="application/fhir+json",
            headers={"Content-Disposition": "attachment; filename=snomed_fhir_bundle.json"},
        )

    if export_format in ("html", "markdown"):
        from worker.snomed.export.annotated_text import annotations_to_html, annotations_to_markdown

        text = " ".join(entry["text"] for entry in (transcription.dialogue_entries or []))

        if export_format == "html":
            html_content = annotations_to_html(text, annotations)
            return StreamingResponse(
                iter([html_content]),
                media_type="text/html",
                headers={"Content-Disposition": "attachment; filename=snomed_annotated.html"},
            )

        md_content = annotations_to_markdown(text, annotations)
        return StreamingResponse(
            iter([md_content]),
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=snomed_annotated.md"},
        )

    # Default: JSON
    return SnomedAnnotationListResponse(
        annotations=[
            SnomedAnnotationResponse(
                id=a.id,
                transcription_id=a.transcription_id,
                source_type=a.source_type,
                source_id=a.source_id,
                text_span=a.text_span,
                start_char=a.start_char,
                end_char=a.end_char,
                entity_type=a.entity_type,
                snomed_concept_id=a.snomed_concept_id,
                snomed_preferred_term=a.snomed_preferred_term,
                snomed_fsn=a.snomed_fsn,
                confidence_score=a.confidence_score,
                is_verified=a.is_verified,
                verified_by=a.verified_by,
                alternative_concepts=a.alternative_concepts,
                created_datetime=a.created_datetime,
                updated_datetime=a.updated_datetime,
            )
            for a in annotations
        ],
        total_count=len(annotations),
    )


@snomed_router.get(
    "/snomed-concepts/search",
    response_model=SnomedConceptSearchResponse,
)
async def search_snomed_concepts(
    current_user: UserDep,
    q: str = Query(min_length=2, max_length=200, description="Search term"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results to return"),
) -> SnomedConceptSearchResponse:
    """Search SNOMED CT concepts by term (prefix match)."""
    if not settings.SNOMED_CODING_ENABLED:
        raise HTTPException(status_code=503, detail="SNOMED coding is not enabled")

    service = SnomedConceptSearchService.get_instance()
    service.load(settings.SNOMED_CT_DATA_PATH)

    results = service.search(q, limit=limit)

    return SnomedConceptSearchResponse(
        results=[
            SnomedConceptSearchResult(
                concept_id=c.concept_id,
                preferred_term=c.preferred_term,
                fsn=c.fsn,
                semantic_tag=c.semantic_tag,
            )
            for c in results
        ],
        total_count=len(results),
        query=q,
    )


def _annotations_to_csv(annotations: list[SnomedAnnotation]) -> StreamingResponse:
    """Convert annotations to a CSV streaming response."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "text_span",
        "entity_type",
        "snomed_concept_id",
        "snomed_preferred_term",
        "snomed_fsn",
        "confidence_score",
        "is_verified",
        "start_char",
        "end_char",
    ])
    for a in annotations:
        writer.writerow([
            a.text_span,
            a.entity_type,
            a.snomed_concept_id,
            a.snomed_preferred_term,
            a.snomed_fsn,
            a.confidence_score,
            a.is_verified,
            a.start_char,
            a.end_char,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=snomed_annotations.csv"},
    )
