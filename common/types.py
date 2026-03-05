import uuid
from datetime import datetime
from enum import IntEnum, StrEnum, auto

from pydantic import BaseModel, Field

from common.database.postgres_models import (
    ContentSource,
    DialogueEntry,
    HallucinationType,
    JobStatus,
    SourceType,
    TemplateType,
)


class TranscriptionMetadata(BaseModel):
    """Pydantic model for transcription metadata."""

    id: uuid.UUID
    created_datetime: datetime
    title: str | None = None
    text: str
    status: JobStatus


class PaginatedTranscriptionsResponse(BaseModel):
    """Paginated response for transcriptions."""

    items: list[TranscriptionMetadata]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class TranscriptionCreateRequest(BaseModel):
    recording_id: uuid.UUID
    template_name: str
    template_id: uuid.UUID | None = None
    agenda: str | None = None
    title: str | None = None


class RecordingCreateRequest(BaseModel):
    file_extension: str


class RecordingCreateResponse(BaseModel):
    id: uuid.UUID
    upload_url: str


class TranscriptionCreateResponse(BaseModel):
    id: uuid.UUID


class TranscriptionConfirmResponse(BaseModel):
    id: uuid.UUID


class TranscriptionPatchRequest(BaseModel):
    title: str | None = None
    dialogue_entries: list[DialogueEntry] | None = None


class ChatCreateRequest(BaseModel):
    user_content: str


class ChatGetResponse(BaseModel):
    id: uuid.UUID
    created_datetime: datetime
    updated_datetime: datetime
    user_content: str
    assistant_content: str | None
    status: JobStatus


class ChatGetAllResponse(BaseModel):
    chat: list[ChatGetResponse]


class ChatCreateResponse(BaseModel):
    id: uuid.UUID


class GetUserResponse(BaseModel):
    id: uuid.UUID
    created_datetime: datetime
    updated_datetime: datetime
    email: str
    data_retention_days: int | None


class DataRetentionUpdateResponse(BaseModel):
    data_retention_days: int | None


class TranscriptionGetResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    dialogue_entries: list[DialogueEntry] | None
    status: JobStatus
    created_datetime: datetime


class SingleRecording(BaseModel):
    id: uuid.UUID
    url: str
    extension: str


class MinuteListItem(BaseModel):
    id: uuid.UUID
    created_datetime: datetime
    updated_datetime: datetime
    transcription_id: uuid.UUID
    template_name: str
    agenda: str | None


class MinutesCreateRequest(BaseModel):
    template_name: str = Field(description="Name of the template to use for the minutes")
    template_id: uuid.UUID | None = Field(description="Optional id of user template")
    agenda: str | None = Field(description="The agenda for the meeting", default=None)


class AiEdit(BaseModel):
    instruction: str
    source_id: uuid.UUID


class MinuteVersionCreateRequest(BaseModel):
    ai_edit_instructions: AiEdit | None = Field(
        default=None,
        description="If the content source is an AI edit, store the instruction and source version id here",
    )
    content_source: ContentSource
    html_content: str = Field(default="")


class MinutesPatchRequest(BaseModel):
    html_content: str | None = None


class MinuteVersionResponse(BaseModel):
    id: uuid.UUID
    minute_id: uuid.UUID
    status: JobStatus
    created_datetime: datetime
    html_content: str
    error: str | None
    ai_edit_instructions: str | None
    content_source: ContentSource


class SpeakerPrediction(BaseModel):
    original_speaker: str
    predicted_name: str
    confidence: float


class SpeakerPredictionOutput(BaseModel):
    predictions: list[SpeakerPrediction]


class MinutesResponse(BaseModel):
    minutes: str


class MeetingCheck(BaseModel):
    is_long_meeting: bool


class TaskType(IntEnum):
    # messages have a natural ordering in which we want them to happen
    TRANSCRIPTION = 1
    MINUTE = 2
    EDIT = 3
    INTERACTIVE = 4
    SNOMED_CODING = 5


class EditMessageData(BaseModel):
    source_id: uuid.UUID = Field(description="ID of the source message")


class TranscriptionJobMessageData(BaseModel):
    transcription_service: str = Field(description="Name of the transcription service")
    job_name: str = Field(
        description="job name to identify asynchronous jobs. Not used in case of synchronous jobs",
        default="synchronous",
    )
    transcript: list[DialogueEntry] | None = Field(description="Transcript of the transcription", default=None)


class SNOMEDCodingMessageData(BaseModel):
    source_type: str = Field(description="Source type for SNOMED coding: transcript or minute")


class WorkerMessage(BaseModel):
    id: uuid.UUID
    type: TaskType
    data: EditMessageData | TranscriptionJobMessageData | SNOMEDCodingMessageData | None = Field(default=None)


class LLMHallucination(BaseModel):
    hallucination_type: HallucinationType = Field(description="Type of hallucination")
    hallucination_text: str | None = Field(description="Text of hallucination", default=None)
    hallucination_reason: str | None = Field(description="Reason for hallucination", default=None)


MinuteAndHallucinations = tuple[str, list[LLMHallucination] | None]


class MeetingType(StrEnum):
    too_short = auto()
    short = auto()
    standard = auto()


class AgendaUsage(StrEnum):
    NOT_USED = auto()
    OPTIONAL = auto()
    REQUIRED = auto()


class TemplateMetadata(BaseModel):
    name: str
    description: str
    category: str
    agenda_usage: AgendaUsage


class CreateQuestion(BaseModel):
    position: int
    title: str
    description: str


class Question(CreateQuestion):
    id: uuid.UUID


class PatchUserTemplateRequest(BaseModel):
    name: str | None = None
    content: str | None = None
    description: str | None = None
    questions: list[CreateQuestion | Question] | None = None


class TemplateResponse(BaseModel):
    id: uuid.UUID
    updated_datetime: datetime
    name: str
    content: str
    description: str
    type: TemplateType
    questions: list[Question] | None


class CreateUserTemplateRequest(BaseModel):
    name: str
    content: str
    description: str
    type: TemplateType
    questions: list[CreateQuestion] | None = None


class AlternativeSnomedConcept(BaseModel):
    """A candidate SNOMED concept from the top-K results."""

    concept_id: str = Field(description="SNOMED CT concept identifier")
    preferred_term: str = Field(description="SNOMED CT preferred term")
    fsn: str | None = Field(default=None, description="Fully specified name")
    confidence_score: float = Field(description="Confidence score for this candidate")


class SnomedAnnotationResponse(BaseModel):
    """Response schema for a single SNOMED annotation."""

    id: uuid.UUID
    transcription_id: uuid.UUID
    source_type: SourceType
    source_id: uuid.UUID | None
    text_span: str
    start_char: int
    end_char: int
    entity_type: str | None
    snomed_concept_id: str | None
    snomed_preferred_term: str | None
    snomed_fsn: str | None
    confidence_score: float | None
    is_verified: bool
    verified_by: uuid.UUID | None
    alternative_concepts: list[AlternativeSnomedConcept] | None
    created_datetime: datetime
    updated_datetime: datetime


class SnomedAnnotationListResponse(BaseModel):
    """Response schema for listing SNOMED annotations for a transcription."""

    annotations: list[SnomedAnnotationResponse]
    total_count: int


class SnomedAnnotationVerifyRequest(BaseModel):
    """Request to verify or correct a SNOMED annotation."""

    is_verified: bool = Field(description="Whether the annotation is verified as correct")
    snomed_concept_id: str | None = Field(
        default=None, description="Corrected SNOMED concept ID if overriding the model's choice"
    )
    snomed_preferred_term: str | None = Field(default=None, description="Corrected preferred term")
    snomed_fsn: str | None = Field(default=None, description="Corrected fully specified name")


class SnomedConceptSearchResult(BaseModel):
    """A single SNOMED concept in search results."""

    concept_id: str = Field(description="SNOMED CT concept identifier")
    preferred_term: str = Field(description="Preferred term for the concept")
    fsn: str = Field(description="Fully specified name")
    semantic_tag: str = Field(description="Semantic tag (e.g., finding, procedure)")


class SnomedConceptSearchResponse(BaseModel):
    """Response for SNOMED concept search endpoint."""

    results: list[SnomedConceptSearchResult] = Field(description="Matching concepts")
    total_count: int = Field(description="Number of results returned")
    query: str = Field(description="The search query")
