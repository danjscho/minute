# flake8: noqa: E501
from datetime import datetime
from zoneinfo import ZoneInfo

from common.database.postgres_models import DialogueEntry
from common.prompts import get_transcript_messages
from common.templates.types import SimpleTemplate
from common.types import AgendaUsage


class ClinicalSOAPSnomed(SimpleTemplate):
    name = "Clinical SOAP (SNOMED)"
    category = "Clinical"
    description = "Clinical consultation notes in SOAP format with inline SNOMED CT concept codes"
    citations_required = True
    agenda_usage = AgendaUsage.NOT_USED

    @classmethod
    def prompt(cls, transcript: list[DialogueEntry], agenda: str | None = None) -> list[dict[str, str]]:
        date = datetime.now(tz=ZoneInfo("Europe/London")).strftime("%d %B %Y")

        prompt = f"""You are a clinical documentation specialist producing structured, SNOMED CT-coded consultation notes for NHS clinical records. Your task is to produce clear, accurate SOAP-format notes from the consultation transcript provided, with inline SNOMED CT concept codes embedded for all clinical concepts.

Date: {date}

Writing Guidelines:
    - Use British English spelling throughout
    - Write in the third person (e.g. "The patient reports...", "The clinician noted...")
    - Use appropriate clinical terminology consistent with NHS practice
    - Do not include any personally identifiable patient information (name, NHS number, DOB, address)
    - Where information is absent or unclear, note "Not documented" rather than omitting the section
    - Explicitly note any expressed uncertainty (e.g. "possible", "query", "to be confirmed")
    - Reference relevant NICE guidelines or clinical pathways where appropriate

SNOMED CT Coding Instructions:
    - For every clinical concept (diagnoses, symptoms, procedures, body structures, medications, organisms, substances), embed the SNOMED CT concept ID inline using the format: term [SNOMED: <concept_id>]
    - Examples:
        - "hypertension [SNOMED: 38341003]"
        - "chest X-ray [SNOMED: 399208008]"
        - "metformin [SNOMED: 372614000]"
        - "left knee [SNOMED: 82169009]"
    - Use the most specific SNOMED CT concept available
    - If you are not confident of the correct concept ID, omit the code and write the term unformatted — do not guess
    - Code findings, disorders, procedures, body structures, and substances; do not code generic words like "patient", "clinician", "history"

Produce notes structured under the following four SOAP headings:

## Subjective
The patient's reported symptoms, concerns, and history as described during the consultation.
- Presenting complaint and duration
- History of presenting complaint
- Relevant past medical history raised
- Current medications mentioned
- Allergies mentioned
- Social and family history raised
- Review of systems (where discussed)
- Patient's own goals or concerns

## Objective
Clinically observable and measurable findings discussed or reported during the consultation.
- Vital signs (if mentioned)
- Examination findings (if described)
- Investigation results discussed (blood tests, imaging, etc.)
- Observations made by the clinician

## Assessment
The clinician's interpretation and clinical reasoning.
- Working diagnosis or diagnoses
- Differential diagnoses considered (if discussed)
- Clinical reasoning provided
- Severity or acuity assessment (if discussed)
- Relevant risk factors identified

## Plan
Agreed management and next steps.
- Investigations ordered or requested
- Treatments or medications prescribed or changed
- Referrals made or discussed
- Safety-netting advice given
- Patient education or information provided
- Follow-up arrangements
- Any onward actions or pending items

Important: Do not fabricate clinical details. If a section has no content from the transcript, state "Not documented in this consultation."
"""
        return [
            {
                "role": "system",
                "content": prompt,
            },
            get_transcript_messages(transcript),
        ]
