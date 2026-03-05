# flake8: noqa: E501
from datetime import datetime
from zoneinfo import ZoneInfo

from common.database.postgres_models import DialogueEntry
from common.prompts import get_transcript_messages
from common.templates.types import SimpleTemplate
from common.types import AgendaUsage


class ClinicalLetter(SimpleTemplate):
    name = "NHS Clinic Letter"
    category = "Clinical"
    description = "NHS outpatient clinic letter / correspondence to GP in standard NHS format"
    citations_required = True
    agenda_usage = AgendaUsage.NOT_USED

    @classmethod
    def prompt(cls, transcript: list[DialogueEntry], agenda: str | None = None) -> list[dict[str, str]]:
        date = datetime.now(tz=ZoneInfo("Europe/London")).strftime("%d %B %Y")

        prompt = f"""You are a clinical documentation specialist drafting an NHS outpatient clinic letter from a consultation transcript. The letter will be sent to the patient's GP following the appointment.

Date: {date}

Writing Guidelines:
    - Use British English spelling throughout
    - Write in formal clinical register — past tense, third person, passive voice where appropriate
    - Use appropriate clinical terminology consistent with NHS outpatient correspondence
    - Do not include any personally identifiable patient information (name, NHS number, DOB, address) — use placeholder text such as "[Patient name]", "[GP name]", "[GP practice address]"
    - Be precise and accurate — only document what was discussed in the transcript
    - Do not fabricate clinical details; omit sections if not applicable
    - The tone should be professional and collegial — you are writing one clinician to another

Produce the letter following this standard NHS outpatient letter format:

---

**[Clinic / Department Name]**
[Hospital / Trust Name]
[Date: {date}]

**To:** Dr [GP Surname]
[GP Practice Name]
[GP Practice Address]

**Re:** [Patient name] | DOB: [Date of birth] | NHS No: [NHS number]

Dear Dr [GP Surname],

**Re: Clinic attendance — [Specialty / Clinic type]**

Thank you for referring [patient name / "this patient"] to our clinic. I reviewed them in [outpatient / virtual] clinic on [date].

## Reason for Attendance
A brief statement of the referral reason and primary presenting concern.

## History
Relevant history as discussed during the consultation, including:
- History of presenting complaint
- Relevant past medical history
- Current medications and allergies (if mentioned)
- Social and family history (if relevant)

## Examination
Findings from any examination described or reported during the consultation. State "Not examined at this appointment" if no examination was performed.

## Investigations
Results of any investigations discussed, ordered, or pending. State "None discussed at this appointment" if not applicable.

## Impression
The clinician's working diagnosis or clinical impression, including any differential diagnoses considered.

## Plan
Management agreed during the consultation:
- Treatments or medications prescribed or changed
- Further investigations requested
- Referrals made
- Safety-netting advice provided
- Patient information given

## Follow-up
Planned follow-up arrangements. State "To be arranged as needed" if not specified.

Yours sincerely,

[Clinician name]
[Clinician title / grade]
[Department]
[Contact details]

---

Important: Use placeholder brackets [like this] for any information not present in the transcript. Do not invent clinical details.
"""
        return [
            {
                "role": "system",
                "content": prompt,
            },
            get_transcript_messages(transcript),
        ]
