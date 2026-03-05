"""
Mock SNOMED CT data for testing.

Generates synthetic SNOMED CT concepts for unit tests without
requiring actual SNOMED CT release files.
"""

from worker.snomed.data.concept_db import ConceptDatabase
from worker.snomed.data.rf2_parser import SNOMEDConcept

# Mock concepts for testing - representative clinical terms
MOCK_CONCEPTS: list[dict] = [
    # Clinical Findings
    {
        "concept_id": "25064002",
        "preferred_term": "Headache",
        "fsn": "Headache (finding)",
        "synonyms": ["Cephalalgia", "Head pain", "Pain in head"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "386661006",
        "preferred_term": "Fever",
        "fsn": "Fever (finding)",
        "synonyms": ["Pyrexia", "Febrile", "Elevated temperature"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "49727002",
        "preferred_term": "Cough",
        "fsn": "Cough (finding)",
        "synonyms": ["Coughing", "Tussis"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "267036007",
        "preferred_term": "Dyspnea",
        "fsn": "Dyspnea (finding)",
        "synonyms": ["Shortness of breath", "Breathlessness", "SOB", "Difficulty breathing"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "22298006",
        "preferred_term": "Myocardial infarction",
        "fsn": "Myocardial infarction (disorder)",
        "synonyms": ["Heart attack", "MI", "Acute myocardial infarction", "AMI"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "73211009",
        "preferred_term": "Diabetes mellitus",
        "fsn": "Diabetes mellitus (disorder)",
        "synonyms": ["Diabetes", "DM", "Sugar diabetes"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "38341003",
        "preferred_term": "Hypertension",
        "fsn": "Hypertensive disorder, systemic arterial (disorder)",
        "synonyms": ["High blood pressure", "HTN", "Elevated blood pressure"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "195967001",
        "preferred_term": "Asthma",
        "fsn": "Asthma (disorder)",
        "synonyms": ["Bronchial asthma", "Asthmatic"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "84757009",
        "preferred_term": "Epilepsy",
        "fsn": "Epilepsy (disorder)",
        "synonyms": ["Seizure disorder", "Epileptic disorder"],
        "semantic_tag": "finding",
    },
    {
        "concept_id": "35489007",
        "preferred_term": "Depression",
        "fsn": "Depressive disorder (disorder)",
        "synonyms": ["Depressive illness", "Clinical depression", "Major depression"],
        "semantic_tag": "finding",
    },
    # Procedures
    {
        "concept_id": "387713003",
        "preferred_term": "Surgical procedure",
        "fsn": "Surgical procedure (procedure)",
        "synonyms": ["Surgery", "Operation", "Surgical operation"],
        "semantic_tag": "procedure",
    },
    {
        "concept_id": "241615005",
        "preferred_term": "Magnetic resonance imaging",
        "fsn": "Magnetic resonance imaging (procedure)",
        "synonyms": ["MRI", "MRI scan", "MR imaging"],
        "semantic_tag": "procedure",
    },
    {
        "concept_id": "77477000",
        "preferred_term": "Computed tomography",
        "fsn": "Computerized axial tomography (procedure)",
        "synonyms": ["CT scan", "CAT scan", "CT"],
        "semantic_tag": "procedure",
    },
    {
        "concept_id": "108220007",
        "preferred_term": "Blood test",
        "fsn": "Hematologic test (procedure)",
        "synonyms": ["Blood work", "Blood analysis", "Hematology test"],
        "semantic_tag": "procedure",
    },
    {
        "concept_id": "252416005",
        "preferred_term": "Echocardiography",
        "fsn": "Echocardiography (procedure)",
        "synonyms": ["Echo", "Cardiac ultrasound", "Heart ultrasound"],
        "semantic_tag": "procedure",
    },
    {
        "concept_id": "28163009",
        "preferred_term": "Colonoscopy",
        "fsn": "Colonoscopy (procedure)",
        "synonyms": ["Colonoscopic examination"],
        "semantic_tag": "procedure",
    },
    # Body Structures
    {
        "concept_id": "80891009",
        "preferred_term": "Heart",
        "fsn": "Heart structure (body structure)",
        "synonyms": ["Cardiac structure", "Heart organ"],
        "semantic_tag": "body_structure",
    },
    {
        "concept_id": "39607008",
        "preferred_term": "Lung",
        "fsn": "Lung structure (body structure)",
        "synonyms": ["Pulmonary structure", "Lung organ"],
        "semantic_tag": "body_structure",
    },
    {
        "concept_id": "12738006",
        "preferred_term": "Brain",
        "fsn": "Brain structure (body structure)",
        "synonyms": ["Cerebral structure", "Brain organ", "Encephalon"],
        "semantic_tag": "body_structure",
    },
    {
        "concept_id": "10200004",
        "preferred_term": "Liver",
        "fsn": "Liver structure (body structure)",
        "synonyms": ["Hepatic structure", "Liver organ"],
        "semantic_tag": "body_structure",
    },
    {
        "concept_id": "64033007",
        "preferred_term": "Kidney",
        "fsn": "Kidney structure (body structure)",
        "synonyms": ["Renal structure", "Kidney organ"],
        "semantic_tag": "body_structure",
    },
]


def generate_mock_concepts() -> dict[str, SNOMEDConcept]:
    """
    Generate mock SNOMED CT concepts for testing.

    Returns:
        Dictionary mapping concept_id to SNOMEDConcept
    """
    concepts = {}
    for data in MOCK_CONCEPTS:
        concept = SNOMEDConcept(
            concept_id=data["concept_id"],
            preferred_term=data["preferred_term"],
            fsn=data["fsn"],
            synonyms=data["synonyms"],
            semantic_tag=data["semantic_tag"],
            is_active=True,
            parent_ids=[],
        )
        concepts[concept.concept_id] = concept

    return concepts


def generate_mock_concept_database() -> ConceptDatabase:
    """
    Generate a mock ConceptDatabase for testing.

    Returns:
        ConceptDatabase with mock concepts
    """
    concepts = generate_mock_concepts()
    db = ConceptDatabase(concepts)
    db.set_metadata(version="mock", source="Mock data for testing")
    return db


# Mock clinical texts with annotated entities for NER testing.
# Entity offsets are character-based and match the text exactly.
MOCK_CLINICAL_TEXTS: list[dict] = [
    {
        "text": "Patient presents with headache and fever. Blood pressure elevated.",
        "entities": [
            {"text": "headache", "start": 22, "end": 30, "entity_type": "finding"},
            {"text": "fever", "start": 35, "end": 40, "entity_type": "finding"},
            {"text": "Blood pressure", "start": 42, "end": 56, "entity_type": "observable_entity"},
        ],
    },
    {
        "text": "MRI of the brain showed no abnormalities. CT scan scheduled for next week.",
        "entities": [
            {"text": "MRI", "start": 0, "end": 3, "entity_type": "procedure"},
            {"text": "brain", "start": 11, "end": 16, "entity_type": "body_structure"},
            {"text": "CT scan", "start": 42, "end": 49, "entity_type": "procedure"},
        ],
    },
    {
        "text": "History of diabetes mellitus and hypertension. Currently on metformin.",
        "entities": [
            {"text": "diabetes mellitus", "start": 11, "end": 28, "entity_type": "finding"},
            {"text": "hypertension", "start": 33, "end": 45, "entity_type": "finding"},
            {"text": "metformin", "start": 61, "end": 70, "entity_type": "substance"},
        ],
    },
    {
        "text": "Echocardiography revealed mild mitral valve regurgitation. Heart function preserved.",
        "entities": [
            {"text": "Echocardiography", "start": 0, "end": 16, "entity_type": "procedure"},
            {"text": "mitral valve regurgitation", "start": 31, "end": 57, "entity_type": "finding"},
            {"text": "Heart", "start": 59, "end": 64, "entity_type": "body_structure"},
        ],
    },
    {
        "text": "No significant findings on examination.",
        "entities": [],
    },
]


# Mock linking data: entities with expected SNOMED concept IDs.
# Used for testing the entity linker.
MOCK_LINKING_DATA: list[dict] = [
    {"entity_text": "Headache", "entity_type": "finding", "gold_concept_id": "25064002"},
    {"entity_text": "Fever", "entity_type": "finding", "gold_concept_id": "386661006"},
    {"entity_text": "Cough", "entity_type": "finding", "gold_concept_id": "49727002"},
    {"entity_text": "Shortness of breath", "entity_type": "finding", "gold_concept_id": "267036007"},
    {"entity_text": "Heart attack", "entity_type": "finding", "gold_concept_id": "22298006"},
    {"entity_text": "Diabetes", "entity_type": "finding", "gold_concept_id": "73211009"},
    {"entity_text": "High blood pressure", "entity_type": "finding", "gold_concept_id": "38341003"},
    {"entity_text": "MRI", "entity_type": "procedure", "gold_concept_id": "241615005"},
    {"entity_text": "CT scan", "entity_type": "procedure", "gold_concept_id": "77477000"},
    {"entity_text": "Heart", "entity_type": "body_structure", "gold_concept_id": "80891009"},
    {"entity_text": "Brain", "entity_type": "body_structure", "gold_concept_id": "12738006"},
    {"entity_text": "Lung", "entity_type": "body_structure", "gold_concept_id": "39607008"},
]


def get_mock_concept_by_term(term: str) -> SNOMEDConcept | None:
    """
    Get a mock concept by preferred term.

    Args:
        term: Preferred term to search for (case-insensitive)

    Returns:
        SNOMEDConcept if found, None otherwise
    """
    term_lower = term.lower()
    for data in MOCK_CONCEPTS:
        if data["preferred_term"].lower() == term_lower:
            return SNOMEDConcept(
                concept_id=data["concept_id"],
                preferred_term=data["preferred_term"],
                fsn=data["fsn"],
                synonyms=data["synonyms"],
                semantic_tag=data["semantic_tag"],
                is_active=True,
            )
    return None
