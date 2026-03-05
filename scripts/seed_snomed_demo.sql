-- Seed script: creates a demo user, transcription, and SNOMED annotations
-- for visual testing of the Clinical Codes tab.
--
-- Usage (with docker compose running):
--   docker compose exec db psql -U minute -d minute -f /dev/stdin < scripts/seed_snomed_demo.sql

-- Create the local dev user if it doesn't exist
INSERT INTO "user" (id, email)
VALUES ('00000000-0000-4000-a000-000000000001', 'test@test.co.uk')
ON CONFLICT (id) DO NOTHING;

-- Create a completed transcription with clinical dialogue
INSERT INTO transcription (id, user_id, title, status, dialogue_entries)
VALUES (
    '00000000-0000-4000-a000-000000000010',
    '00000000-0000-4000-a000-000000000001',
    'Clinical Assessment Demo',
    'COMPLETED',
    '[
        {"speaker": "Clinician", "text": "The patient presents with a severe headache and elevated blood pressure.", "start_time": 0.0, "end_time": 5.0},
        {"speaker": "Clinician", "text": "Physical examination reveals tenderness in the right upper quadrant.", "start_time": 5.0, "end_time": 10.0},
        {"speaker": "Clinician", "text": "We will schedule an MRI of the brain and start the patient on paracetamol for pain relief.", "start_time": 10.0, "end_time": 16.0},
        {"speaker": "Clinician", "text": "There is also mild dyspnoea on exertion which warrants further investigation.", "start_time": 16.0, "end_time": 21.0}
    ]'::jsonb
)
ON CONFLICT (id) DO NOTHING;

-- Insert SNOMED annotations for the transcription
INSERT INTO snomed_annotation (id, transcription_id, source_type, text_span, start_char, end_char, entity_type, snomed_concept_id, snomed_preferred_term, snomed_fsn, confidence_score, is_verified, alternative_concepts)
VALUES
    -- Headache
    ('00000000-0000-4000-b000-000000000001',
     '00000000-0000-4000-a000-000000000010',
     'TRANSCRIPT', 'severe headache', 34, 49, 'finding',
     '25064002', 'Headache', 'Headache (finding)',
     0.94, false,
     '[{"concept_id": "25064002", "preferred_term": "Headache", "score": 0.94},
       {"concept_id": "162209004", "preferred_term": "Severe headache", "score": 0.91},
       {"concept_id": "230461009", "preferred_term": "Tension headache", "score": 0.72}]'::jsonb),

    -- Elevated blood pressure
    ('00000000-0000-4000-b000-000000000002',
     '00000000-0000-4000-a000-000000000010',
     'TRANSCRIPT', 'elevated blood pressure', 54, 77, 'finding',
     '38341003', 'Hypertension', 'Hypertensive disorder, systemic arterial (disorder)',
     0.89, false,
     '[{"concept_id": "38341003", "preferred_term": "Hypertension", "score": 0.89},
       {"concept_id": "24184005", "preferred_term": "Increased blood pressure", "score": 0.85}]'::jsonb),

    -- Right upper quadrant tenderness
    ('00000000-0000-4000-b000-000000000003',
     '00000000-0000-4000-a000-000000000010',
     'TRANSCRIPT', 'tenderness in the right upper quadrant', 120, 159, 'finding',
     '301717006', 'Right upper quadrant tenderness', 'Right upper quadrant tenderness (finding)',
     0.87, false,
     '[{"concept_id": "301717006", "preferred_term": "Right upper quadrant tenderness", "score": 0.87}]'::jsonb),

    -- MRI of brain
    ('00000000-0000-4000-b000-000000000004',
     '00000000-0000-4000-a000-000000000010',
     'TRANSCRIPT', 'MRI of the brain', 180, 196, 'procedure',
     '241615005', 'MRI of brain', 'Magnetic resonance imaging of brain (procedure)',
     0.96, true,
     '[{"concept_id": "241615005", "preferred_term": "MRI of brain", "score": 0.96}]'::jsonb),

    -- Paracetamol
    ('00000000-0000-4000-b000-000000000005',
     '00000000-0000-4000-a000-000000000010',
     'TRANSCRIPT', 'paracetamol', 226, 237, 'substance',
     '387517004', 'Paracetamol', 'Paracetamol (substance)',
     0.98, false,
     '[{"concept_id": "387517004", "preferred_term": "Paracetamol", "score": 0.98},
       {"concept_id": "90332006", "preferred_term": "Paracetamol product", "score": 0.82}]'::jsonb),

    -- Dyspnoea
    ('00000000-0000-4000-b000-000000000006',
     '00000000-0000-4000-a000-000000000010',
     'TRANSCRIPT', 'dyspnoea on exertion', 280, 300, 'finding',
     '267036007', 'Dyspnoea on exertion', 'Dyspnea on exertion (finding)',
     0.91, false,
     '[{"concept_id": "267036007", "preferred_term": "Dyspnoea on exertion", "score": 0.91},
       {"concept_id": "230145002", "preferred_term": "Difficulty breathing", "score": 0.78}]'::jsonb)

ON CONFLICT (id) DO NOTHING;
