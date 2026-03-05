"""Add snomed_annotation table

Revision ID: a1b2c3d4e5f6
Revises: 9d080ca9fe6c
Create Date: 2026-02-04 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "9d080ca9fe6c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Use postgresql.ENUM with create_type=False to prevent automatic creation
# during op.create_table(). The enum is created explicitly via raw SQL.
source_type_enum = postgresql.ENUM("TRANSCRIPT", "MINUTE", "MINUTE_VERSION", name="sourcetype", create_type=False)


def upgrade() -> None:
    # Create enum idempotently — it may already exist from SQLModel metadata
    op.execute(
        "DO $$ BEGIN "
        "CREATE TYPE sourcetype AS ENUM ('TRANSCRIPT', 'MINUTE', 'MINUTE_VERSION'); "
        "EXCEPTION WHEN duplicate_object THEN NULL; "
        "END $$"
    )
    op.create_table(
        "snomed_annotation",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("created_datetime", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_datetime", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("transcription_id", sa.Uuid(), nullable=False),
        sa.Column("source_type", source_type_enum, nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=True),
        sa.Column("text_span", sqlmodel.sql.sqltypes.AutoString(length=500), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("entity_type", sqlmodel.sql.sqltypes.AutoString(length=50), nullable=True),
        sa.Column("snomed_concept_id", sqlmodel.sql.sqltypes.AutoString(length=20), nullable=True),
        sa.Column("snomed_preferred_term", sqlmodel.sql.sqltypes.AutoString(length=500), nullable=True),
        sa.Column("snomed_fsn", sqlmodel.sql.sqltypes.AutoString(length=500), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("is_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("verified_by", sa.Uuid(), nullable=True),
        sa.Column("alternative_concepts", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["transcription_id"], ["transcription.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["verified_by"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_snomed_annotation_transcription_id", "snomed_annotation", ["transcription_id"])
    op.create_index("ix_snomed_annotation_snomed_concept_id", "snomed_annotation", ["snomed_concept_id"])


def downgrade() -> None:
    op.drop_index("ix_snomed_annotation_snomed_concept_id", table_name="snomed_annotation")
    op.drop_index("ix_snomed_annotation_transcription_id", table_name="snomed_annotation")
    op.drop_table("snomed_annotation")
    op.execute("DROP TYPE IF EXISTS sourcetype")
