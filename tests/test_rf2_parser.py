"""Tests for RF2 parser."""

import zipfile

from worker.snomed.data.rf2_parser import RF2Parser, extract_semantic_tag


class TestExtractSemanticTag:
    def test_finding(self) -> None:
        assert extract_semantic_tag("Headache (finding)") == "finding"

    def test_disorder(self) -> None:
        # "disorder" maps to "finding" in SEMANTIC_TAGS
        assert extract_semantic_tag("Diabetes mellitus (disorder)") == "finding"

    def test_procedure(self) -> None:
        assert extract_semantic_tag("Magnetic resonance imaging (procedure)") == "procedure"

    def test_body_structure(self) -> None:
        assert extract_semantic_tag("Heart structure (body structure)") == "body_structure"

    def test_unknown_tag_returned_as_is(self) -> None:
        assert extract_semantic_tag("Aspirin (product)") == "product"

    def test_no_parentheses(self) -> None:
        assert extract_semantic_tag("Headache") == ""

    def test_nested_parentheses(self) -> None:
        # Should use rfind to get the outermost
        assert extract_semantic_tag("Term (with (nested)) (finding)") == "finding"

    def test_empty_string(self) -> None:
        assert extract_semantic_tag("") == ""


class TestRF2ParserWithZip:
    """Test RF2 parsing using a synthetic ZIP archive."""

    def _build_rf2_zip(self, tmp_path):
        """Create a minimal RF2 ZIP with concept, description, and relationship files."""
        concept_content = (
            "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n"
            "25064002\t20240101\t1\t900000000000207008\t900000000000074008\n"
            "386661006\t20240101\t1\t900000000000207008\t900000000000074008\n"
            "999999999\t20240101\t0\t900000000000207008\t900000000000074008\n"
        )
        mod = "900000000000207008"
        fsn_type = "900000000000003001"
        syn_type = "900000000000013009"
        case_sig = "900000000000020002"
        desc_hdr = "id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n"
        description_content = (
            desc_hdr
            + f"1\t20240101\t1\t{mod}\t25064002\ten\t{fsn_type}\tHeadache (finding)\t{case_sig}\n"
            + f"2\t20240101\t1\t{mod}\t25064002\ten\t{syn_type}\tCephalalgia\t{case_sig}\n"
            + f"3\t20240101\t1\t{mod}\t386661006\ten\t{fsn_type}\tFever (finding)\t{case_sig}\n"
            + f"4\t20240101\t0\t{mod}\t25064002\ten\t{syn_type}\tInactive synonym\t{case_sig}\n"
        )
        relationship_content = (
            "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId\n"
            "1\t20240101\t1\t900000000000207008\t25064002\t404684003\t0\t116680003\t900000000000011006\t900000000000451002\n"
            "2\t20240101\t0\t900000000000207008\t386661006\t404684003\t0\t116680003\t900000000000011006\t900000000000451002\n"
        )

        zip_path = tmp_path / "SnomedCT_Test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Snapshot/sct2_Concept_Snapshot.txt", concept_content)
            zf.writestr("Snapshot/sct2_Description_Snapshot.txt", description_content)
            zf.writestr("Snapshot/sct2_Relationship_Snapshot.txt", relationship_content)
        return zip_path

    def test_parse_concepts_from_zip(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        concepts = parser.parse_concepts()

        # Should have 2 active concepts (999999999 is inactive)
        assert len(concepts) == 2
        assert "25064002" in concepts
        assert "386661006" in concepts
        assert "999999999" not in concepts

    def test_descriptions_parsed(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        concepts = parser.parse_concepts()

        headache = concepts["25064002"]
        assert headache.fsn == "Headache (finding)"
        assert headache.semantic_tag == "finding"
        assert headache.preferred_term == "Headache"
        assert "Cephalalgia" in headache.synonyms

    def test_inactive_descriptions_excluded(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        concepts = parser.parse_concepts()
        assert "Inactive synonym" not in concepts["25064002"].synonyms

    def test_relationships_parsed(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        concepts = parser.parse_concepts()

        # Active relationship: 25064002 -> 404684003
        assert "404684003" in concepts["25064002"].parent_ids
        # Inactive relationship should not be included
        assert concepts["386661006"].parent_ids == []

    def test_filter_by_semantic_tags(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        concepts = parser.parse_concepts()

        filtered = parser.filter_by_semantic_tags(concepts, tags=["finding"])
        assert "25064002" in filtered
        assert "386661006" in filtered

    def test_filter_by_semantic_tags_excludes(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        concepts = parser.parse_concepts()

        filtered = parser.filter_by_semantic_tags(concepts, tags=["procedure"])
        assert len(filtered) == 0

    def test_filter_by_hierarchies(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        concepts = parser.parse_concepts()

        # 25064002 has parent 404684003 (Clinical Finding root)
        filtered = parser.filter_by_hierarchies(concepts, hierarchy_roots=["404684003"])
        assert "25064002" in filtered
        # 386661006 has no active is-a to root, so not included
        assert "386661006" not in filtered

    def test_is_archive_detection(self, tmp_path) -> None:
        zip_path = self._build_rf2_zip(tmp_path)
        parser = RF2Parser(zip_path)
        assert parser._is_archive is True  # noqa: SLF001

        dir_parser = RF2Parser(tmp_path)
        assert dir_parser._is_archive is False  # noqa: SLF001

    def test_missing_rf2_file_returns_empty(self, tmp_path) -> None:
        """Parser handles missing files gracefully."""
        empty_zip = tmp_path / "empty.zip"
        with zipfile.ZipFile(empty_zip, "w"):
            pass

        parser = RF2Parser(empty_zip)
        concepts = parser.parse_concepts()
        assert len(concepts) == 0
