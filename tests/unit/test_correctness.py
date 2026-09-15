from dataos.contracts.correctness import CorrectnessClass, required_evidence_for


def test_c0_requires_only_basic_evidence():
    ev = required_evidence_for(CorrectnessClass.C0_INFORMATIONAL)
    assert ev == ["parse_success", "source_hash", "schema_evidence"]


def test_c4_is_superset_of_all_lower_classes():
    c1 = set(required_evidence_for(CorrectnessClass.C1_TRANSFORMATIONAL))
    c2 = set(required_evidence_for(CorrectnessClass.C2_ANALYTICAL))
    c3 = set(required_evidence_for(CorrectnessClass.C3_PREDICTIVE))
    c4 = set(required_evidence_for(CorrectnessClass.C4_AUTHORITATIVE))

    assert c1 <= c2 <= c3 <= c4
    assert "human_approval" in c4
    assert "reconciliation" in c4
    assert "human_approval" not in c3
