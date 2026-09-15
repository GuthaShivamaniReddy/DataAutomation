import pytest

from dataos.semantics.dictionary import SemanticDictionary
from dataos.semantics.models import SemanticMetric


def test_load_from_json_round_trip(sample_semantic_dictionary, tmp_path):
    assert set(sample_semantic_dictionary.metrics) == {"recognized_revenue", "gross_revenue", "order_count"}
    assert sample_semantic_dictionary.calendar.timezone == "America/New_York"

    out_path = tmp_path / "roundtrip.json"
    sample_semantic_dictionary.save_to_json(out_path)
    reloaded = SemanticDictionary.load_from_json(out_path)
    assert reloaded == sample_semantic_dictionary


def test_find_metric_candidates_by_alias(sample_semantic_dictionary):
    candidates = sample_semantic_dictionary.find_metric_candidates("revenue")
    assert {c.semantic_id for c in candidates} == {"recognized_revenue", "gross_revenue"}


def test_find_metric_candidates_unique_match(sample_semantic_dictionary):
    candidates = sample_semantic_dictionary.find_metric_candidates("orders")
    assert [c.semantic_id for c in candidates] == ["order_count"]


def test_find_metric_candidates_no_match(sample_semantic_dictionary):
    assert sample_semantic_dictionary.find_metric_candidates("churn rate") == []


def test_find_metric_candidates_normalizes_underscores_and_case(sample_semantic_dictionary):
    candidates = sample_semantic_dictionary.find_metric_candidates("Recognized_Revenue")
    assert [c.semantic_id for c in candidates] == ["recognized_revenue"]


def test_add_metric_never_silently_overwrites_governed_definition(sample_semantic_dictionary):
    duplicate = SemanticMetric(
        semantic_id="recognized_revenue",
        name="recognized_revenue",
        formula="something_else",
    )
    with pytest.raises(ValueError):
        sample_semantic_dictionary.add_metric(duplicate)
