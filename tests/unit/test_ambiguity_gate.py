import polars as pl

from dataos.compiler.ambiguity_gate import AmbiguityGate
from dataos.compiler.extraction import ExtractedMetric, RawExtraction
from dataos.compiler.semantic_resolver import ResolvedTerm
from dataos.ingestion.profiling import profile_dataset


def test_ambiguous_resolved_metric_blocks():
    gate = AmbiguityGate()
    extraction = RawExtraction(objective="x", metrics=[ExtractedMetric(term="revenue")])
    resolved = [ResolvedTerm(term="revenue", status="AMBIGUOUS", candidate_ids=["a", "b"])]

    result = gate.evaluate(extraction=extraction, resolved_metrics=resolved, sources=["orders"])

    assert result.decision == "BLOCK"
    assert any("revenue" in c.issue for c in result.clarifications)
    assert result.clarifications[0].options == ["a", "b"]


def test_material_term_ungoverned_blocks():
    gate = AmbiguityGate()
    extraction = RawExtraction(objective="x", metrics=[ExtractedMetric(term="growth")])
    resolved = [ResolvedTerm(term="growth", status="UNGOVERNED")]

    result = gate.evaluate(extraction=extraction, resolved_metrics=resolved, sources=["orders"])
    assert result.decision == "BLOCK"


def test_ungoverned_with_explicit_formula_does_not_block():
    gate = AmbiguityGate()
    extraction = RawExtraction(
        objective="x",
        metrics=[ExtractedMetric(term="widget_ratio", formula="sum(widgets) / sum(total)")],
    )
    resolved = [ResolvedTerm(term="widget_ratio", status="UNGOVERNED")]

    result = gate.evaluate(extraction=extraction, resolved_metrics=resolved, sources=["orders"])
    assert result.decision == "PROCEED"
    assert result.clarifications == []


def test_governed_metric_does_not_block():
    gate = AmbiguityGate()
    extraction = RawExtraction(objective="x", metrics=[ExtractedMetric(term="order_count")])
    resolved = [ResolvedTerm(term="order_count", status="GOVERNED")]

    result = gate.evaluate(extraction=extraction, resolved_metrics=resolved, sources=["orders"])
    assert result.decision == "PROCEED"


def test_no_sources_blocks():
    gate = AmbiguityGate()
    extraction = RawExtraction(objective="x")
    result = gate.evaluate(extraction=extraction, resolved_metrics=[], sources=[])
    assert result.decision == "BLOCK"
    assert any("source" in c.issue for c in result.clarifications)


def test_date_bucket_without_timezone_blocks():
    gate = AmbiguityGate()
    extraction = RawExtraction(objective="x", group_by=["month"])
    result = gate.evaluate(extraction=extraction, resolved_metrics=[], sources=["orders"])
    assert result.decision == "BLOCK"
    assert any("timezone" in c.question.lower() for c in result.clarifications)


def test_date_bucket_with_timezone_does_not_block_on_that_check():
    gate = AmbiguityGate()
    extraction = RawExtraction(objective="x", group_by=["month"], timezone="America/New_York")
    result = gate.evaluate(extraction=extraction, resolved_metrics=[], sources=["orders"])
    assert not any("timezone" in c.question.lower() for c in result.clarifications)


def test_multiple_plausible_columns_for_ungoverned_term_blocks():
    df = pl.DataFrame(
        {
            "gross_sales": [1, 2],
            "net_sales": [1, 2],
            "recognized_revenue": [1, 2],
        }
    )
    profile = profile_dataset(df)

    gate = AmbiguityGate()
    extraction = RawExtraction(objective="x", metrics=[ExtractedMetric(term="sales")])
    resolved = [ResolvedTerm(term="sales", status="UNGOVERNED")]

    result = gate.evaluate(
        extraction=extraction, resolved_metrics=resolved, sources=["orders"], profile=profile
    )
    assert result.decision == "BLOCK"
    column_clarifications = [c for c in result.clarifications if "column" in c.issue]
    assert column_clarifications
    assert set(column_clarifications[0].options) == {"gross_sales", "net_sales"}
