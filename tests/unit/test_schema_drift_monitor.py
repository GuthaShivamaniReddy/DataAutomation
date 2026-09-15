import polars as pl

from dataos.compiler.schema_drift_monitor import SchemaDriftMonitor
from dataos.ingestion.profiling import profile_dataset
from dataos.semantics.models import SemanticCalendar, SemanticMetric
from dataos.semantics.dictionary import SemanticDictionary


def _profile(df: pl.DataFrame):
    return profile_dataset(df)


def test_identical_schema_reports_no_drift():
    df = pl.DataFrame({"order_id": [1, 2, 3], "region": ["East", "West", "East"]})
    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(df), current_profile=_profile(df.clone())
    )

    assert report.drift_status == "NONE"
    assert report.changes == []
    assert report.automation_action == "CONTINUE"


def test_removed_column_is_breaking():
    baseline = pl.DataFrame({"order_id": [1, 2], "region": ["East", "West"]})
    current = pl.DataFrame({"order_id": [1, 2]})

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert report.drift_status == "BREAKING"
    assert report.automation_action == "STOP"
    removed = [c for c in report.changes if c.kind == "column_removed"]
    assert removed and removed[0].column == "region"


def test_added_column_is_compatible_only():
    baseline = pl.DataFrame({"order_id": [1, 2]})
    current = pl.DataFrame({"order_id": [1, 2], "region": ["East", "West"]})

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert report.drift_status == "COMPATIBLE"
    assert report.automation_action == "CONTINUE"
    assert report.changes[0].kind == "column_added"


def test_renamed_column_is_reported_as_independent_add_and_remove_never_matched():
    # Constitution / Section 25: never automatically remap a renamed field
    # by similarity - "region" -> "territory" must show up as one removal
    # and one addition, never a single "renamed" entry.
    baseline = pl.DataFrame({"order_id": [1, 2], "region": ["East", "West"]})
    current = pl.DataFrame({"order_id": [1, 2], "territory": ["East", "West"]})

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    kinds = {(c.kind, c.column) for c in report.changes}
    assert ("column_removed", "region") in kinds
    assert ("column_added", "territory") in kinds
    assert not any(c.kind == "renamed" for c in report.changes)


def test_dtype_change_is_breaking():
    baseline = pl.DataFrame({"amount": [1, 2, 3]})  # Int64
    current = pl.DataFrame({"amount": ["1", "2", "3"]})  # Utf8

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert report.drift_status == "BREAKING"
    dtype_changes = [c for c in report.changes if c.kind == "dtype_changed"]
    assert dtype_changes and dtype_changes[0].column == "amount"


def test_key_uniqueness_lost_is_breaking():
    baseline = pl.DataFrame({"order_id": [1, 2, 3]})  # unique -> candidate key
    current = pl.DataFrame({"order_id": [1, 1, 3]})  # duplicate appeared

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert report.drift_status == "BREAKING"
    assert any(c.kind == "key_uniqueness_lost" and c.column == "order_id" for c in report.changes)


def test_null_rate_increase_beyond_threshold_is_review_required():
    # Values are not all-distinct in either frame, so neither profile
    # accidentally flags net_amount as a candidate key - isolates this
    # test to the null-rate check alone.
    baseline = pl.DataFrame({"net_amount": [1.0, 1.0, 2.0, 2.0]})  # 0% null
    current = pl.DataFrame({"net_amount": [1.0, None, None, 2.0]})  # 50% null

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert report.drift_status == "REVIEW_REQUIRED"
    assert report.automation_action == "QUARANTINE"
    assert any(c.kind == "null_rate_increased" and c.column == "net_amount" for c in report.changes)


def test_small_null_rate_increase_within_threshold_is_not_flagged():
    baseline = pl.DataFrame({"x": list(range(100))})  # 0% null
    values = [float(i) for i in range(100)]
    values[0] = None  # 1% null - below default 5% threshold
    current = pl.DataFrame({"x": values})

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert not any(c.kind == "null_rate_increased" for c in report.changes)


def test_large_row_count_shift_is_review_required():
    baseline = pl.DataFrame({"x": list(range(100))})
    current = pl.DataFrame({"x": list(range(10))})  # 90% drop

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert report.drift_status == "REVIEW_REQUIRED"
    assert any(c.kind == "row_count_shift" for c in report.changes)


def test_date_coverage_regression_is_review_required():
    baseline = pl.DataFrame({"order_date": ["2024-01-01", "2024-06-01"]}).with_columns(
        pl.col("order_date").str.to_date()
    )
    current = pl.DataFrame({"order_date": ["2024-01-01", "2024-02-01"]}).with_columns(
        pl.col("order_date").str.to_date()
    )

    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(baseline), current_profile=_profile(current)
    )

    assert any(c.kind == "date_coverage_regressed" and c.column == "order_date" for c in report.changes)


def test_removed_governed_metric_is_breaking():
    baseline_dict = SemanticDictionary(
        metrics={"order_count": SemanticMetric(semantic_id="order_count", name="order_count", formula="count(orders.order_id)")}
    )
    current_dict = SemanticDictionary(metrics={})

    df = pl.DataFrame({"order_id": [1]})
    report = SchemaDriftMonitor().compare(
        source_name="orders",
        baseline_profile=_profile(df),
        current_profile=_profile(df.clone()),
        baseline_dictionary=baseline_dict,
        current_dictionary=current_dict,
    )

    assert report.drift_status == "BREAKING"
    assert any(c.kind == "semantic_definition_removed" for c in report.changes)


def test_changed_governed_metric_version_is_review_required():
    baseline_dict = SemanticDictionary(
        metrics={
            "order_count": SemanticMetric(
                semantic_id="order_count", name="order_count", formula="count(orders.order_id)", version=1
            )
        }
    )
    current_dict = SemanticDictionary(
        metrics={
            "order_count": SemanticMetric(
                semantic_id="order_count", name="order_count", formula="count(orders.id)", version=2
            )
        }
    )

    df = pl.DataFrame({"order_id": [1]})
    report = SchemaDriftMonitor().compare(
        source_name="orders",
        baseline_profile=_profile(df),
        current_profile=_profile(df.clone()),
        baseline_dictionary=baseline_dict,
        current_dictionary=current_dict,
    )

    assert report.drift_status == "REVIEW_REQUIRED"
    assert any(c.kind == "semantic_definition_changed" for c in report.changes)


def test_calendar_policy_change_is_review_required():
    baseline_dict = SemanticDictionary(calendar=SemanticCalendar(timezone="UTC"))
    current_dict = SemanticDictionary(calendar=SemanticCalendar(timezone="America/New_York"))

    df = pl.DataFrame({"order_id": [1]})
    report = SchemaDriftMonitor().compare(
        source_name="orders",
        baseline_profile=_profile(df),
        current_profile=_profile(df.clone()),
        baseline_dictionary=baseline_dict,
        current_dictionary=current_dict,
    )

    assert report.drift_status == "REVIEW_REQUIRED"
    assert any(c.kind == "calendar_policy_changed" for c in report.changes)


def test_narrative_is_none_without_an_llm_client():
    df = pl.DataFrame({"order_id": [1]})
    report = SchemaDriftMonitor().compare(
        source_name="orders", baseline_profile=_profile(df), current_profile=_profile(df.clone())
    )
    assert report.narrative is None


def test_narrative_is_populated_without_changing_drift_status_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    df = pl.DataFrame({"order_id": [1]})
    monitor = SchemaDriftMonitor(llm_client=DeterministicLLMClient())
    report = monitor.compare(source_name="orders", baseline_profile=_profile(df), current_profile=_profile(df.clone()))

    assert report.narrative is not None
    assert report.drift_status == "NONE"  # the narrative never changes the decision
