from datetime import datetime

import polars as pl

from dataos.compiler.data_quality_assessor import DataQualityAssessor
from dataos.compiler.schema_mapping_agent import SchemaMappingAgent
from dataos.contracts.requirement_contract import DefinitionStatus, Metric, RequirementContract, Source, TimeSpec, UnitsPolicy
from dataos.ingestion.profiling import profile_dataset


def _contract(**kwargs) -> RequirementContract:
    kwargs.setdefault("objective", "x")
    kwargs.setdefault("sources", [Source(name="orders")])
    return RequirementContract(**kwargs)


def test_clean_data_with_no_mapping_is_pass():
    orders = pl.DataFrame({"order_id": [1, 2, 3]})
    profiles = {"orders": profile_dataset(orders)}

    result = DataQualityAssessor().assess(contract=_contract(), profiles=profiles)

    assert result.fitness == "PASS"
    assert result.issues == []
    assert result.evidence_refs == ["profile:orders"]


def test_moderate_null_rate_on_mapped_field_requires_a_rule():
    orders = pl.DataFrame({"order_id": [1, 2, 3, 4, 5], "customer_age": [30, None, 40, None, 50]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(
        metrics=[Metric(name="avg_age", source_fields=["orders.customer_age"], definition_status=DefinitionStatus.GOVERNED)]
    )
    schema_mapping = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    result = DataQualityAssessor().assess(contract=contract, profiles=profiles, schema_mapping=schema_mapping)

    assert result.fitness == "CONDITIONAL"
    assert any(i.rule == "completeness:orders.customer_age" and i.severity == "REQUIRES_RULE" for i in result.issues)


def test_very_high_null_rate_on_mapped_field_is_blocking():
    orders = pl.DataFrame({"order_id": [1, 2, 3, 4], "note": [None, None, None, "x"]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(
        metrics=[Metric(name="note_metric", source_fields=["orders.note"], definition_status=DefinitionStatus.GOVERNED)]
    )
    schema_mapping = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    result = DataQualityAssessor().assess(contract=contract, profiles=profiles, schema_mapping=schema_mapping)

    assert result.fitness == "FAIL"
    assert any(i.rule == "completeness:orders.note" and i.severity == "BLOCKING" for i in result.issues)


def test_dataset_with_no_candidate_key_is_a_warning():
    orders = pl.DataFrame({"region": ["East", "East", "West"], "channel": ["web", "store", "web"]})
    profiles = {"orders": profile_dataset(orders)}

    result = DataQualityAssessor().assess(contract=_contract(), profiles=profiles)

    assert result.fitness == "PASS"  # WARNING alone does not force CONDITIONAL
    assert any(i.rule == "uniqueness:orders" and i.severity == "WARNING" for i in result.issues)


def test_duplicate_rows_require_a_rule():
    orders = pl.DataFrame({"order_id": [1, 1, 2], "amount": [10.0, 10.0, 20.0]})
    profiles = {"orders": profile_dataset(orders)}

    result = DataQualityAssessor().assess(contract=_contract(), profiles=profiles)

    assert result.fitness == "CONDITIONAL"
    assert any(i.rule == "duplicates:orders" and i.severity == "REQUIRES_RULE" for i in result.issues)


def test_mostly_duplicate_rows_is_blocking():
    orders = pl.DataFrame({"order_id": [1, 1, 1, 1, 2]})
    profiles = {"orders": profile_dataset(orders)}

    result = DataQualityAssessor().assess(contract=_contract(), profiles=profiles)

    assert result.fitness == "FAIL"
    assert any(i.rule == "duplicates:orders" and i.severity == "BLOCKING" for i in result.issues)


def test_cross_source_dtype_mismatch_is_a_warning():
    orders = pl.DataFrame({"customer_id": [1, 2]})
    customers = pl.DataFrame({"customer_id": ["1", "2"]})
    profiles = {"orders": profile_dataset(orders), "customers": profile_dataset(customers)}
    contract = _contract(sources=[Source(name="orders"), Source(name="customers")])

    result = DataQualityAssessor().assess(contract=contract, profiles=profiles)

    assert result.fitness == "PASS"
    assert any(i.rule == "cross_source_dtype:customer_id" for i in result.issues)


def test_money_metric_without_currency_policy_requires_a_rule():
    orders = pl.DataFrame({"order_id": [1], "net_amount": [100.0]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(
        metrics=[Metric(name="revenue", source_fields=["orders.net_amount"], definition_status=DefinitionStatus.GOVERNED)]
    )

    result = DataQualityAssessor().assess(contract=contract, profiles=profiles)

    assert result.fitness == "CONDITIONAL"
    assert any(i.rule == "currency_policy" for i in result.issues)


def test_money_metric_with_currency_policy_declared_has_no_issue():
    orders = pl.DataFrame({"order_id": [1], "net_amount": [100.0]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(
        metrics=[Metric(name="revenue", source_fields=["orders.net_amount"], definition_status=DefinitionStatus.GOVERNED)],
        units=UnitsPolicy(currency="USD"),
    )

    result = DataQualityAssessor().assess(contract=contract, profiles=profiles)

    assert not any(i.rule == "currency_policy" for i in result.issues)


def test_time_range_declared_with_no_date_column_is_blocking():
    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(time=TimeSpec(range="last 30 days"))

    result = DataQualityAssessor().assess(contract=contract, profiles=profiles)

    assert result.fitness == "FAIL"
    assert any(i.rule == "time_coverage" for i in result.issues)


def test_time_range_declared_with_a_date_column_has_no_issue():
    orders = pl.DataFrame({"order_id": [1], "created_at": [datetime(2024, 1, 1)]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(time=TimeSpec(range="last 30 days"))

    result = DataQualityAssessor().assess(contract=contract, profiles=profiles)

    assert not any(i.rule == "time_coverage" for i in result.issues)


def test_narrative_is_none_without_an_llm_client():
    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}

    result = DataQualityAssessor().assess(contract=_contract(), profiles=profiles)

    assert result.narrative is None


def test_narrative_is_populated_without_changing_fitness_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}

    assessor = DataQualityAssessor(llm_client=DeterministicLLMClient())
    result = assessor.assess(contract=_contract(), profiles=profiles)

    assert result.narrative is not None
    assert result.fitness == "PASS"
