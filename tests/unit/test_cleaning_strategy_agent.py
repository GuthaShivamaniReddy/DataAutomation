import polars as pl

from dataos.compiler.cleaning_strategy_agent import CleaningStrategyAgent
from dataos.compiler.data_quality_assessor import DataQualityAssessor
from dataos.compiler.schema_mapping_agent import SchemaMappingAgent
from dataos.contracts.requirement_contract import DefinitionStatus, Metric, RequirementContract, Source
from dataos.ingestion.profiling import profile_dataset


def _contract(**kwargs) -> RequirementContract:
    kwargs.setdefault("objective", "x")
    kwargs.setdefault("sources", [Source(name="orders")])
    return RequirementContract(**kwargs)


def test_error_null_policy_produces_a_non_lossy_non_approval_rule():
    contract = _contract(null_policy={"net_amount": "error"})

    result = CleaningStrategyAgent().propose(contract=contract)

    assert result.blocking_items == []
    rule = result.rules[0]
    assert rule.rule_id == "null_policy:net_amount"
    assert rule.lossy is False
    assert rule.approval_required is False


def test_ignore_null_policy_produces_a_non_lossy_non_approval_rule():
    contract = _contract(null_policy={"customer_age": "ignore"})

    result = CleaningStrategyAgent().propose(contract=contract)

    rule = result.rules[0]
    assert rule.lossy is False
    assert rule.approval_required is False


def test_fill_null_policy_produces_a_lossy_approval_required_rule():
    contract = _contract(null_policy={"customer_age": "fill:0"})

    result = CleaningStrategyAgent().propose(contract=contract)

    rule = result.rules[0]
    assert rule.action == "fill with the governed literal value '0'"
    assert rule.lossy is True
    assert rule.approval_required is True


def test_unrecognized_null_policy_is_a_blocking_item_not_a_guessed_rule():
    contract = _contract(null_policy={"customer_age": "some_custom_thing"})

    result = CleaningStrategyAgent().propose(contract=contract)

    assert result.rules == []
    assert result.blocking_items[0].field == "customer_age"


def test_completeness_issue_without_null_policy_is_blocking():
    orders = pl.DataFrame({"order_id": [1, 2, 3, 4, 5], "customer_age": [30, None, 40, None, 50]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(
        metrics=[Metric(name="avg_age", source_fields=["orders.customer_age"], definition_status=DefinitionStatus.GOVERNED)]
    )
    schema_mapping = SchemaMappingAgent().map(contract=contract, profiles=profiles)
    quality_report = DataQualityAssessor().assess(contract=contract, profiles=profiles, schema_mapping=schema_mapping)

    result = CleaningStrategyAgent().propose(contract=contract, quality_report=quality_report)

    assert result.rules == []
    assert any(b.field == "orders.customer_age" for b in result.blocking_items)


def test_completeness_issue_with_null_policy_is_resolved_by_the_declared_policy():
    orders = pl.DataFrame({"order_id": [1, 2, 3, 4, 5], "customer_age": [30, None, 40, None, 50]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(
        metrics=[Metric(name="avg_age", source_fields=["orders.customer_age"], definition_status=DefinitionStatus.GOVERNED)],
        null_policy={"customer_age": "ignore"},
    )
    schema_mapping = SchemaMappingAgent().map(contract=contract, profiles=profiles)
    quality_report = DataQualityAssessor().assess(contract=contract, profiles=profiles, schema_mapping=schema_mapping)

    result = CleaningStrategyAgent().propose(contract=contract, quality_report=quality_report)

    assert not any(b.field == "orders.customer_age" for b in result.blocking_items)
    assert any(r.rule_id == "null_policy:customer_age" for r in result.rules)


def test_duplicate_rows_issue_produces_a_lossy_approval_required_dedup_rule():
    orders = pl.DataFrame({"order_id": [1, 1, 2]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract()
    quality_report = DataQualityAssessor().assess(contract=contract, profiles=profiles)

    result = CleaningStrategyAgent().propose(contract=contract, quality_report=quality_report)

    rule = next(r for r in result.rules if r.rule_id == "dedup:orders")
    assert rule.lossy is True
    assert rule.approval_required is True


def test_currency_policy_issue_is_never_resolved_by_a_guessed_rule():
    orders = pl.DataFrame({"order_id": [1], "net_amount": [100.0]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract(
        metrics=[Metric(name="revenue", source_fields=["orders.net_amount"], definition_status=DefinitionStatus.GOVERNED)]
    )
    quality_report = DataQualityAssessor().assess(contract=contract, profiles=profiles)

    result = CleaningStrategyAgent().propose(contract=contract, quality_report=quality_report)

    assert not any(r.rule_id.startswith("currency") for r in result.rules)
    assert any(b.field == "currency_policy" for b in result.blocking_items)


def test_warning_severity_issue_is_not_acted_on():
    orders = pl.DataFrame({"region": ["East", "East", "West"], "channel": ["web", "store", "web"]})
    profiles = {"orders": profile_dataset(orders)}
    contract = _contract()
    quality_report = DataQualityAssessor().assess(contract=contract, profiles=profiles)
    assert any(i.severity == "WARNING" for i in quality_report.issues)  # sanity check on the fixture

    result = CleaningStrategyAgent().propose(contract=contract, quality_report=quality_report)

    assert result.rules == []
    assert result.blocking_items == []


def test_no_findings_produces_an_empty_plan():
    result = CleaningStrategyAgent().propose(contract=_contract())

    assert result.rules == []
    assert result.blocking_items == []


def test_narrative_is_none_without_an_llm_client():
    contract = _contract(null_policy={"net_amount": "error"})

    result = CleaningStrategyAgent().propose(contract=contract)

    assert result.narrative is None


def test_narrative_is_populated_without_changing_rules_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    contract = _contract(null_policy={"net_amount": "error"})

    agent = CleaningStrategyAgent(llm_client=DeterministicLLMClient())
    result = agent.propose(contract=contract)

    assert result.narrative is not None
    assert result.rules[0].rule_id == "null_policy:net_amount"
