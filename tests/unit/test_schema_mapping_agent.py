from datetime import datetime
from pathlib import Path

import polars as pl

from dataos.compiler.schema_mapping_agent import SchemaMappingAgent
from dataos.contracts.requirement_contract import DefinitionStatus, Metric, RequirementContract, Source
from dataos.ingestion.profiling import profile_dataset
from dataos.semantics.dictionary import SemanticDictionary

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "semantic_dictionary" / "sample_dictionary.json"


def test_governed_metric_source_fields_all_resolve():
    orders = pl.DataFrame({"order_id": [1, 2], "net_amount": [100.0, 200.0]})
    refunds = pl.DataFrame({"refund_id": [1], "amount": [10.0]})
    profiles = {"orders": profile_dataset(orders), "refunds": profile_dataset(refunds)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders"), Source(name="refunds")],
        metrics=[
            Metric(
                name="recognized_revenue",
                source_fields=["orders.net_amount", "refunds.amount"],
                definition_status=DefinitionStatus.GOVERNED,
            )
        ],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert len(result.mappings) == 2
    assert all(m.status == "MAPPED" and m.confidence == "HIGH" for m in result.mappings)
    assert result.blocking_items == []


def test_source_field_with_missing_dataset_is_blocking():
    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[Metric(name="revenue", source_fields=["refunds.amount"], definition_status=DefinitionStatus.GOVERNED)],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.mappings[0].status == "MISSING"
    assert any("no unambiguous source field" in b.reason for b in result.blocking_items)


def test_source_field_with_missing_column_is_blocking():
    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[Metric(name="revenue", source_fields=["orders.net_amount"], definition_status=DefinitionStatus.GOVERNED)],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.mappings[0].status == "MISSING"
    assert "does not exist" in result.mappings[0].evidence[0]


def test_money_concept_with_non_numeric_dtype_is_downgraded_to_missing():
    orders = pl.DataFrame({"order_id": [1], "net_amount": ["100.00"]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[Metric(name="revenue", source_fields=["orders.net_amount"], definition_status=DefinitionStatus.GOVERNED)],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.mappings[0].status == "MISSING"
    assert any("non-numeric dtype" in e for e in result.mappings[0].evidence)


def test_dimension_resolved_via_governed_dictionary():
    dictionary = SemanticDictionary.load_from_json(_FIXTURE)
    orders = pl.DataFrame({"order_id": [1], "region": ["East"]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(objective="x", sources=[Source(name="orders")], dimensions=["region"])

    result = SchemaMappingAgent(dictionary=dictionary).map(contract=contract, profiles=profiles)

    mapping = result.mappings[0]
    assert mapping.status == "MAPPED"
    assert mapping.confidence == "HIGH"
    assert mapping.dataset == "orders"
    assert mapping.column == "region"


def test_dimension_without_dictionary_resolves_by_unique_column_name():
    orders = pl.DataFrame({"order_id": [1], "channel": ["web"]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(objective="x", sources=[Source(name="orders")], dimensions=["channel"])

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.mappings[0].status == "MAPPED"
    assert result.mappings[0].confidence == "MEDIUM"


def test_dimension_ambiguous_across_multiple_datasets():
    a = pl.DataFrame({"status": ["open"]})
    b = pl.DataFrame({"status": ["closed"]})
    profiles = {"a": profile_dataset(a), "b": profile_dataset(b)}
    contract = RequirementContract(
        objective="x", sources=[Source(name="a"), Source(name="b")], dimensions=["status"]
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.mappings[0].status == "AMBIGUOUS"
    assert any(b.concept == "status" for b in result.blocking_items)


def test_dimension_missing_when_no_column_matches():
    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(objective="x", sources=[Source(name="orders")], dimensions=["region"])

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.mappings[0].status == "MISSING"


def test_two_concepts_mapped_to_the_same_column_is_blocking():
    orders = pl.DataFrame({"order_id": [1], "amount": [10.0]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="revenue", source_fields=["orders.amount"], definition_status=DefinitionStatus.GOVERNED),
            Metric(name="cost", source_fields=["orders.amount"], definition_status=DefinitionStatus.GOVERNED),
        ],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert all(m.status == "MAPPED" for m in result.mappings)
    assert any("mapped to multiple distinct concepts" in b.reason for b in result.blocking_items)


def test_identifier_with_nulls_gets_a_non_blocking_evidence_note():
    orders = pl.DataFrame({"order_id": [1, None, 3]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[Metric(name="order_count", source_fields=["orders.order_id"], definition_status=DefinitionStatus.GOVERNED)],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    mapping = result.mappings[0]
    assert mapping.status == "MAPPED"
    assert any("null value" in e for e in mapping.evidence)
    assert not any("null value" in b.reason for b in result.blocking_items)


def test_date_column_gets_a_time_role_evidence_tag():
    orders = pl.DataFrame({"order_id": [1], "created_at": [datetime(2024, 1, 1)]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(objective="x", sources=[Source(name="orders")], dimensions=["created_at"])

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    mapping = result.mappings[0]
    assert mapping.status == "MAPPED"
    assert "time_role=created_time" in mapping.evidence


def test_metric_still_blocked_upstream_is_skipped():
    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[Metric(name="revenue", definition_status=DefinitionStatus.AMBIGUOUS)],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.mappings == []
    assert result.blocking_items == []


def test_narrative_is_none_without_an_llm_client():
    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[Metric(name="order_count", source_fields=["orders.order_id"], definition_status=DefinitionStatus.GOVERNED)],
    )

    result = SchemaMappingAgent().map(contract=contract, profiles=profiles)

    assert result.narrative is None


def test_narrative_is_populated_without_changing_mappings_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    orders = pl.DataFrame({"order_id": [1]})
    profiles = {"orders": profile_dataset(orders)}
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[Metric(name="order_count", source_fields=["orders.order_id"], definition_status=DefinitionStatus.GOVERNED)],
    )

    agent = SchemaMappingAgent(llm_client=DeterministicLLMClient())
    result = agent.map(contract=contract, profiles=profiles)

    assert result.narrative is not None
    assert result.mappings[0].status == "MAPPED"
