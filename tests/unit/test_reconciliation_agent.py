import polars as pl

from dataos.compiler.reconciliation_agent import ReconciliationAgent
from dataos.contracts.requirement_contract import Metric, RequirementContract, Source
from dataos.workflow.dsl import Workflow, WorkflowStep


def _contract(**kwargs) -> RequirementContract:
    defaults = dict(objective="x", sources=[Source(name="orders")])
    defaults.update(kwargs)
    return RequirementContract(**defaults)


def _workflow(steps: list[WorkflowStep]) -> Workflow:
    return Workflow(workflow_version=1, requirement_contract_id="rc_1", sources=["orders"], steps=steps)


def test_cast_step_row_conservation_passes_when_rows_preserved():
    before = pl.DataFrame({"order_date": ["2024-01-01", "2024-01-02"]})
    after = pl.DataFrame({"order_date": [None, None]})  # cast output, same row count
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="cast", operation_version="1.0", inputs=["source:orders"])]
    )

    report = ReconciliationAgent().reconcile(
        contract=_contract(), workflow=workflow, artifacts={"source:orders": before, "s1": after}
    )

    assert report.status == "PASS"
    assert report.tests[0].name == "row_count_conservation:s1"
    assert report.tests[0].passed is True


def test_cast_step_row_conservation_fails_when_rows_silently_dropped():
    before = pl.DataFrame({"a": [1, 2, 3]})
    after = pl.DataFrame({"a": [1, 2]})  # a row vanished - should never happen for cast
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="cast", operation_version="1.0", inputs=["source:orders"])]
    )

    report = ReconciliationAgent().reconcile(
        contract=_contract(), workflow=workflow, artifacts={"source:orders": before, "s1": after}
    )

    assert report.status == "FAIL"
    assert report.tests[0].passed is False


def test_filtering_operations_are_not_subject_to_row_conservation():
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="select_filter", operation_version="1.0", inputs=["source:orders"])]
    )
    report = ReconciliationAgent().reconcile(contract=_contract(), workflow=workflow, artifacts={})
    assert report.tests == []
    assert report.status == "PASS"


def test_group_subtotals_reconcile_to_ungrouped_total():
    input_df = pl.DataFrame({"region": ["East", "East", "West"], "net_amount": [10.0, 20.0, 5.0]})
    output_df = pl.DataFrame({"region": ["East", "West"], "revenue": [30.0, 5.0]})
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"group_by": ["region"], "metrics": [{"name": "revenue", "column": "net_amount", "fn": "sum"}]},
                requirement_refs=["revenue"],
            )
        ]
    )
    contract = _contract(metrics=[Metric(name="revenue", formula="sum(orders.net_amount)")])

    report = ReconciliationAgent().reconcile(
        contract=contract, workflow=workflow, artifacts={"source:orders": input_df, "s1": output_df}
    )

    assert report.status == "PASS"
    test = next(t for t in report.tests if t.name == "group_subtotals_sum_to_total:revenue")
    assert test.observed == 35.0
    assert test.expected == 35.0
    assert test.passed is True


def test_group_subtotals_mismatch_fails():
    input_df = pl.DataFrame({"region": ["East", "West"], "net_amount": [10.0, 5.0]})
    output_df = pl.DataFrame({"region": ["East", "West"], "revenue": [999.0, 5.0]})  # corrupted subtotal
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"group_by": ["region"], "metrics": [{"name": "revenue", "column": "net_amount", "fn": "sum"}]},
                requirement_refs=["revenue"],
            )
        ]
    )
    contract = _contract(metrics=[Metric(name="revenue", formula="sum(orders.net_amount)")])

    report = ReconciliationAgent().reconcile(
        contract=contract, workflow=workflow, artifacts={"source:orders": input_df, "s1": output_df}
    )

    assert report.status == "FAIL"


def test_ungrouped_sum_metric_has_no_anchor_and_is_unavailable():
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"metrics": [{"name": "revenue", "column": "net_amount", "fn": "sum"}]},  # no group_by
                requirement_refs=["revenue"],
            )
        ]
    )
    contract = _contract(metrics=[Metric(name="revenue", formula="sum(orders.net_amount)")])

    report = ReconciliationAgent().reconcile(contract=contract, workflow=workflow, artifacts={})

    assert report.status == "UNAVAILABLE"
    assert report.tests == []
    assert any("revenue" in r for r in report.blocking_reasons)


def test_non_sum_metric_and_unrelated_steps_produce_no_tests():
    contract = _contract(metrics=[Metric(name="order_count", formula="count(orders.order_id)")])
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="deduplicate", operation_version="1.0", inputs=["source:orders"])]
    )
    report = ReconciliationAgent().reconcile(contract=contract, workflow=workflow, artifacts={})
    assert report.tests == []
    assert report.blocking_reasons == []
    assert report.status == "PASS"
