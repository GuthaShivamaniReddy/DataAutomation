from dataos.compiler.validation_rule_generator import ValidationRuleGenerator
from dataos.contracts.requirement_contract import Metric, RequirementContract, Source
from dataos.workflow.dsl import Workflow, WorkflowStep


def _contract(**kwargs) -> RequirementContract:
    defaults = dict(objective="x", sources=[Source(name="orders")])
    defaults.update(kwargs)
    return RequirementContract(**defaults)


def _workflow(steps: list[WorkflowStep]) -> Workflow:
    return Workflow(workflow_version=1, requirement_contract_id="rc_1", sources=["orders"], steps=steps)


def test_error_null_policy_generates_blocking_check():
    contract = _contract(null_policy={"net_amount": "error"})
    rules = ValidationRuleGenerator().generate(contract=contract, workflow=_workflow([]))

    assert len(rules) == 1
    rule = rules[0]
    assert rule.check_id == "null_rate:net_amount"
    assert rule.check_function == "check_null_rate"
    assert rule.check_args == {"column": "net_amount", "max_rate": 0.0}
    assert rule.severity == "BLOCKING"
    assert rule.on_fail == "STOP"


def test_ignore_and_fill_null_policy_generate_no_checks():
    contract = _contract(null_policy={"a": "ignore", "b": "fill:0"})
    rules = ValidationRuleGenerator().generate(contract=contract, workflow=_workflow([]))
    assert rules == []


def test_sum_metric_generates_dual_computation_check_with_declared_tolerance():
    contract = _contract(
        metrics=[Metric(name="revenue", formula="sum(orders.net_amount)")],
        tolerances={"revenue": 5.0},
    )
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"metrics": [{"name": "revenue", "column": "net_amount", "fn": "sum"}]},
                requirement_refs=["revenue"],
            )
        ]
    )
    rules = ValidationRuleGenerator().generate(contract=contract, workflow=workflow)

    reconciliation = [r for r in rules if r.check_id == "reconciliation:revenue"][0]
    # Runs against the step's INPUT, not its output - a group-by step's
    # output no longer has the raw "net_amount" column to sum.
    assert reconciliation.stage == "source:orders"
    assert reconciliation.check_function == "check_dual_computation"
    assert reconciliation.check_args == {"column": "net_amount", "agg": "sum", "tolerance": 5.0}
    assert reconciliation.tolerance == 5.0
    assert reconciliation.on_fail == "QUARANTINE"


def test_sum_metric_without_declared_tolerance_uses_default():
    contract = _contract(metrics=[Metric(name="revenue", formula="sum(orders.net_amount)")])
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"metrics": [{"name": "revenue", "column": "net_amount", "fn": "sum"}]},
                requirement_refs=["revenue"],
            )
        ]
    )
    rules = ValidationRuleGenerator().generate(contract=contract, workflow=workflow)

    reconciliation = [r for r in rules if r.check_id == "reconciliation:revenue"][0]
    assert reconciliation.tolerance == 0.01
    assert reconciliation.stage == "source:orders"


def test_sum_metric_with_no_covering_step_generates_no_reconciliation_check():
    # The metric was declared but never actually computed by any step -
    # there is no real column to reconcile against, and fabricating one
    # would be exactly the kind of guess the platform must never make.
    # IndependentVerifier's own requirement-coverage check is what flags
    # this case as a defect, not the Validation Rule Generator.
    contract = _contract(metrics=[Metric(name="revenue", formula="sum(orders.net_amount)")])
    rules = ValidationRuleGenerator().generate(contract=contract, workflow=_workflow([]))
    assert rules == []


def test_non_sum_metric_generates_no_reconciliation_check():
    contract = _contract(metrics=[Metric(name="order_count", formula="count(orders.order_id)")])
    rules = ValidationRuleGenerator().generate(contract=contract, workflow=_workflow([]))
    assert rules == []


def test_join_step_generates_structural_cardinality_check():
    workflow = _workflow(
        [
            WorkflowStep(
                id="j1",
                operation_id="join",
                operation_version="1.0",
                inputs=["source:orders", "source:customers"],
                params={"expected_cardinality": "many_to_one"},
            )
        ]
    )
    rules = ValidationRuleGenerator().generate(contract=_contract(), workflow=workflow)

    assert len(rules) == 1
    rule = rules[0]
    assert rule.check_id == "join_cardinality:j1"
    assert rule.check_function is None
    assert "many_to_one" in rule.predicate


def test_validate_schema_step_generates_required_column_unique_and_null_checks():
    workflow = _workflow(
        [
            WorkflowStep(
                id="v1",
                operation_id="validate_schema",
                operation_version="1.0",
                inputs=["source:orders"],
                params={
                    "required_columns": ["order_id"],
                    "unique_keys": ["order_id"],
                    "null_policy": {"region": 0.0},
                },
            )
        ]
    )
    rules = ValidationRuleGenerator().generate(contract=_contract(), workflow=workflow)
    rule_ids = {r.check_id for r in rules}

    assert "required_column:v1:order_id" in rule_ids
    assert "unique:v1:order_id" in rule_ids
    assert "null_rate:v1:region" in rule_ids

    unique_rule = next(r for r in rules if r.check_id == "unique:v1:order_id")
    assert unique_rule.check_function == "check_no_duplicate_rows"
    assert unique_rule.check_args == {"subset": ["order_id"]}


def test_acceptance_tests_generate_one_check_each():
    contract = _contract(acceptance_tests=["row_count > 0", "no negative amounts"])
    rules = ValidationRuleGenerator().generate(contract=contract, workflow=_workflow([]))

    assert [r.check_id for r in rules] == ["acceptance:0", "acceptance:1"]
    assert rules[0].predicate == "row_count > 0"
    assert rules[0].requirement_ref == "row_count > 0"
    assert rules[0].check_function is None
