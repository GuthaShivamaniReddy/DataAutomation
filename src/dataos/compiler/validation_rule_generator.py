"""Validation Rule Generator.

Source of truth: AI Prompt Library Section 19 "Validation Rule Generator":
"Generate executable validation rules from the Requirement Contract,
source schema, workflow plan, and organization policies... Every
validation must include severity, exact predicate, tolerance, and action
on failure. Avoid vague rules such as 'looks reasonable.'"

Deterministic code, not an LLM call - unlike the Requirement Compiler and
Workflow Planner (which genuinely interpret free text / choose among an
open-ended combination of operations), every rule this module produces is
mechanically derived from fields the `RequirementContract` and `Workflow`
already declare explicitly: `null_policy`, `tolerances`, a `join` step's
`expected_cardinality`, a `validate_schema` step's own params, and
`acceptance_tests`. Section 19's own list also includes "accepted
ranges/enums" and "date boundaries" - `RequirementContract` has no field
for either today, so this generator produces nothing for them rather than
inventing a threshold (Global Constitution rule 1: "Never guess... when
more than one interpretation is plausible").

Every generated `check_function` name is validated against the real
functions in `dataos.validation.checks` (never a string that doesn't
resolve to anything) - the same "never reference an unregistered
primitive" discipline `WorkflowPlanner` applies to operation ids.
`check_function` is `None` only for checks a registry Operation already
enforces structurally at execution time (e.g. join-key uniqueness), where
this generator's job is to surface that guarantee in the manifest, not to
re-implement it as a second, possibly-divergent check.

When constructed with an `LLMClient`, `generate()` additionally asks it to
narrate each already-built `CheckSpec` into `CheckSpec.narrative` - never
to decide `check_function`, `check_args`, `tolerance`, `severity`, or
`on_fail`, all of which are already fixed by the time the model is ever
called. Narration is best-effort: a `CheckSpec` the model did not answer
for simply keeps `narrative=None`, exactly as if no `llm_client` had been
supplied.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

import dataos.validation.checks as checks
from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import VALIDATION_RULE_GENERATOR_SYSTEM_PROMPT
from dataos.contracts.requirement_contract import RequirementContract
from dataos.llm.client import LLMClient
from dataos.workflow.dsl import Workflow, WorkflowStep

_DEFAULT_RECONCILIATION_TOLERANCE = 0.01

Severity = Literal["BLOCKING", "WARNING"]
OnFail = Literal["STOP", "QUARANTINE", "WARN"]


class CheckSpec(BaseModel):
    check_id: str
    stage: str
    """The artifact this check runs against: a workflow step id (its
    output), a step's own input reference (e.g. "source:orders" - used
    when a check needs pre-transform data a later step's output no longer
    has, such as a sum reconciliation against a group-by step's input),
    or a fixed stage name ("contract", "release") for checks not tied to
    one step's data at all."""
    predicate: str
    """Human-readable statement of the assertion - never a vague claim."""
    check_function: str | None = None
    """Name of a function in `dataos.validation.checks`, or None for a
    check a registry Operation already enforces structurally."""
    check_args: dict = Field(default_factory=dict)
    tolerance: float = 0.0
    severity: Severity = "BLOCKING"
    on_fail: OnFail = "STOP"
    requirement_ref: str | None = None
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see class
    docstring) - never authoritative; `predicate` remains the exact
    assertion."""


class ValidationRuleGenerator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def generate(self, *, contract: RequirementContract, workflow: Workflow) -> list[CheckSpec]:
        rules: list[CheckSpec] = []
        rules.extend(self._null_policy_checks(contract))
        rules.extend(self._reconciliation_checks(contract, workflow))
        rules.extend(self._join_cardinality_checks(workflow))
        rules.extend(self._schema_step_checks(workflow))
        rules.extend(self._acceptance_test_checks(contract))

        unknown_functions = sorted(
            {r.check_function for r in rules if r.check_function is not None} - set(checks.__all__)
        )
        if unknown_functions:
            raise ValueError(
                f"ValidationRuleGenerator produced check_function name(s) not present in "
                f"dataos.validation.checks: {unknown_functions}"
            )

        if self._llm_client is not None and rules:
            narratives = narrate(
                self._llm_client,
                system_prompt=VALIDATION_RULE_GENERATOR_SYSTEM_PROMPT,
                items=[
                    {"ref": r.check_id, "facts": {"predicate": r.predicate, "severity": r.severity, "stage": r.stage}}
                    for r in rules
                ],
            )
            rules = [
                r.model_copy(update={"narrative": narratives[r.check_id]}) if r.check_id in narratives else r
                for r in rules
            ]

        return rules

    def _null_policy_checks(self, contract: RequirementContract) -> list[CheckSpec]:
        rules = []
        for column, policy in contract.null_policy.items():
            if policy != "error":
                continue  # "ignore" / "fill:<value>" are cleaning rules, not validation assertions
            rules.append(
                CheckSpec(
                    check_id=f"null_rate:{column}",
                    stage="contract",
                    predicate=f"null_rate({column}) <= 0.0",
                    check_function="check_null_rate",
                    check_args={"column": column, "max_rate": 0.0},
                    severity="BLOCKING",
                    on_fail="STOP",
                    requirement_ref=column,
                )
            )
        return rules

    def _reconciliation_checks(self, contract: RequirementContract, workflow: Workflow) -> list[CheckSpec]:
        covering_step: dict[str, WorkflowStep] = {
            ref: step for step in workflow.steps for ref in step.requirement_refs
        }
        rules = []
        for metric in contract.metrics:
            formula = (metric.formula or "").strip().lower()
            if not formula.startswith("sum("):
                continue
            step = covering_step.get(metric.name)
            if step is None:
                # Nothing has actually computed this metric yet - there is
                # no real column to reconcile against. IndependentVerifier's
                # own requirement-coverage check already flags "no completed
                # step covers this metric" as a defect; fabricating a check
                # against a guessed column name here would be exactly the
                # "guess when more than one interpretation is plausible"
                # the Global Constitution forbids.
                continue
            column = _metric_column(step, metric.name)
            if column is None:
                continue  # the covering step doesn't expose a plain output column for this metric
            if len(step.inputs) != 1:
                continue  # ambiguous which input holds the pre-aggregation raw values
            tolerance = contract.tolerances.get(metric.name, _DEFAULT_RECONCILIATION_TOLERANCE)
            rules.append(
                CheckSpec(
                    check_id=f"reconciliation:{metric.name}",
                    # `sum(column)` is a pre-aggregation value - a group-by
                    # step's own OUTPUT no longer has the source column at
                    # all, only the aggregated metric alias. This check
                    # must run against the step's INPUT, not its output.
                    stage=step.inputs[0],
                    predicate=f"dual_computation(sum, {column}) within {tolerance}",
                    check_function="check_dual_computation",
                    check_args={"column": column, "agg": "sum", "tolerance": tolerance},
                    tolerance=tolerance,
                    severity="BLOCKING",
                    on_fail="QUARANTINE",
                    requirement_ref=metric.name,
                )
            )
        return rules

    def _join_cardinality_checks(self, workflow: Workflow) -> list[CheckSpec]:
        rules = []
        for step in workflow.steps:
            if step.operation_id != "join":
                continue
            expected_cardinality = step.params.get("expected_cardinality", "unknown")
            rules.append(
                CheckSpec(
                    check_id=f"join_cardinality:{step.id}",
                    stage=step.id,
                    predicate=f"join keys satisfy declared cardinality '{expected_cardinality}'",
                    check_function=None,  # enforced structurally by JoinOperation.check_preconditions/run_join
                    check_args={"expected_cardinality": expected_cardinality},
                    severity="BLOCKING",
                    on_fail="STOP",
                    requirement_ref=None,
                )
            )
        return rules

    def _schema_step_checks(self, workflow: Workflow) -> list[CheckSpec]:
        rules = []
        for step in workflow.steps:
            if step.operation_id != "validate_schema":
                continue
            params = step.params

            for column in params.get("required_columns", []):
                rules.append(
                    CheckSpec(
                        check_id=f"required_column:{step.id}:{column}",
                        stage=step.id,
                        predicate=f"column '{column}' exists",
                        check_function=None,  # enforced structurally by ValidateSchemaOperation.check_preconditions
                        check_args={"column": column},
                        severity="BLOCKING",
                        on_fail="STOP",
                    )
                )

            unique_keys = params.get("unique_keys") or []
            if unique_keys:
                rules.append(
                    CheckSpec(
                        check_id=f"unique:{step.id}:{','.join(unique_keys)}",
                        stage=step.id,
                        predicate=f"no duplicate rows on {unique_keys}",
                        check_function="check_no_duplicate_rows",
                        check_args={"subset": unique_keys},
                        severity="BLOCKING",
                        on_fail="STOP",
                    )
                )

            for column, max_rate in (params.get("null_policy") or {}).items():
                rules.append(
                    CheckSpec(
                        check_id=f"null_rate:{step.id}:{column}",
                        stage=step.id,
                        predicate=f"null_rate({column}) <= {max_rate}",
                        check_function="check_null_rate",
                        check_args={"column": column, "max_rate": max_rate},
                        severity="BLOCKING",
                        on_fail="STOP",
                    )
                )
        return rules

    def _acceptance_test_checks(self, contract: RequirementContract) -> list[CheckSpec]:
        return [
            CheckSpec(
                check_id=f"acceptance:{i}",
                stage="release",
                predicate=test,
                check_function=None,  # free text today - no NL-to-predicate compiler exists yet
                severity="BLOCKING",
                on_fail="STOP",
                requirement_ref=test,
            )
            for i, test in enumerate(contract.acceptance_tests)
        ]


def _metric_column(step: WorkflowStep, metric_name: str) -> str | None:
    metrics_param = step.params.get("metrics")
    if not isinstance(metrics_param, list):
        return None
    for m in metrics_param:
        if isinstance(m, dict) and m.get("name") == metric_name:
            return m.get("column")
    return None
