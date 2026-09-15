"""Independent Reconciliation Agent.

Source of truth: AI Prompt Library Section 20 "Reconciliation Agent":
"Verify material totals using independent evidence or an independently
implemented calculation path... Do not accept the primary calculation as
its own proof. If no independent anchor exists, explicitly state that
reconciliation is unavailable and require the alternative verification
policy."

Deterministic code, not an LLM call - same reasoning as every other
gate/generator in this package: an independent check is only independent
if it is not itself a model's judgment call. Every test here re-derives
its answer from dataframes read fresh from the artifact store - never
from a step's own self-reported evidence dict - so a bug in an
operation's own bookkeeping cannot pass its own reconciliation.

Covers the two of Section 20's example patterns that are honestly
derivable from this codebase's actual operations today, and refuses to
fabricate the rest:
  - "transformed row counts reconcile [to filter/drop audit counts]": for
    operations that must conserve row count exactly (cast, derive,
    export), an independent `check_row_count_conservation` between the
    step's real input and output artifacts.
  - "regional totals sum to global total": for an `aggregate` step with a
    non-empty `group_by` and a `sum`-type metric, the metric's per-group
    subtotals (read from the step's own output) must sum to an
    independently recomputed ungrouped total (read from the step's
    input).
Section 20 also lists external control-total anchors ("sum of detail
equals verified control total", ledger opening/closing balances,
source-vs-target transfer counts) - `RequirementContract` has no field
for an external control total today, so a `sum` metric with no group_by
anchor is reported UNAVAILABLE rather than silently skipped, exactly per
Section 20's own instruction.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from dataos.contracts.requirement_contract import RequirementContract
from dataos.validation.checks.conservation import check_row_count_conservation
from dataos.validation.checks.reconciliation import check_reconciliation
from dataos.workflow.dsl import Workflow, WorkflowStep

_ROW_CONSERVING_OPERATIONS = frozenset({"cast", "derive", "export"})
_DEFAULT_TOLERANCE = 0.01

Status = Literal["PASS", "FAIL", "UNAVAILABLE"]


class ReconciliationTest(BaseModel):
    name: str
    expected: float | int | str | None = None
    observed: float | int | str | None = None
    difference: float | None = None
    tolerance: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    passed: bool = True


class ReconciliationReport(BaseModel):
    status: Status
    tests: list[ReconciliationTest] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


class ReconciliationAgent:
    def reconcile(
        self,
        *,
        contract: RequirementContract,
        workflow: Workflow,
        artifacts: dict[str, pl.DataFrame],
    ) -> ReconciliationReport:
        tests: list[ReconciliationTest] = []
        blocking_reasons: list[str] = []

        for step in workflow.steps:
            if step.operation_id not in _ROW_CONSERVING_OPERATIONS:
                continue
            test = self._row_conservation_test(step, artifacts)
            if test is not None:
                tests.append(test)

        covering_step = {ref: step for step in workflow.steps for ref in step.requirement_refs}
        for metric in contract.metrics:
            formula = (metric.formula or "").strip().lower()
            if not formula.startswith("sum("):
                continue
            step = covering_step.get(metric.name)
            if step is None or step.operation_id != "aggregate":
                continue  # not an aggregate output - no subtotal/total relationship exists to check

            group_by = step.params.get("group_by") or []
            if not group_by:
                blocking_reasons.append(
                    f"metric '{metric.name}' has no independent reconciliation anchor available "
                    "(no group_by subtotal to reconcile against, and no external control total is supported)"
                )
                continue

            test = self._group_subtotal_test(metric.name, step, artifacts)
            if test is not None:
                tests.append(test)
            else:
                blocking_reasons.append(
                    f"metric '{metric.name}' could not be independently reconciled - required data unavailable"
                )

        failed = [t for t in tests if not t.passed]
        if failed:
            status: Status = "FAIL"
        elif blocking_reasons:
            status = "UNAVAILABLE"
        else:
            status = "PASS"

        return ReconciliationReport(status=status, tests=tests, blocking_reasons=blocking_reasons)

    def _row_conservation_test(
        self, step: WorkflowStep, artifacts: dict[str, pl.DataFrame]
    ) -> ReconciliationTest | None:
        if len(step.inputs) != 1:
            return None
        before = artifacts.get(step.inputs[0])
        after = artifacts.get(step.id)
        if before is None or after is None:
            return None

        result = check_row_count_conservation(
            before, after, allow_decrease=False, check_id=f"row_count_conservation:{step.id}"
        )
        return ReconciliationTest(
            name=f"row_count_conservation:{step.id}",
            expected=result.expected,
            observed=result.observed,
            evidence_refs=[step.inputs[0], step.id],
            passed=result.passed,
        )

    def _group_subtotal_test(
        self, metric_name: str, step: WorkflowStep, artifacts: dict[str, pl.DataFrame]
    ) -> ReconciliationTest | None:
        if len(step.inputs) != 1:
            return None
        input_df = artifacts.get(step.inputs[0])
        output_df = artifacts.get(step.id)
        if input_df is None or output_df is None:
            return None

        column = _metric_column(step, metric_name)
        if column is None or column not in input_df.columns or metric_name not in output_df.columns:
            return None

        ungrouped_total = float(input_df[column].sum() or 0.0)
        subtotal_sum = float(output_df[metric_name].sum() or 0.0)
        result = check_reconciliation(
            observed=subtotal_sum,
            expected=ungrouped_total,
            tolerance=_DEFAULT_TOLERANCE,
            check_id=f"regional_total:{metric_name}",
        )
        return ReconciliationTest(
            name=f"group_subtotals_sum_to_total:{metric_name}",
            expected=result.expected,
            observed=result.observed,
            difference=abs(subtotal_sum - ungrouped_total),
            tolerance=_DEFAULT_TOLERANCE,
            evidence_refs=[step.inputs[0], step.id],
            passed=result.passed,
        )


def _metric_column(step: WorkflowStep, metric_name: str) -> str | None:
    metrics_param = step.params.get("metrics")
    if not isinstance(metrics_param, list):
        return None
    for m in metrics_param:
        if isinstance(m, dict) and m.get("name") == metric_name:
            return m.get("column")
    return None
