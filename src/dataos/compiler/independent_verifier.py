"""Independent Result Verifier.

Source of truth: AI Prompt Library Section 21 "Independent Result
Verifier": "You did not create the primary plan. Independently decide
whether the produced outputs satisfy the approved Requirement Contract...
Act as an adversarial reviewer. Search for reasons the result could be
wrong. Do not reward plausibility." and its mandatory stop conditions:
"Any blocking validation failed. Any material requirement lacks evidence.
Result relies on an unapproved assumption."

Deterministic code, not an LLM call, for the same reason every other gate
in this package is (`AmbiguityGate`, `PolicyGate`): Section 8's Agent
Boundaries table itself specifies the Verifier Agent is "Read-only;
independent prompt/context; cannot modify output" - nothing here trusts a
model's account of what happened. Every check re-derives its answer from
the `StepRunRecord`s the orchestrator itself already wrote, and from the
`RequirementContract`/`Workflow` the earlier gates already validated -
never from a fresh model call that could hallucinate success.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.contracts.requirement_contract import RequirementContract
from dataos.workflow.dsl import Workflow
from dataos.workflow.store import RunRecord, StepRunRecord

Verdict = Literal["PASS", "FAIL", "NEEDS_REVIEW"]


class RequirementCoverage(BaseModel):
    requirement_ref: str
    status: Literal["PASS", "FAIL"]
    evidence_refs: list[str] = Field(default_factory=list)


class VerifierResult(BaseModel):
    verdict: Verdict
    requirement_coverage: list[RequirementCoverage] = Field(default_factory=list)
    defects: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    release_recommendation: Literal["RELEASE", "QUARANTINE"]


class IndependentVerifier:
    def verify(
        self,
        *,
        contract: RequirementContract,
        workflow: Workflow,
        run: RunRecord,
        step_runs: list[StepRunRecord],
    ) -> VerifierResult:
        defects: list[str] = []
        unverified_claims: list[str] = []

        steps_by_id = {s.step_id: s for s in step_runs}

        # Blueprint 1.2 "No hidden mutations" / Phase 3 exit criterion
        # "source versions pinned": every declared source must actually
        # have a locked snapshot id on this run, not just a name.
        for source in workflow.sources:
            if source not in run.source_snapshot_ids:
                defects.append(f"source '{source}' has no locked snapshot id on this run")

        # C/D. Execution evidence and blocking validation: every declared
        # step must have actually COMPLETED with real evidence attached. An
        # operation that hit a blocking condition raises and the step is
        # recorded FAILED, never COMPLETED - so re-deriving "did every step
        # complete with evidence" from the records is exactly "did every
        # blocking check pass", without trusting the run's own state label.
        for step in workflow.steps:
            record = steps_by_id.get(step.id)
            if record is None:
                defects.append(f"step '{step.id}' has no execution record")
            elif record.status != "COMPLETED":
                defects.append(f"step '{step.id}' did not complete (status={record.status})")
            elif not record.evidence:
                defects.append(f"step '{step.id}' completed with no evidence")

        # A. Requirement coverage: every declared metric must be traceable
        # to at least one completed, evidenced step via requirement_refs -
        # a metric the plan never actually computed is exactly the
        # "material requirement lacks evidence" stop condition.
        requirement_coverage: list[RequirementCoverage] = []
        for metric in contract.metrics:
            covering_steps = [
                s.id
                for s in workflow.steps
                if metric.name in s.requirement_refs
                and steps_by_id.get(s.id) is not None
                and steps_by_id[s.id].status == "COMPLETED"
            ]
            status: Literal["PASS", "FAIL"] = "PASS" if covering_steps else "FAIL"
            if not covering_steps:
                defects.append(f"requirement '{metric.name}' has no completed step covering it")
            requirement_coverage.append(
                RequirementCoverage(requirement_ref=metric.name, status=status, evidence_refs=covering_steps)
            )

        # H. Output format / acceptance tests: an acceptance test declared
        # on the contract must not have been silently dropped by planning.
        for test in contract.acceptance_tests:
            if test not in workflow.final_acceptance_tests:
                defects.append(f"acceptance test dropped during planning: '{test}'")

        # E. Reconciliation: any reconciliation evidence a step reported
        # must show as passed. Defense in depth - an operation whose own
        # reconciliation check failed already raises and never reaches
        # COMPLETED, so this branch should be unreachable in practice; an
        # adversarial reviewer checks it anyway rather than assume that.
        has_reconciliation_evidence = False
        for record in step_runs:
            for entry in _reconciliation_entries(record.evidence):
                has_reconciliation_evidence = True
                if entry.get("passed") is False:
                    defects.append(f"step '{record.step_id}' reports a failed reconciliation: {entry}")

        sum_metrics = [m for m in contract.metrics if (m.formula or "").strip().lower().startswith("sum(")]
        if sum_metrics and not has_reconciliation_evidence:
            unverified_claims.append(
                "contract declares sum-type metric(s) but no step reported independent reconciliation evidence"
            )

        # F. No unauthorized side effects / assumptions: re-derive that no
        # prohibited operation ran and no clarification was left open,
        # rather than trust the Policy Gate's earlier decision or the
        # contract's own status field.
        used_operations = {s.operation_id for s in workflow.steps}
        for op in sorted(a for a in contract.prohibited_actions if a in used_operations):
            defects.append(f"workflow used prohibited operation '{op}'")
        if contract.clarifications:
            defects.append("requirement contract has unresolved clarifications")

        if defects:
            verdict: Verdict = "FAIL"
            release_recommendation: Literal["RELEASE", "QUARANTINE"] = "QUARANTINE"
        elif unverified_claims:
            verdict = "NEEDS_REVIEW"
            release_recommendation = "QUARANTINE"
        else:
            verdict = "PASS"
            release_recommendation = "RELEASE"

        return VerifierResult(
            verdict=verdict,
            requirement_coverage=requirement_coverage,
            defects=defects,
            unverified_claims=unverified_claims,
            release_recommendation=release_recommendation,
        )


def _reconciliation_entries(evidence: dict) -> list[dict]:
    entries: list[dict] = []
    reconciliation = evidence.get("reconciliation")
    if isinstance(reconciliation, list):
        entries.extend(e for e in reconciliation if isinstance(e, dict))
    external = evidence.get("external_reconciliation")
    if isinstance(external, dict):
        entries.append(external)
    return entries
