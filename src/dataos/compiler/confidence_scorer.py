"""Evidence and Confidence Scorer.

Source of truth: AI Prompt Library Section 22 "Confidence & Evidence
Scorer": "Score confidence in the process, not subjective confidence in
an answer... Do not average away a critical failure. Any critical
component that fails forces overall status to NOT_RELEASABLE regardless
of numeric score."

Deterministic code, not an LLM call - same reasoning as every other
gate/generator/agent in this package, and doubly so here: a *confidence*
score is exactly the kind of thing a model should never get to assign to
its own work. `overall_status` is never computed independently from the
per-dimension scores below it - it is taken directly from the
`ReleaseDecision` the Release Gate (Section 23) already made, so this
scorer can add detail but can never be more lenient than the hard gates
it reports on. That is Section 22's "preserving hard failure gates"
requirement enforced by construction rather than by a numeric threshold
this module would otherwise have to invent.

Every per-dimension score below is derived only from evidence this
codebase already computes elsewhere (`RequirementContract`,
`ValidationReport`, `ReconciliationReport`, `VerifierResult`,
`StepRunRecord`s) - never a fresh calculation. Section 22 also lists
"model/prediction uncertainty (if applicable)" as a dimension; no
forecast/ML operation exists in this codebase yet, so it is omitted
rather than scored with an invented placeholder.

When constructed with an `LLMClient`, `score()` additionally asks it to
narrate the already-computed scores into `ConfidenceReport.narrative` -
never to decide any score or `overall_status`, both of which are already
fixed by the time the model is ever called.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from dataos.compiler.independent_verifier import VerifierResult
from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import CONFIDENCE_SCORER_SYSTEM_PROMPT
from dataos.compiler.reconciliation_agent import ReconciliationReport
from dataos.compiler.release_gate import ReleaseDecision
from dataos.contracts.requirement_contract import DefinitionStatus, RequirementContract
from dataos.llm.client import LLMClient
from dataos.validation.engine import ValidationReport
from dataos.workflow.dsl import Workflow
from dataos.workflow.store import RunRecord, StepRunRecord

_DEFINITION_STATUS_WEIGHT: dict[DefinitionStatus, float] = {
    DefinitionStatus.GOVERNED: 1.0,
    DefinitionStatus.EXPLICIT: 0.85,
    DefinitionStatus.INFERRED_UNIQUE: 0.7,
    DefinitionStatus.AMBIGUOUS: 0.0,
    DefinitionStatus.MISSING: 0.0,
}

# ValidationRuleGenerator's own check_id namespacing (null_rate:*,
# unique:*, required_column:*) is what separates "source data fitness"
# from the broader "validation coverage" dimension below - both read the
# same ValidationReport, just different subsets of it.
_DATA_QUALITY_CHECK_PREFIXES = ("null_rate:", "unique:", "required_column:")


class ConfidenceScores(BaseModel):
    requirements: float
    semantics: float
    data_quality: float
    execution: float
    validation: float
    reconciliation: float
    verification: float


class ConfidenceReport(BaseModel):
    overall_status: Literal["RELEASABLE", "NOT_RELEASABLE"]
    scores: ConfidenceScores
    critical_failures: list[str]
    evidence_refs: list[str]
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see class
    docstring) - never authoritative; `overall_status` remains the
    decision."""


class ConfidenceScorer:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def score(
        self,
        *,
        contract: RequirementContract,
        run: RunRecord,
        workflow: Workflow,
        step_runs: list[StepRunRecord],
        validation_report: ValidationReport,
        reconciliation_report: ReconciliationReport,
        verifier_result: VerifierResult,
        release_decision: ReleaseDecision,
    ) -> ConfidenceReport:
        scores = ConfidenceScores(
            requirements=self._requirements_score(contract),
            semantics=self._semantics_score(contract),
            data_quality=self._data_quality_score(validation_report),
            execution=self._execution_score(workflow, step_runs),
            validation=self._validation_score(validation_report),
            reconciliation=self._reconciliation_score(reconciliation_report),
            verification=self._verification_score(verifier_result),
        )

        overall_status: Literal["RELEASABLE", "NOT_RELEASABLE"] = (
            "RELEASABLE" if release_decision.decision == "RELEASE" else "NOT_RELEASABLE"
        )

        evidence_refs = [f"run:{run.run_id}"] + [f"step:{s.id}" for s in workflow.steps]
        critical_failures = list(release_decision.reason_codes)

        narrative = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=CONFIDENCE_SCORER_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {
                            "overall_status": overall_status,
                            "scores": scores.model_dump(),
                            "critical_failures": critical_failures,
                        },
                    }
                ],
            )
            narrative = narratives.get("summary")

        return ConfidenceReport(
            overall_status=overall_status,
            scores=scores,
            critical_failures=critical_failures,
            evidence_refs=evidence_refs,
            narrative=narrative,
        )

    def _requirements_score(self, contract: RequirementContract) -> float:
        if contract.clarifications or not contract.sources:
            return 0.0
        if not contract.metrics:
            return 1.0
        return sum(0.0 if m.definition_status.is_blocking else 1.0 for m in contract.metrics) / len(
            contract.metrics
        )

    def _semantics_score(self, contract: RequirementContract) -> float:
        if not contract.metrics:
            return 1.0
        return sum(_DEFINITION_STATUS_WEIGHT[m.definition_status] for m in contract.metrics) / len(
            contract.metrics
        )

    def _data_quality_score(self, validation_report: ValidationReport) -> float:
        results = [r for r in validation_report.results if r.check_id.startswith(_DATA_QUALITY_CHECK_PREFIXES)]
        if not results:
            return 1.0
        return sum(1.0 for r in results if r.passed) / len(results)

    def _validation_score(self, validation_report: ValidationReport) -> float:
        if not validation_report.results:
            return 1.0
        return sum(1.0 for r in validation_report.results if r.passed) / len(validation_report.results)

    def _execution_score(self, workflow: Workflow, step_runs: list[StepRunRecord]) -> float:
        if not workflow.steps:
            return 1.0
        completed = {s.step_id for s in step_runs if s.status == "COMPLETED"}
        return sum(1.0 for s in workflow.steps if s.id in completed) / len(workflow.steps)

    def _reconciliation_score(self, reconciliation_report: ReconciliationReport) -> float:
        total = len(reconciliation_report.tests) + len(reconciliation_report.blocking_reasons)
        if total == 0:
            return 1.0
        passed = sum(1.0 for t in reconciliation_report.tests if t.passed)
        return passed / total

    def _verification_score(self, verifier_result: VerifierResult) -> float:
        if verifier_result.defects:
            return 0.0
        if not verifier_result.requirement_coverage:
            return 1.0
        return sum(1.0 for c in verifier_result.requirement_coverage if c.status == "PASS") / len(
            verifier_result.requirement_coverage
        )
