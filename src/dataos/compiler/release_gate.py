"""Release / Quarantine Gate.

Source of truth: AI Prompt Library Section 23 "Final Release / Quarantine
Gate": "You are the only component authorized to mark an analytical
result as released... Otherwise QUARANTINE. Return exact reason codes and
remediation steps. Never downgrade a blocking failure to a warning."

`dataos.workflow.state_machine.RunState` (Blueprint 9.2) only defines
RELEASED and QUARANTINED as `VERIFYING`'s outgoing edges - there is no
separate "BLOCK" state to transition into, so this gate's `decision` is
narrower than Section 23's own `RELEASE|QUARANTINE|BLOCK` output contract
on purpose: in this codebase, BLOCK and QUARANTINE are the same terminal
state.

Deterministic code, not an LLM call, for the same reason as every other
gate in this package: finality is a policy decision, never a model's
confidence (Global Constitution "Finality Rule": "You may label a result
FINAL only when the release policy explicitly permits you to do so").

When constructed with an `LLMClient`, `decide()` additionally asks it to
narrate the already-made decision into `ReleaseDecision.narrative` - never
to decide `decision`, `reason_codes`, or `required_remediation`, all of
which are already fixed by the time the model is ever called.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.independent_verifier import VerifierResult
from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import RELEASE_GATE_SYSTEM_PROMPT
from dataos.compiler.reconciliation_agent import ReconciliationReport
from dataos.contracts.requirement_contract import RequirementContract, RequirementStatus
from dataos.llm.client import LLMClient
from dataos.validation.engine import ValidationReport
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunRecord


class ReleaseDecision(BaseModel):
    decision: Literal["RELEASE", "QUARANTINE"]
    reason_codes: list[str] = Field(default_factory=list)
    required_remediation: list[str] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see class
    docstring) - never authoritative; `decision` remains the decision."""


class ReleaseGate:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def decide(
        self,
        *,
        contract: RequirementContract,
        run: RunRecord,
        verifier_result: VerifierResult,
        validation_report: ValidationReport | None = None,
        reconciliation_report: ReconciliationReport | None = None,
    ) -> ReleaseDecision:
        reason_codes: list[str] = []
        required_remediation: list[str] = []

        # Section 23's own condition list, in order:
        if run.state != RunState.VERIFYING:
            reason_codes.append(f"run is in state '{run.state.value}', not VERIFYING")
        if contract.status != RequirementStatus.APPROVED:
            reason_codes.append(f"requirement contract status is '{contract.status.value}', not APPROVED")
        if not contract.ready_for_planning:
            reason_codes.append("requirement contract is not ready_for_planning")
        if verifier_result.verdict != "PASS":
            reason_codes.append(f"independent verifier verdict is '{verifier_result.verdict}', not PASS")
        if verifier_result.release_recommendation != "RELEASE":
            reason_codes.append("independent verifier does not recommend RELEASE")
        if validation_report is not None and not validation_report.passed:
            reason_codes.append("one or more blocking validation checks failed")
            required_remediation.extend(f.check_id for f in validation_report.blocking_failures)
        if reconciliation_report is not None and reconciliation_report.status != "PASS":
            # Section 20: an UNAVAILABLE reconciliation "requires the
            # alternative verification policy" - this codebase has no such
            # policy to satisfy yet, so - per the Constitution's "no
            # destructive default" bias - UNAVAILABLE blocks release the
            # same as FAIL, never silently treated as good enough.
            reason_codes.append(f"reconciliation status is '{reconciliation_report.status}', not PASS")
            required_remediation.extend(t.name for t in reconciliation_report.tests if not t.passed)
            required_remediation.extend(reconciliation_report.blocking_reasons)

        # Never downgrade a blocking failure to a warning: any reason code
        # forces QUARANTINE outright, regardless of how many others agree.
        decision: Literal["RELEASE", "QUARANTINE"] = "QUARANTINE" if reason_codes else "RELEASE"
        if decision == "QUARANTINE":
            required_remediation = list(verifier_result.defects) + required_remediation
        else:
            required_remediation = []

        narrative = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=RELEASE_GATE_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {
                            "decision": decision,
                            "reason_codes": reason_codes,
                            "required_remediation": required_remediation,
                        },
                    }
                ],
            )
            narrative = narratives.get("summary")

        return ReleaseDecision(
            decision=decision,
            reason_codes=reason_codes,
            required_remediation=required_remediation,
            narrative=narrative,
        )
