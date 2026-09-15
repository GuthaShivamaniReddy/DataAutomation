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
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.independent_verifier import VerifierResult
from dataos.contracts.requirement_contract import RequirementContract, RequirementStatus
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunRecord


class ReleaseDecision(BaseModel):
    decision: Literal["RELEASE", "QUARANTINE"]
    reason_codes: list[str] = Field(default_factory=list)
    required_remediation: list[str] = Field(default_factory=list)


class ReleaseGate:
    def decide(
        self,
        *,
        contract: RequirementContract,
        run: RunRecord,
        verifier_result: VerifierResult,
    ) -> ReleaseDecision:
        reason_codes: list[str] = []

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

        # Never downgrade a blocking failure to a warning: any reason code
        # forces QUARANTINE outright, regardless of how many others agree.
        decision: Literal["RELEASE", "QUARANTINE"] = "QUARANTINE" if reason_codes else "RELEASE"
        required_remediation = list(verifier_result.defects) if decision == "QUARANTINE" else []

        return ReleaseDecision(
            decision=decision,
            reason_codes=reason_codes,
            required_remediation=required_remediation,
        )
