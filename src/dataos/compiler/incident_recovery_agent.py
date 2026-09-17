"""Runtime Incident and Recovery Agent.

Source of truth: AI Prompt Library Section 26 "Runtime Incident and
Recovery Agent": "Given execution logs, validation failures, tool
errors, and workflow state, propose only safe recovery actions...
Preserve the failed run and evidence. Never skip a failed blocking check
merely to complete the workflow. Distinguish transient infrastructure
errors from deterministic data/logic failures. Retry only idempotent
steps or steps with an approved compensating transaction. Never duplicate
external writes... Escalate repeated or unexplained deterministic
failures to engineering review."

Deterministic code, not an LLM call - same reasoning as every other gate
in this package: the failure class and the safe action are fully
determined by the `RunRecord`/`StepRunRecord` evidence already durably
recorded by `RunStore` (Blueprint Section 16), never a judgment call.

"Transient infrastructure errors" is not one of this codebase's
`ErrorCode` values (Appendix C's taxonomy is entirely deterministic
data/logic/policy failures) - a real infra error/crash never gets wrapped
into a `StepRunRecord.error` at all, because `orchestrator.run()` only
catches `PlatformError`. Its actual, correctly-recognized signature is a
step stuck at status "RUNNING" with the run itself not in a terminal
state: exactly the crash-safe-resume case `orchestrator.py`'s own
docstring describes, so the one safe action for it really is RETRY -
calling `run()` again, which by construction only re-executes unfinished
steps.

Every other `ErrorCode` is a deterministic data/logic/policy defect.
Section 26's own rule ("never skip a failed blocking check merely to
complete the workflow") means none of these ever gets RETRY - the
default is QUARANTINE (already the state `run()` left it in) until a
human fixes the underlying contract/plan/policy, except:
  - `EXTERNAL_WRITE_UNVERIFIED` - a possible partial write must never be
    retried blindly (Section 26: "never duplicate external writes");
    the safe action is STOP pending reconciliation through
    `ConnectorWritePlanner.verify()` (Section 29), or ROLLBACK only when
    the caller passes a real, non-default `rollback_policy` recorded on
    an `AutomationSpec` (Section 24) - never invented here.
  - `POLICY_DENIED` - not an engineering defect at all; the safe action
    is STOP pending the missing approval.

`prior_failure_count` is caller-supplied because this codebase's
`RunStore` keeps only the latest attempt per step, not attempt history -
inventing a count this class cannot actually observe would violate the
same "never guess" discipline as everything else here. Above a fixed
threshold it always escalates, regardless of failure class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import INCIDENT_RECOVERY_AGENT_SYSTEM_PROMPT
from dataos.llm.client import LLMClient
from dataos.workflow.store import RunRecord, StepRunRecord

FailureClass = Literal["TRANSIENT", "DATA", "LOGIC", "SECURITY", "EXTERNAL_SIDE_EFFECT", "UNKNOWN"]
SafeAction = Literal["RETRY", "ROLLBACK", "QUARANTINE", "STOP", "ESCALATE"]

_ESCALATION_THRESHOLD = 3

_DEFAULT_ROLLBACK_POLICY_PLACEHOLDER = "no compensating action defined for this automation's write-capable steps"

_FAILURE_CLASS_BY_CODE: dict[str, FailureClass] = {
    "REQ_AMBIGUOUS": "LOGIC",
    "REQ_MISSING": "LOGIC",
    "SEMANTIC_UNGOVERNED": "LOGIC",
    "SCHEMA_MISSING": "DATA",
    "SCHEMA_DRIFT": "DATA",
    "JOIN_UNSAFE": "DATA",
    "DATA_QUALITY_BLOCK": "DATA",
    "VALIDATION_FAIL": "DATA",
    "RECONCILIATION_FAIL": "DATA",
    "VERIFICATION_FAIL": "LOGIC",
    "POLICY_DENIED": "SECURITY",
    "EXTERNAL_WRITE_UNVERIFIED": "EXTERNAL_SIDE_EFFECT",
    "MODEL_OUTPUT_INVALID": "LOGIC",
    "NO_SAFE_OPERATION": "LOGIC",
    "PREDICTION_UNSUPPORTED": "DATA",
    "EVIDENCE_INSUFFICIENT": "DATA",
}


class IncidentRecoveryResult(BaseModel):
    failure_class: FailureClass
    safe_action: SafeAction
    retry_from_step: str | None = None
    conditions: list[str] = Field(default_factory=list)
    evidence_to_preserve: list[str] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `failure_class`/`safe_action`
    remain the decision."""


class IncidentRecoveryAgent:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def diagnose(
        self,
        *,
        run: RunRecord,
        step_runs: list[StepRunRecord],
        prior_failure_count: int = 0,
        rollback_policy: str | None = None,
    ) -> IncidentRecoveryResult:
        result = self._diagnose(
            run=run, step_runs=step_runs, prior_failure_count=prior_failure_count, rollback_policy=rollback_policy
        )

        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=INCIDENT_RECOVERY_AGENT_SYSTEM_PROMPT,
                items=[{"ref": "summary", "facts": result.model_dump(exclude={"narrative"})}],
            )
            result = result.model_copy(update={"narrative": narratives.get("summary")})

        return result

    def _diagnose(
        self,
        *,
        run: RunRecord,
        step_runs: list[StepRunRecord],
        prior_failure_count: int = 0,
        rollback_policy: str | None = None,
    ) -> IncidentRecoveryResult:
        preserved = [f"RunRecord for '{run.run_id}' (state={run.state.value})"]

        stalled_step = next((s for s in step_runs if s.status == "RUNNING"), None)
        failed_step = next((s for s in step_runs if s.status == "FAILED"), None)

        if failed_step is not None:
            preserved.append(f"StepRunRecord for '{failed_step.step_id}' (error={failed_step.error})")
            result = self._diagnose_failed_step(failed_step, rollback_policy)
        elif stalled_step is not None:
            preserved.append(f"StepRunRecord for '{stalled_step.step_id}' (stalled at RUNNING)")
            result = IncidentRecoveryResult(
                failure_class="TRANSIENT",
                safe_action="RETRY",
                retry_from_step=stalled_step.step_id,
                conditions=[
                    "run() is crash-safe and will only re-execute this step; every already-COMPLETED step is skipped"
                ],
            )
        else:
            return IncidentRecoveryResult(
                failure_class="UNKNOWN",
                safe_action="STOP",
                conditions=["no failed or stalled step evidence was found for this run - nothing to recover"],
                evidence_to_preserve=preserved,
            )

        if prior_failure_count >= _ESCALATION_THRESHOLD:
            result = result.model_copy(
                update={
                    "safe_action": "ESCALATE",
                    "retry_from_step": None,
                    "conditions": [
                        f"this step has already failed {prior_failure_count} time(s) - escalate to engineering "
                        "review rather than retrying again",
                        *result.conditions,
                    ],
                }
            )

        return result.model_copy(update={"evidence_to_preserve": preserved + result.evidence_to_preserve})

    def _diagnose_failed_step(self, failed_step: StepRunRecord, rollback_policy: str | None) -> IncidentRecoveryResult:
        error = failed_step.error or {}
        code = error.get("code")
        failure_class = _FAILURE_CLASS_BY_CODE.get(code, "UNKNOWN") if code else "UNKNOWN"

        if failure_class == "UNKNOWN":
            return IncidentRecoveryResult(
                failure_class="UNKNOWN",
                safe_action="ESCALATE",
                conditions=["failure could not be classified from recorded evidence - escalate to engineering review"],
            )

        if failure_class == "SECURITY":
            return IncidentRecoveryResult(
                failure_class=failure_class,
                safe_action="STOP",
                conditions=["obtain the required approval token(s) before re-attempting - this is a policy decision, not an engineering defect"],
            )

        if failure_class == "EXTERNAL_SIDE_EFFECT":
            has_approved_rollback = rollback_policy is not None and rollback_policy != _DEFAULT_ROLLBACK_POLICY_PLACEHOLDER
            if has_approved_rollback:
                return IncidentRecoveryResult(
                    failure_class=failure_class,
                    safe_action="ROLLBACK",
                    conditions=[f"execute the approved compensating transaction: {rollback_policy}"],
                )
            return IncidentRecoveryResult(
                failure_class=failure_class,
                safe_action="STOP",
                conditions=[
                    "reconcile via the Connector/External Write Planner's post-write verification before any "
                    "retry is considered - never retry blindly, to avoid a duplicate external write"
                ],
                evidence_to_preserve=["provider transaction id/idempotency key in the step's evidence, if any"],
            )

        # DATA / LOGIC
        return IncidentRecoveryResult(
            failure_class=failure_class,
            safe_action="QUARANTINE",
            conditions=[
                f"resolve the underlying {failure_class.lower()} issue ({code}) before re-attempting this "
                "workflow - do not retry unchanged"
            ],
        )
