"""Automation Workflow Builder.

Source of truth: AI Prompt Library Section 24 "Automation Workflow
Builder": "Convert an approved one-time workflow into a recurring
automation without changing its business meaning... Never auto-adapt to
a material schema or semantic change. Drift must stop or quarantine the
run until reviewed."

Deterministic code for everything this codebase can honestly derive: a
`Workflow`/`RequirementContract` that already reached RELEASED once
already carries its own `workflow_version`, `requirement_contract_id`,
and each step's `operation_id@operation_version` - `build()` reads those
directly into `VersionPins` rather than asking anyone (model or caller)
to restate them. `preflight`, `approval_gates`, and `validation_policy`/
`failure_policy` are likewise derived from `Workflow.release_policy`,
`Workflow.approval_gates`, and `PolicyGate.DEFAULT_APPROVAL_REQUIRED_OPERATIONS`
(the same single source of truth `ConnectorWritePlanner` uses).

`trigger`, `notifications`, `retry_policy`, and `rollback_policy` are
always caller-supplied - the same "never guess" discipline
`RequirementCompiler` applies to `sources`: nothing in a `Workflow`
implies a cron schedule, a notification channel, or a compensating
action, so this builder never invents one. When constructed with an
`LLMClient`, `build()` additionally narrates the resulting
`AutomationSpec` into `AutomationSpec.narrative` - never decides any of
its fields, the same split every other retrofitted gate in this package
uses.

Section 24's "Drift must stop or quarantine the run until reviewed" is
enforced separately, by `WorkflowOrchestrator.start_automated_run()`:
every subsequent scheduled firing must present a `Workflow` whose version
pins still match this `AutomationSpec` exactly, and must supply
`baseline_profiles` so `SchemaDriftMonitor` (Section 25) runs - an
automation is the one context in this codebase where drift-checking is
mandatory, not optional.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.policy_gate import DEFAULT_APPROVAL_REQUIRED_OPERATIONS
from dataos.compiler.prompts import AUTOMATION_WORKFLOW_BUILDER_SYSTEM_PROMPT
from dataos.llm.client import LLMClient
from dataos.workflow.dsl import Workflow


class VersionPins(BaseModel):
    workflow_version: int
    requirement_contract_id: str
    operation_versions: dict[str, str]
    """operation_id -> operation_version, one entry per distinct operation
    this workflow's steps use."""


class AutomationSpec(BaseModel):
    automation_id: str
    trigger: dict
    version_pins: VersionPins
    idempotency_key: str
    preflight: list[str] = Field(default_factory=list)
    validation_policy: str
    failure_policy: str
    notifications: list[str] = Field(default_factory=list)
    retry_policy: str
    rollback_policy: str
    approval_gates: list[str] = Field(default_factory=list)
    sla: str | None = None
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; every other field remains the
    decision."""


class AutomationWorkflowBuilder:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def build(
        self,
        *,
        automation_id: str,
        workflow: Workflow,
        trigger: dict,
        notifications: list[str] | None = None,
        retry_policy: str = "no automatic retry - a QUARANTINED run must be reviewed and re-triggered manually",
        rollback_policy: str = "no compensating action defined for this automation's write-capable steps",
        sla: str | None = None,
    ) -> AutomationSpec:
        operation_versions = {step.operation_id: step.operation_version for step in workflow.steps}
        version_pins = VersionPins(
            workflow_version=workflow.workflow_version,
            requirement_contract_id=workflow.requirement_contract_id,
            operation_versions=operation_versions,
        )

        approval_gates = sorted(
            set(workflow.approval_gates)
            | {step.operation_id for step in workflow.steps if step.operation_id in DEFAULT_APPROVAL_REQUIRED_OPERATIONS}
        )

        spec = AutomationSpec(
            automation_id=automation_id,
            trigger=trigger,
            version_pins=version_pins,
            idempotency_key=_idempotency_key(automation_id, version_pins),
            preflight=[
                "schema drift check passes against the approved baseline (SchemaDriftMonitor, Section 25)",
                "workflow/operation version pins match this AutomationSpec exactly",
                "policy gate approvals are present for every write-capable step (PolicyGate)",
            ],
            validation_policy=(
                f"on_failure={workflow.release_policy.on_failure}; "
                f"require_all_acceptance_tests={workflow.release_policy.require_all_acceptance_tests}"
            ),
            failure_policy=(
                "QUARANTINE - never auto-adapt to a material schema or semantic change (Section 24); "
                "a version-pin mismatch or BREAKING/REVIEW_REQUIRED drift stops the run before it starts"
            ),
            notifications=list(notifications or []),
            retry_policy=retry_policy,
            rollback_policy=rollback_policy,
            approval_gates=approval_gates,
            sla=sla,
        )

        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=AUTOMATION_WORKFLOW_BUILDER_SYSTEM_PROMPT,
                items=[{"ref": "summary", "facts": spec.model_dump(exclude={"narrative"})}],
            )
            spec = spec.model_copy(update={"narrative": narratives.get("summary")})

        return spec


def _idempotency_key(automation_id: str, version_pins: VersionPins) -> str:
    payload = json.dumps({"automation_id": automation_id, **version_pins.model_dump()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
