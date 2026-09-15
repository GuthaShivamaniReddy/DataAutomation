"""Connector / External Write Planner.

Source of truth: AI Prompt Library Section 29 "Connector and External
Write Planner": "Plan reads/writes to external systems using least
privilege... Default to read-only. Any write must reference an approved
workflow step and authorization token. Any delete or irreversible action
requires explicit high-risk approval and, where possible, a recoverable
staging step."

This calls a real LLM for the same reason `WorkflowPlanner` and
`RequirementCompiler` do - narrating idempotency behavior, rate-limit
posture, and rollback support in useful, step-specific prose is a
genuine interpretation task, not a mechanical lookup. But every
security-critical field is still deterministic, never trusted to the
model, the same "model proposes structure, deterministic code assembles
the real object" split `WorkflowPlanner` applies to operations:
  - `provider`, `resource`, and `action` (READ/WRITE/DELETE) are always
    read from the real `WorkflowStep` and the registry's known connector
    operations (`_CONNECTOR_SPECS`) - never from the model's response.
  - `approval_required`/`approval_token` are always derived from
    `PolicyGate.DEFAULT_APPROVAL_REQUIRED_OPERATIONS`, the single source
    of truth `PolicyGate` itself enforces at `start_run` - this planner
    can describe an action, but can never be the thing that decides
    whether it is allowed to run.
  - The model may enrich `prechecks`/`postchecks` with step-specific
    detail, but can never cause a hardcoded safety precondition to
    disappear from the plan - the two lists are always unioned, never
    replaced.
  - `verify()` is plain code, not a second LLM call: independently
    re-deriving a SHA-256 file hash is a deterministic computation an
    LLM cannot improve on and must not be trusted to merely assert.

The only write-capable primitive in this codebase today is `export`
(`registry/operations/export.py`), which writes to a local filesystem
path - there is no database, API, or SaaS connector implemented yet, so
`_CONNECTOR_SPECS` covers exactly that one action honestly rather than a
speculative multi-provider framework. `ExternalActionPlan` is
provider-agnostic on purpose so a real connector added later slots into
the same shape; a step whose `operation_id` is not in `_CONNECTOR_SPECS`
is never even sent to the model - "default to read-only" in the
strongest sense: no recognized external action, no plan entry, nothing
asked of the LLM about it.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.constitution import GLOBAL_CONSTITUTION
from dataos.compiler.policy_gate import DEFAULT_APPROVAL_REQUIRED_OPERATIONS
from dataos.compiler.prompts import CONNECTOR_WRITE_PLANNER_SYSTEM_PROMPT
from dataos.compiler.raw_connector_plan import RawConnectorPlan, RawExternalAction
from dataos.ingestion.hashing import sha256_file
from dataos.llm.client import LLMClient
from dataos.workflow.dsl import Workflow, WorkflowStep
from dataos.workflow.store import StepRunRecord

ActionType = Literal["READ", "WRITE", "DELETE"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class ConnectorSpec(BaseModel):
    """The fixed, honest description of what a given write-capable
    operation does - not per-step data, just the operation's own
    characteristics, and the safety-critical floor the model's own
    prechecks/postchecks can only add to."""

    provider: str
    action: ActionType
    idempotency: str
    rate_limit_retry: str
    transactional: bool
    prechecks: list[str]
    postchecks: list[str]
    risk_level: RiskLevel


_CONNECTOR_SPECS: dict[str, ConnectorSpec] = {
    "export": ConnectorSpec(
        provider="local_filesystem",
        action="WRITE",
        idempotency=(
            "overwrites the destination path deterministically - re-running the same step with the "
            "same input reproduces the same file, never a duplicate"
        ),
        rate_limit_retry="not applicable - a local filesystem write has no external rate limit",
        transactional=False,
        prechecks=[
            "format is an approved type (csv/parquet/json)",
            "destination path has no '..' traversal segment (ExportOperation precondition)",
            "destination path resolves to a local file, not an external URL (Security Guard)",
        ],
        postchecks=[
            "destination file exists",
            "recorded checksum matches an independently recomputed SHA-256 of the written file",
            "recorded row_count and schema_manifest are present",
        ],
        risk_level="HIGH",
    ),
}


class ExternalActionPlan(BaseModel):
    step_id: str
    provider: str
    resource: str
    action: ActionType
    scope: str
    idempotency: str
    rate_limit_retry: str
    transactional: bool
    prechecks: list[str] = Field(default_factory=list)
    postchecks: list[str] = Field(default_factory=list)
    approval_required: bool
    approval_token: str | None = None
    risk_level: RiskLevel


class ExternalActionVerification(BaseModel):
    step_id: str
    verified: bool
    checks_passed: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class ConnectorWritePlanner:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def plan(self, workflow: Workflow) -> list[ExternalActionPlan]:
        candidates = [(step, _CONNECTOR_SPECS[step.operation_id]) for step in workflow.steps if step.operation_id in _CONNECTOR_SPECS]
        if not candidates:
            return []

        context = {
            "steps": [
                {
                    "step_id": step.id,
                    "operation_id": step.operation_id,
                    "provider": spec.provider,
                    "action": spec.action,
                    "resource": _resource_for(step, spec),
                }
                for step, spec in candidates
            ]
        }

        raw_plan = self._llm_client.complete_structured(
            system_prompt=f"{GLOBAL_CONSTITUTION}\n\n{CONNECTOR_WRITE_PLANNER_SYSTEM_PROMPT}",
            user_prompt=json.dumps(context),
            response_model=RawConnectorPlan,
        )
        raw_by_step = {a.step_id: a for a in raw_plan.actions}

        return [self._build_plan(step, spec, raw_by_step.get(step.id)) for step, spec in candidates]

    def _build_plan(
        self, step: WorkflowStep, spec: ConnectorSpec, raw_action: RawExternalAction | None
    ) -> ExternalActionPlan:
        resource = _resource_for(step, spec)
        approval_required = step.operation_id in DEFAULT_APPROVAL_REQUIRED_OPERATIONS

        # The model may enrich these with step-specific detail; a hardcoded
        # safety precondition from `spec` can never disappear from the plan.
        prechecks = sorted(set(spec.prechecks) | set(raw_action.prechecks if raw_action else []))
        postchecks = sorted(set(spec.postchecks) | set(raw_action.postchecks if raw_action else []))

        return ExternalActionPlan(
            step_id=step.id,
            provider=spec.provider,
            resource=resource,
            action=spec.action,
            scope=(raw_action.scope if raw_action and raw_action.scope else f"filesystem write access to '{resource}' and its parent directory only"),
            idempotency=(raw_action.idempotency if raw_action and raw_action.idempotency else spec.idempotency),
            rate_limit_retry=(raw_action.rate_limit_retry if raw_action and raw_action.rate_limit_retry else spec.rate_limit_retry),
            transactional=(raw_action.transactional if raw_action else spec.transactional),
            prechecks=prechecks,
            postchecks=postchecks,
            approval_required=approval_required,
            approval_token=step.operation_id if approval_required else None,
            risk_level=spec.risk_level,
        )

    def verify(
        self, plans: list[ExternalActionPlan], step_runs: dict[str, StepRunRecord]
    ) -> list[ExternalActionVerification]:
        """Independent post-write verification (Section 29's own
        "post-write verification" field): plain code, not a second LLM
        call - re-derives its answer from the actual file on disk, never
        from the step's self-reported `evidence` dict alone, and never
        from a model's assertion that a hash "matches"."""
        return [self._verify_one(plan, step_runs.get(plan.step_id)) for plan in plans]

    def _verify_one(self, plan: ExternalActionPlan, record: StepRunRecord | None) -> ExternalActionVerification:
        checks_passed: list[str] = []
        issues: list[str] = []

        if record is None or record.status != "COMPLETED":
            issues.append(f"step '{plan.step_id}' did not complete - no write to verify")
            return ExternalActionVerification(step_id=plan.step_id, verified=False, issues=issues)

        evidence = record.evidence
        recorded_checksum = evidence.get("checksum")
        recorded_path = evidence.get("destination_path", plan.resource)

        if not recorded_checksum:
            issues.append("no checksum recorded in step evidence")
        if "row_count" not in evidence:
            issues.append("no row_count recorded in step evidence")
        if "schema_manifest" not in evidence:
            issues.append("no schema_manifest recorded in step evidence")

        if recorded_checksum:
            try:
                actual_checksum = sha256_file(recorded_path)
            except OSError:
                issues.append(f"destination file '{recorded_path}' does not exist on disk")
            else:
                if actual_checksum == recorded_checksum:
                    checks_passed.append("recorded checksum matches an independently recomputed SHA-256")
                else:
                    issues.append(
                        f"recorded checksum '{recorded_checksum}' does not match the independently "
                        f"recomputed checksum '{actual_checksum}' of the file currently on disk"
                    )

        if not issues:
            checks_passed.extend(["row_count recorded", "schema_manifest recorded"])

        return ExternalActionVerification(step_id=plan.step_id, verified=not issues, checks_passed=checks_passed, issues=issues)


def _resource_for(step: WorkflowStep, spec: ConnectorSpec) -> str:
    if spec.provider == "local_filesystem":
        return str(step.params.get("destination_path", "<unresolved>"))
    return "<unresolved>"
