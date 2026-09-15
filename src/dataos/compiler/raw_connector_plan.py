"""The structured-output schema requested from the LLM by the Connector /
External Write Planner.

Source of truth: AI Prompt Library Section 29 "Connector and External
Write Planner" - "Required output contract". Extends the contract's own
per-action shape with `step_id` (not present in Section 29's generic JSON
example) so a proposed action can be tied back to a real `WorkflowStep` -
the same "model proposes structure, deterministic code assembles the real
object" split used by `RawExtraction` -> `RequirementContract` and
`RawPlan` -> `Workflow`. Security-critical fields (`provider`, `resource`,
`action` READ/WRITE/DELETE, `approval_required`) are deliberately absent
here - `ConnectorWritePlanner` always derives those from the real
`WorkflowStep` and the registry's known connector operations, never from
the model; only the descriptive/narrative fields are ever taken from the
LLM's response, and even those cannot remove a hardcoded safety precheck
or postcheck (see `ConnectorWritePlanner._build_plan`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RawExternalAction(BaseModel):
    step_id: str
    scope: str = ""
    idempotency: str = ""
    rate_limit_retry: str = ""
    transactional: bool = False
    prechecks: list[str] = Field(default_factory=list)
    postchecks: list[str] = Field(default_factory=list)


class RawConnectorPlan(BaseModel):
    actions: list[RawExternalAction] = Field(default_factory=list)
