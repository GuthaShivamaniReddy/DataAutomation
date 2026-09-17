"""The structured-output schema requested from the LLM by the Workflow
Planner.

Source of truth: AI Prompt Library Section 9 "Workflow Planner" - "Required
output contract". Deliberately narrower than `dataos.workflow.dsl.Workflow`:
the model's job is to propose steps, not to construct the final typed
`Workflow` itself - `requirement_contract_id`, `workflow_version`, and
`sources` are caller-supplied context the model never invents (same
"deterministic code assembles the real object" split used by
`RawExtraction` -> `RequirementContract`). `type` is kept only as
informational evidence; `WorkflowStep` has no such field since a step's
real behavior is fully determined by its `operation_id`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StepType = Literal[
    "PROFILE", "FILTER", "CAST", "DEDUPLICATE", "JOIN", "DERIVE", "AGGREGATE", "PIVOT", "UNPIVOT",
    "CLEAN_TEXT", "VALIDATE", "MODEL", "EXPORT", "WRITE",
]
"""Section 9's own enum (PROFILE/FILTER/JOIN/DERIVE/AGGREGATE/VALIDATE/
MODEL/EXPORT/WRITE) plus CAST/DEDUPLICATE/PIVOT/UNPIVOT/CLEAN_TEXT -
additive step types for registered operations Section 9 predates.
`type` stays informational only; a step's real behavior is fully
determined by its `operation_id` (see module docstring)."""


class RawPlanStep(BaseModel):
    id: str
    type: StepType
    operation_id: str
    operation_version: str
    inputs: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)
    on_failure: Literal["STOP", "QUARANTINE"] = "QUARANTINE"
    requirement_refs: list[str] = Field(default_factory=list)


class RawPlan(BaseModel):
    plan_id: str
    steps: list[RawPlanStep] = Field(default_factory=list)
    final_acceptance_tests: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    approval_gates: list[str] = Field(default_factory=list)
