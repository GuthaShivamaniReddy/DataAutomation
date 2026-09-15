"""Typed workflow DSL.

Source of truth: Reliability-First Master Blueprint Section 9.1 "Typed
workflow example" and AI Prompt Library Section 9 "Workflow Planner"
output contract (per-step `inputs`/`output`, `on_failure: STOP|QUARANTINE`).

Every step references its inputs explicitly by id - either `"source:<name>"`
for a declared workflow source, or another step's `id`. There is no
implicit "previous step" chaining: Blueprint 1.2 rule "No untyped agent
actions" means a plan must be fully explicit before execution, not
inferred from list position.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from dataos.errors import ErrorCode, PlatformError

SOURCE_PREFIX = "source:"


class WorkflowStep(BaseModel):
    id: str
    operation_id: str
    operation_version: str
    inputs: list[str]
    params: dict = Field(default_factory=dict)
    on_failure: Literal["STOP", "QUARANTINE"] = "QUARANTINE"
    requirement_refs: list[str] = Field(default_factory=list)

    @property
    def full_operation_id(self) -> str:
        return f"{self.operation_id}@{self.operation_version}"

    @field_validator("id")
    @classmethod
    def _id_not_source_prefixed(cls, v: str) -> str:
        if v.startswith(SOURCE_PREFIX):
            raise ValueError(f"step id must not start with reserved prefix '{SOURCE_PREFIX}': {v}")
        return v


class ReleasePolicy(BaseModel):
    require_all_acceptance_tests: bool = True
    on_failure: Literal["quarantine", "stop"] = "quarantine"


class Workflow(BaseModel):
    workflow_version: int
    requirement_contract_id: str
    sources: list[str]
    steps: list[WorkflowStep]
    release_policy: ReleasePolicy = Field(default_factory=ReleasePolicy)
    final_acceptance_tests: list[str] = Field(default_factory=list)
    approval_gates: list[str] = Field(default_factory=list)

    def step_by_id(self, step_id: str) -> WorkflowStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no step with id '{step_id}' in this workflow")

    def validate_dag(self) -> None:
        """Duplicate step ids, unknown input references, and cycles are
        all structurally invalid plans - Blueprint 1.2 "No untyped agent
        actions" / "the planner emits a typed plan that is
        schema-validated before execution"."""
        step_ids = [s.id for s in self.steps]
        duplicates = {sid for sid in step_ids if step_ids.count(sid) > 1}
        if duplicates:
            raise PlatformError(
                ErrorCode.VALIDATION_FAIL,
                f"duplicate step id(s) in workflow: {sorted(duplicates)}",
            )

        valid_refs = {f"{SOURCE_PREFIX}{s}" for s in self.sources} | set(step_ids)
        for step in self.steps:
            unknown = [ref for ref in step.inputs if ref not in valid_refs]
            if unknown:
                raise PlatformError(
                    ErrorCode.SCHEMA_MISSING,
                    f"step '{step.id}' references unknown input(s): {unknown}",
                    evidence={"step_id": step.id, "unknown_refs": unknown, "valid_refs": sorted(valid_refs)},
                )

        self._topological_order()  # raises VALIDATION_FAIL on a cycle

    def execution_order(self) -> list[str]:
        """Kahn's-algorithm topological sort of step ids (sources are not
        included - they are pre-seeded, already-COMPLETED nodes)."""
        return self._topological_order()

    def _topological_order(self) -> list[str]:
        step_ids = [s.id for s in self.steps]
        step_id_set = set(step_ids)
        dependencies: dict[str, set[str]] = {
            s.id: {ref for ref in s.inputs if ref in step_id_set} for s in self.steps
        }

        ordered: list[str] = []
        remaining = set(step_ids)
        while remaining:
            ready = sorted(sid for sid in remaining if not (dependencies[sid] & remaining))
            if not ready:
                raise PlatformError(
                    ErrorCode.VALIDATION_FAIL,
                    f"cycle detected among workflow steps: {sorted(remaining)}",
                    evidence={"remaining_steps": sorted(remaining)},
                )
            ordered.extend(ready)
            remaining -= set(ready)
        return ordered
