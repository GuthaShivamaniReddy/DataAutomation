"""Operation Registry Selector.

Source of truth: AI Prompt Library Section 10 "Operation Registry
Selector": "Choose the safest approved deterministic primitive for each
workflow step. You are not allowed to invent an unregistered operation...
If no registered operation satisfies the step, return NO_SAFE_OPERATION.
Do not fall back to arbitrary code without explicit engineering policy
that allows a sandboxed extension path."

Deterministic code, not an LLM call - which registered operation a step
type maps to, and whether that operation is actually present in a given
`OperationRegistry`, are both provable facts, never a judgment call.

This is deliberately a separate, standalone component from
`WorkflowPlanner`, not a duplicate of it: `WorkflowPlanner` already
enforces the same "no unregistered operation" rule (see its own
docstring), but only for a plan the LLM has *already* filled in with a
concrete `operation_id`/`operation_version`, and it never surfaces
*why* an operation was safe to pick or what parameters it actually
requires. This selector works one level earlier, from the abstract step
`type` Section 9's own output contract already defines
(`PROFILE|FILTER|JOIN|DERIVE|AGGREGATE|VALIDATE|MODEL|EXPORT|WRITE`), and
answers Section 10's required contract directly: a `reason` grounded in
the registry's own tested/versioned contract (Blueprint Section 7), the
`required_params` an operation's typed `Params` model actually declares,
and a `certification` string - or `NO_SAFE_OPERATION` when nothing in the
registry can honestly satisfy that step type. `PROFILE` (a plain
function, not a registry `Operation`), `MODEL` (no forecasting primitive
exists yet - see the ML Suitability Gate, Section 16), and `WRITE` (goes
through the Connector/External Write Planner, Section 29, never the data
`OperationRegistry`) always resolve to `NO_SAFE_OPERATION` here, honestly
reflecting that this registry has no primitive for them today rather than
guessing one.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import OPERATION_REGISTRY_SELECTOR_SYSTEM_PROMPT
from dataos.errors import PlatformError
from dataos.llm.client import LLMClient
from dataos.registry.registry import OperationRegistry

# Section 9's step `type` enum -> the one registered operation_id this
# codebase's registry uses to satisfy it today. A step type absent from
# this mapping (PROFILE, MODEL, WRITE - see module docstring) always
# resolves to NO_SAFE_OPERATION, not a guess.
_STEP_TYPE_TO_OPERATION_ID: dict[str, str] = {
    "FILTER": "select_filter",
    "CAST": "cast",
    "DEDUPLICATE": "deduplicate",
    "DERIVE": "derive",
    "AGGREGATE": "aggregate",
    "JOIN": "join",
    "VALIDATE": "validate_schema",
    "EXPORT": "export",
}

# Ranking criterion 6 ("reversibility / idempotency") is the one bullet
# that cannot be read off the registry mechanically - it is a property of
# what each operation actually does. Fixed, reviewed text per operation_id
# rather than inferred at call time.
_REVERSIBILITY_NOTES: dict[str, str] = {
    "select_filter": "reversible - operates on a copy; rows/columns dropped are never mutated in the source",
    "cast": "reversible - produces a new typed copy; the operation's evidence records the original dtype",
    "deduplicate": "lossy - removed rows are not recoverable from the output alone; requires an explicit rule and audit evidence",
    "derive": "reversible - adds a new column; never mutates an existing one",
    "aggregate": "lossy by design (row-level detail is not retained) - requires a declared grain and a reconciliation check",
    "join": "reversible on its inputs (neither source is mutated) but can amplify/duplicate rows - requires the Join Safety Reviewer's contract",
    "validate_schema": "read-only - never mutates data",
    "export": "terminal/irreversible side effect - requires policy approval before execution",
}


class StepRequest(BaseModel):
    step_id: str
    step_type: str
    operation_id: str | None = None
    """Explicit override: if a plan draft already names a specific
    operation_id, this agent verifies it rather than re-deriving one from
    step_type."""
    operation_version: str | None = None
    """None resolves to the registry's highest registered version for
    the chosen operation_id, exactly like `OperationRegistry.get`."""


class OperationSelection(BaseModel):
    step_id: str
    operation_id: str
    version: str
    reason: str
    required_params: dict[str, bool]
    """Param name -> whether the operation's typed Params model requires
    it (no default)."""
    certification: str


class UnresolvedStep(BaseModel):
    step_id: str
    step_type: str
    reason: str


class SelectorResult(BaseModel):
    selections: list[OperationSelection] = Field(default_factory=list)
    unresolved: list[UnresolvedStep] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `selections`/`unresolved` remain
    the decision."""


class OperationRegistrySelector:
    def __init__(self, registry: OperationRegistry, llm_client: LLMClient | None = None) -> None:
        self._registry = registry
        self._llm_client = llm_client

    def select(self, steps: list[StepRequest]) -> SelectorResult:
        selections: list[OperationSelection] = []
        unresolved: list[UnresolvedStep] = []

        for step in steps:
            operation_id = step.operation_id or _STEP_TYPE_TO_OPERATION_ID.get(step.step_type)
            if operation_id is None:
                unresolved.append(
                    UnresolvedStep(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        reason=f"no approved deterministic operation is mapped to step type '{step.step_type}'",
                    )
                )
                continue

            try:
                operation = self._registry.get(operation_id, step.operation_version)
            except PlatformError as exc:
                unresolved.append(
                    UnresolvedStep(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        reason=str(exc.reason),
                    )
                )
                continue

            required_params = {
                name: field.is_required() for name, field in operation.Params.model_fields.items()
            }
            reversibility = _REVERSIBILITY_NOTES.get(
                operation.operation_id, "reversibility not documented for this operation"
            )
            selections.append(
                OperationSelection(
                    step_id=step.step_id,
                    operation_id=operation.operation_id,
                    version=operation.version,
                    reason=(
                        f"step type '{step.step_type}' maps to registered operation '{operation.full_id()}' - "
                        f"deterministic and versioned per the Operation Registry contract; {reversibility}"
                    ),
                    required_params=required_params,
                    certification=f"registered:{operation.full_id()}",
                )
            )

        narrative_text = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=OPERATION_REGISTRY_SELECTOR_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {
                            "selections": [s.model_dump() for s in selections],
                            "unresolved": [u.model_dump() for u in unresolved],
                        },
                    }
                ],
            )
            narrative_text = narratives.get("summary")

        return SelectorResult(selections=selections, unresolved=unresolved, narrative=narrative_text)
