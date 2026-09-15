"""Run state machine.

Source of truth: Reliability-First Master Blueprint Section 9.2 "State
transition rules". This is the *workflow-run* state machine, distinct
from `RequirementStatus` (dataos.contracts.requirement_contract) which is
the contract's own DRAFT/APPROVED/BLOCKED status - two different state
machines for two different objects, even though a couple of names
overlap.

"There is no state called 'probably correct'. Unverified outputs are
never published as final." (Blueprint Section 9 diagram caption) - this
module is what makes that a structural guarantee: any transition not
explicitly listed in Blueprint 9.2's table raises rather than silently
succeeding.
"""

from __future__ import annotations

from enum import Enum

from dataos.errors import ErrorCode, PlatformError


class RunState(str, Enum):
    DRAFT = "DRAFT"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    RELEASED = "RELEASED"
    QUARANTINED = "QUARANTINED"
    ROLLED_BACK = "ROLLED_BACK"


# Exactly the edges in Blueprint 9.2's table - nothing more.
_ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.DRAFT: frozenset({RunState.NEEDS_CLARIFICATION, RunState.APPROVED}),
    RunState.NEEDS_CLARIFICATION: frozenset({RunState.APPROVED}),
    RunState.APPROVED: frozenset({RunState.RUNNING}),
    RunState.RUNNING: frozenset({RunState.VERIFYING, RunState.QUARANTINED}),
    RunState.VERIFYING: frozenset({RunState.RELEASED, RunState.QUARANTINED}),
    RunState.RELEASED: frozenset({RunState.ROLLED_BACK}),
    RunState.QUARANTINED: frozenset(),
    RunState.ROLLED_BACK: frozenset(),
}


def transition(current: RunState, target: RunState) -> None:
    """Raise PlatformError(VALIDATION_FAIL) unless `current -> target` is
    one of Blueprint 9.2's explicit edges. Callers apply the transition
    themselves (e.g. persisting it) only after this does not raise.
    """
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise PlatformError(
            ErrorCode.VALIDATION_FAIL,
            f"illegal run state transition {current.value} -> {target.value}",
            evidence={"current": current.value, "target": target.value, "allowed": sorted(s.value for s in allowed)},
        )
