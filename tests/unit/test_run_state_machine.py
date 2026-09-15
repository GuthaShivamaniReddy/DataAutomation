import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.workflow.state_machine import RunState, transition


@pytest.mark.parametrize(
    "current,target",
    [
        (RunState.DRAFT, RunState.NEEDS_CLARIFICATION),
        (RunState.DRAFT, RunState.APPROVED),
        (RunState.NEEDS_CLARIFICATION, RunState.APPROVED),
        (RunState.APPROVED, RunState.RUNNING),
        (RunState.RUNNING, RunState.VERIFYING),
        (RunState.RUNNING, RunState.QUARANTINED),
        (RunState.VERIFYING, RunState.RELEASED),
        (RunState.VERIFYING, RunState.QUARANTINED),
        (RunState.RELEASED, RunState.ROLLED_BACK),
    ],
)
def test_allowed_transitions_from_blueprint_9_2(current, target):
    transition(current, target)  # must not raise


@pytest.mark.parametrize(
    "current,target",
    [
        (RunState.DRAFT, RunState.RUNNING),
        (RunState.DRAFT, RunState.RELEASED),
        (RunState.RUNNING, RunState.RELEASED),
        (RunState.QUARANTINED, RunState.RUNNING),
        (RunState.QUARANTINED, RunState.RELEASED),
        (RunState.RELEASED, RunState.RUNNING),
        (RunState.ROLLED_BACK, RunState.APPROVED),
        (RunState.VERIFYING, RunState.APPROVED),
    ],
)
def test_illegal_transitions_rejected(current, target):
    with pytest.raises(PlatformError) as excinfo:
        transition(current, target)
    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


def test_terminal_states_have_no_outgoing_transitions():
    for terminal in (RunState.QUARANTINED, RunState.ROLLED_BACK):
        for candidate in RunState:
            if candidate == terminal:
                continue
            with pytest.raises(PlatformError):
                transition(terminal, candidate)
