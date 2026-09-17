from dataos.compiler.incident_recovery_agent import IncidentRecoveryAgent
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunRecord, StepRunRecord, now_iso


def _run(state: RunState = RunState.RUNNING) -> RunRecord:
    return RunRecord(
        run_id="r1",
        workflow_version=1,
        requirement_contract_id="rc1",
        state=state,
        created_at=now_iso(),
        updated_at=now_iso(),
    )


def test_stalled_running_step_is_transient_and_safe_to_retry():
    step_runs = [StepRunRecord(run_id="r1", step_id="s1", status="RUNNING", started_at=now_iso())]

    result = IncidentRecoveryAgent().diagnose(run=_run(), step_runs=step_runs)

    assert result.failure_class == "TRANSIENT"
    assert result.safe_action == "RETRY"
    assert result.retry_from_step == "s1"


def test_data_failure_is_quarantined_not_retried():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "VALIDATION_FAIL", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(run=_run(RunState.QUARANTINED), step_runs=step_runs)

    assert result.failure_class == "DATA"
    assert result.safe_action == "QUARANTINE"
    assert result.retry_from_step is None


def test_logic_failure_is_quarantined():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "NO_SAFE_OPERATION", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(run=_run(RunState.QUARANTINED), step_runs=step_runs)

    assert result.failure_class == "LOGIC"
    assert result.safe_action == "QUARANTINE"


def test_policy_denied_is_a_security_stop_not_an_escalation():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "POLICY_DENIED", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(run=_run(RunState.QUARANTINED), step_runs=step_runs)

    assert result.failure_class == "SECURITY"
    assert result.safe_action == "STOP"


def test_external_write_unverified_stops_pending_reconciliation_by_default():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "EXTERNAL_WRITE_UNVERIFIED", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(run=_run(RunState.QUARANTINED), step_runs=step_runs)

    assert result.failure_class == "EXTERNAL_SIDE_EFFECT"
    assert result.safe_action == "STOP"


def test_external_write_unverified_rolls_back_with_an_approved_compensating_policy():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "EXTERNAL_WRITE_UNVERIFIED", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(
        run=_run(RunState.QUARANTINED), step_runs=step_runs, rollback_policy="issue a compensating refund via the CRM API"
    )

    assert result.safe_action == "ROLLBACK"


def test_default_rollback_placeholder_is_not_treated_as_approved():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "EXTERNAL_WRITE_UNVERIFIED", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(
        run=_run(RunState.QUARANTINED),
        step_runs=step_runs,
        rollback_policy="no compensating action defined for this automation's write-capable steps",
    )

    assert result.safe_action == "STOP"


def test_unrecognized_error_code_is_unknown_and_escalated():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "TOTALLY_MADE_UP", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(run=_run(RunState.QUARANTINED), step_runs=step_runs)

    assert result.failure_class == "UNKNOWN"
    assert result.safe_action == "ESCALATE"


def test_repeated_failures_escalate_regardless_of_class():
    step_runs = [
        StepRunRecord(
            run_id="r1", step_id="s1", status="FAILED",
            error={"code": "VALIDATION_FAIL", "reason": "x", "evidence": {}},
            started_at=now_iso(), completed_at=now_iso(),
        )
    ]

    result = IncidentRecoveryAgent().diagnose(run=_run(RunState.QUARANTINED), step_runs=step_runs, prior_failure_count=3)

    assert result.safe_action == "ESCALATE"
    assert result.retry_from_step is None


def test_no_incident_evidence_returns_a_neutral_result():
    step_runs = [StepRunRecord(run_id="r1", step_id="s1", status="COMPLETED", started_at=now_iso(), completed_at=now_iso())]

    result = IncidentRecoveryAgent().diagnose(run=_run(RunState.RELEASED), step_runs=step_runs)

    assert result.safe_action == "STOP"
    assert "nothing to recover" in result.conditions[0]


def test_evidence_to_preserve_always_includes_the_run_record():
    step_runs = [StepRunRecord(run_id="r1", step_id="s1", status="RUNNING", started_at=now_iso())]

    result = IncidentRecoveryAgent().diagnose(run=_run(), step_runs=step_runs)

    assert any("RunRecord for 'r1'" in e for e in result.evidence_to_preserve)


def test_narrative_is_none_without_an_llm_client():
    step_runs = [StepRunRecord(run_id="r1", step_id="s1", status="RUNNING", started_at=now_iso())]

    result = IncidentRecoveryAgent().diagnose(run=_run(), step_runs=step_runs)

    assert result.narrative is None


def test_narrative_is_populated_without_changing_the_decision_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    step_runs = [StepRunRecord(run_id="r1", step_id="s1", status="RUNNING", started_at=now_iso())]

    agent = IncidentRecoveryAgent(llm_client=DeterministicLLMClient())
    result = agent.diagnose(run=_run(), step_runs=step_runs)

    assert result.narrative is not None
    assert result.safe_action == "RETRY"
