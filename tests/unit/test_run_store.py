import pytest

from dataos.errors import PlatformError
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunRecord, RunStore, StepRunRecord, now_iso


def _new_run_record(run_id: str) -> RunRecord:
    ts = now_iso()
    return RunRecord(
        run_id=run_id,
        workflow_version=1,
        requirement_contract_id="rc_1",
        state=RunState.DRAFT,
        source_snapshot_ids={"orders": "abc123"},
        created_at=ts,
        updated_at=ts,
    )


def test_create_and_get_run(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.create_run(_new_run_record("run_1"))

    run = store.get_run("run_1")
    assert run is not None
    assert run.state == RunState.DRAFT
    assert run.source_snapshot_ids == {"orders": "abc123"}


def test_get_run_returns_none_for_unknown_id(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    assert store.get_run("does_not_exist") is None


def test_update_run_state_enforces_state_machine(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.create_run(_new_run_record("run_1"))

    updated = store.update_run_state("run_1", RunState.APPROVED)
    assert updated.state == RunState.APPROVED

    with pytest.raises(PlatformError):
        store.update_run_state("run_1", RunState.RELEASED)  # APPROVED -> RELEASED is illegal


def test_upsert_and_get_step_run(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.create_run(_new_run_record("run_1"))

    step = StepRunRecord(run_id="run_1", step_id="s1", status="RUNNING", started_at=now_iso())
    store.upsert_step_run(step)
    fetched = store.get_step_run("run_1", "s1")
    assert fetched is not None
    assert fetched.status == "RUNNING"

    step.status = "COMPLETED"
    step.output_artifact_id = "run_1/s1"
    step.evidence = {"rows_out": 5}
    store.upsert_step_run(step)  # upsert: same (run_id, step_id) updates in place

    fetched_again = store.get_step_run("run_1", "s1")
    assert fetched_again.status == "COMPLETED"
    assert fetched_again.output_artifact_id == "run_1/s1"
    assert fetched_again.evidence == {"rows_out": 5}
    assert len(store.list_step_runs("run_1")) == 1


def test_state_persists_across_a_new_run_store_instance_pointed_at_same_file(tmp_path):
    db_path = tmp_path / "runs.db"

    store_a = RunStore(db_path)
    store_a.create_run(_new_run_record("run_1"))
    store_a.update_run_state("run_1", RunState.APPROVED)
    store_a.upsert_step_run(StepRunRecord(run_id="run_1", step_id="s1", status="COMPLETED"))
    store_a.close()

    # Simulates a process restart: a brand new RunStore instance, same db file.
    store_b = RunStore(db_path)
    run = store_b.get_run("run_1")
    assert run is not None
    assert run.state == RunState.APPROVED

    step = store_b.get_step_run("run_1", "s1")
    assert step is not None
    assert step.status == "COMPLETED"
