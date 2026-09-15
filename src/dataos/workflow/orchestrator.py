"""Crash-safe local workflow orchestrator.

Source of truth: Blueprint Section 9.2's state table and the Phase 3 exit
criterion: "worker crash/retry does not duplicate steps or corrupt run
state; source versions pinned."

The crash-safety mechanism is simple and load-bearing: every step's
completion is durably recorded (RunStore, SQLite, committed immediately)
before `run()` moves on to the next step, and `run()` always checks for
an existing COMPLETED StepRunRecord before doing any work. Calling
`run()` a second time on the same run_id - whether because the process
actually crashed and was restarted, or just because a caller retried -
re-executes nothing that already finished.
"""

from __future__ import annotations

import uuid

import polars as pl

from dataos.compiler.workflow_planner import PlannerOutput, WorkflowPlanner
from dataos.contracts.requirement_contract import RequirementContract
from dataos.errors import ErrorCode, PlatformError
from dataos.ingestion.snapshot import DatasetVersion
from dataos.registry.base import Operation
from dataos.registry.registry import OperationRegistry
from dataos.workflow.artifact_store import ArtifactStore
from dataos.workflow.dsl import SOURCE_PREFIX, Workflow
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunRecord, RunStore, StepRunRecord, now_iso

_TERMINAL_OR_WAITING_STATES = frozenset(
    {
        RunState.DRAFT,
        RunState.NEEDS_CLARIFICATION,
        RunState.APPROVED,
        RunState.VERIFYING,
        RunState.RELEASED,
        RunState.QUARANTINED,
        RunState.ROLLED_BACK,
    }
)


class WorkflowOrchestrator:
    def __init__(
        self,
        registry: OperationRegistry,
        run_store: RunStore,
        artifact_store: ArtifactStore,
        planner: WorkflowPlanner | None = None,
    ) -> None:
        self._registry = registry
        self._run_store = run_store
        self._artifact_store = artifact_store
        self._planner = planner

    def start_run(
        self,
        workflow: Workflow,
        *,
        source_frames: dict[str, pl.DataFrame],
        source_versions: dict[str, DatasetVersion],
    ) -> str:
        """Validate the DAG, create the run, lock source snapshot ids, and
        seed each declared source as an already-COMPLETED pseudo-step so a
        later `run()` (including after a real crash) never needs the
        original in-memory frames again - only the run_id.
        """
        workflow.validate_dag()

        missing_frames = [s for s in workflow.sources if s not in source_frames]
        missing_versions = [s for s in workflow.sources if s not in source_versions]
        if missing_frames or missing_versions:
            raise PlatformError(
                ErrorCode.SCHEMA_MISSING,
                "start_run requires a frame and a DatasetVersion for every declared source",
                evidence={"missing_frames": missing_frames, "missing_versions": missing_versions},
            )

        run_id = str(uuid.uuid4())
        source_snapshot_ids = {name: source_versions[name].snapshot_id for name in workflow.sources}

        self._run_store.create_run(
            RunRecord(
                run_id=run_id,
                workflow_version=workflow.workflow_version,
                requirement_contract_id=workflow.requirement_contract_id,
                state=RunState.DRAFT,
                source_snapshot_ids=source_snapshot_ids,
                created_at=now_iso(),
                updated_at=now_iso(),
            )
        )
        # Blueprint 9.2: DRAFT -> APPROVED requires "contract complete + policy
        # pass". A real Policy/Approval Gate arrives in a later phase; until
        # then this transition is a documented no-op placeholder, not a
        # bypass - the run genuinely cannot proceed without passing through it.
        self._run_store.update_run_state(run_id, RunState.APPROVED)
        # Blueprint 9.2: APPROVED -> RUNNING requires "worker allocated +
        # source snapshot locked" - source_snapshot_ids above is exactly that lock.
        self._run_store.update_run_state(run_id, RunState.RUNNING)

        for name in workflow.sources:
            node_id = f"{SOURCE_PREFIX}{name}"
            artifact_ref = self._artifact_store.save(run_id, node_id, source_frames[name])
            timestamp = now_iso()
            self._run_store.upsert_step_run(
                StepRunRecord(
                    run_id=run_id,
                    step_id=node_id,
                    status="COMPLETED",
                    output_artifact_id=artifact_ref.artifact_id,
                    evidence={"role": "source", "snapshot_id": source_snapshot_ids[name]},
                    started_at=timestamp,
                    completed_at=timestamp,
                )
            )

        return run_id

    def start_run_from_contract(
        self,
        *,
        contract: RequirementContract,
        contract_id: str,
        source_frames: dict[str, pl.DataFrame],
        source_versions: dict[str, DatasetVersion],
        workflow_version: int = 1,
    ) -> tuple[str, PlannerOutput]:
        """Compose the Workflow Planner (AI Prompt Library Section 9) with
        `start_run`: Prompt Library "Agent pipeline" ordering is
        "... -> Semantic Resolver -> Planner -> Policy / Approval Gate ->
        Deterministic Executor -> ...", so planning happens here, before a
        run is created, never inside `run()` itself.

        Callers that already hold a typed `Workflow` (every existing
        caller/test) keep using `start_run` directly - this only exists for
        callers starting from an approved `RequirementContract`. Returns
        the `PlannerOutput` alongside `run_id` because - like `start_run` -
        this orchestrator does not persist the `Workflow` object itself;
        the caller must hold onto it to later call `run(run_id, workflow)`.
        """
        if self._planner is None:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                "orchestrator was constructed without a WorkflowPlanner; pass one to plan from a RequirementContract",
            )

        planner_output = self._planner.plan(
            contract=contract,
            contract_id=contract_id,
            workflow_version=workflow_version,
        )

        run_id = self.start_run(
            planner_output.workflow,
            source_frames=source_frames,
            source_versions=source_versions,
        )

        return run_id, planner_output

    def run(self, run_id: str, workflow: Workflow) -> RunRecord:
        run = self._run_store.get_run(run_id)
        if run is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no run with id '{run_id}'")

        if run.state in _TERMINAL_OR_WAITING_STATES:
            return run  # nothing to do: already finished, quarantined, or not yet approved/running

        for step_id in workflow.execution_order():
            existing = self._run_store.get_step_run(run_id, step_id)
            if existing is not None and existing.status == "COMPLETED":
                continue  # crash-safe resume: never re-execute a finished step

            step = workflow.step_by_id(step_id)

            if len(step.inputs) != 1:
                return self._fail_step_and_quarantine(
                    run_id,
                    step_id,
                    operation_full_id=step.full_operation_id,
                    reason=(
                        f"step '{step_id}' declares {len(step.inputs)} input(s); only "
                        "single-input operations are supported until a multi-input "
                        "operation (e.g. join) is registered"
                    ),
                )

            input_df = self._artifact_store.load_by_ids(run_id, step.inputs[0])

            started_at = now_iso()
            self._run_store.upsert_step_run(
                StepRunRecord(run_id=run_id, step_id=step_id, status="RUNNING", started_at=started_at)
            )

            operation: Operation = self._registry.get(step.operation_id, step.operation_version)
            try:
                result = operation.execute(input_df, step.params)
            except PlatformError as exc:
                self._run_store.upsert_step_run(
                    StepRunRecord(
                        run_id=run_id,
                        step_id=step_id,
                        status="FAILED",
                        operation_full_id=step.full_operation_id,
                        error=exc.to_dict(),
                        started_at=started_at,
                        completed_at=now_iso(),
                    )
                )
                return self._run_store.update_run_state(run_id, RunState.QUARANTINED)

            artifact_ref = self._artifact_store.save(run_id, step_id, result.output)
            self._run_store.upsert_step_run(
                StepRunRecord(
                    run_id=run_id,
                    step_id=step_id,
                    status="COMPLETED",
                    operation_full_id=step.full_operation_id,
                    output_artifact_id=artifact_ref.artifact_id,
                    evidence=result.evidence,
                    started_at=started_at,
                    completed_at=now_iso(),
                )
            )

        return self._run_store.update_run_state(run_id, RunState.VERIFYING)

    def _fail_step_and_quarantine(self, run_id: str, step_id: str, *, operation_full_id: str, reason: str) -> RunRecord:
        timestamp = now_iso()
        self._run_store.upsert_step_run(
            StepRunRecord(
                run_id=run_id,
                step_id=step_id,
                status="FAILED",
                operation_full_id=operation_full_id,
                error={"code": ErrorCode.VALIDATION_FAIL.value, "reason": reason, "evidence": {}},
                started_at=timestamp,
                completed_at=timestamp,
            )
        )
        return self._run_store.update_run_state(run_id, RunState.QUARANTINED)
