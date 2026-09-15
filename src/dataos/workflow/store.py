"""Durable Run / StepRun persistence.

Source of truth: Reliability-First Master Blueprint Section 16 (`run`,
`step_run` entities) and the Phase 3 exit criterion: "worker crash/retry
does not duplicate steps or corrupt run state; source versions pinned."

Every write here is committed immediately (SQLite autocommit-per-call) so
that if the process dies between two steps, the on-disk record reflects
exactly which steps genuinely finished - `WorkflowOrchestrator` relies on
this to safely skip already-COMPLETED steps on a later resume instead of
re-executing them.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from dataos.errors import ErrorCode, PlatformError
from dataos.workflow.state_machine import RunState, transition

DEFAULT_RUN_DB_PATH = Path(".dataos_store") / "runs.db"

StepStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]


class StepRunRecord(BaseModel):
    run_id: str
    step_id: str
    status: StepStatus
    operation_full_id: str | None = None
    output_artifact_id: str | None = None
    evidence: dict = Field(default_factory=dict)
    error: dict | None = None
    started_at: str | None = None
    completed_at: str | None = None


class RunRecord(BaseModel):
    run_id: str
    workflow_version: int
    requirement_contract_id: str
    state: RunState
    source_snapshot_ids: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, db_path: str | Path = DEFAULT_RUN_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, isolation_level=None)  # autocommit
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                workflow_version INTEGER NOT NULL,
                requirement_contract_id TEXT NOT NULL,
                state TEXT NOT NULL,
                source_snapshot_ids TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS step_runs (
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                status TEXT NOT NULL,
                operation_full_id TEXT,
                output_artifact_id TEXT,
                evidence TEXT NOT NULL,
                error TEXT,
                started_at TEXT,
                completed_at TEXT,
                PRIMARY KEY (run_id, step_id)
            )
            """
        )

    def create_run(self, run: RunRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, workflow_version, requirement_contract_id, state,
                               source_snapshot_ids, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.workflow_version,
                run.requirement_contract_id,
                run.state.value,
                json.dumps(run.source_snapshot_ids),
                run.created_at,
                run.updated_at,
            ),
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT run_id, workflow_version, requirement_contract_id, state, "
            "source_snapshot_ids, created_at, updated_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row[0],
            workflow_version=row[1],
            requirement_contract_id=row[2],
            state=RunState(row[3]),
            source_snapshot_ids=json.loads(row[4]),
            created_at=row[5],
            updated_at=row[6],
        )

    def update_run_state(self, run_id: str, new_state: RunState) -> RunRecord:
        run = self.get_run(run_id)
        if run is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no run with id '{run_id}'")
        transition(run.state, new_state)
        self._conn.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
            (new_state.value, now_iso(), run_id),
        )
        updated = self.get_run(run_id)
        assert updated is not None
        return updated

    def upsert_step_run(self, step: StepRunRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO step_runs (run_id, step_id, status, operation_full_id, output_artifact_id,
                                    evidence, error, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, step_id) DO UPDATE SET
                status = excluded.status,
                operation_full_id = excluded.operation_full_id,
                output_artifact_id = excluded.output_artifact_id,
                evidence = excluded.evidence,
                error = excluded.error,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at
            """,
            (
                step.run_id,
                step.step_id,
                step.status,
                step.operation_full_id,
                step.output_artifact_id,
                json.dumps(step.evidence),
                json.dumps(step.error) if step.error is not None else None,
                step.started_at,
                step.completed_at,
            ),
        )

    def get_step_run(self, run_id: str, step_id: str) -> StepRunRecord | None:
        row = self._conn.execute(
            "SELECT run_id, step_id, status, operation_full_id, output_artifact_id, "
            "evidence, error, started_at, completed_at FROM step_runs "
            "WHERE run_id = ? AND step_id = ?",
            (run_id, step_id),
        ).fetchone()
        if row is None:
            return None
        return StepRunRecord(
            run_id=row[0],
            step_id=row[1],
            status=row[2],
            operation_full_id=row[3],
            output_artifact_id=row[4],
            evidence=json.loads(row[5]),
            error=json.loads(row[6]) if row[6] is not None else None,
            started_at=row[7],
            completed_at=row[8],
        )

    def list_step_runs(self, run_id: str) -> list[StepRunRecord]:
        rows = self._conn.execute(
            "SELECT run_id, step_id, status, operation_full_id, output_artifact_id, "
            "evidence, error, started_at, completed_at FROM step_runs WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return [
            StepRunRecord(
                run_id=r[0],
                step_id=r[1],
                status=r[2],
                operation_full_id=r[3],
                output_artifact_id=r[4],
                evidence=json.loads(r[5]),
                error=json.loads(r[6]) if r[6] is not None else None,
                started_at=r[7],
                completed_at=r[8],
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
