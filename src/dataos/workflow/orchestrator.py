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
from pydantic import BaseModel

from dataos.compiler.explanation_agent import Audience, ExplanationAgent, ExplanationOutput
from dataos.compiler.independent_verifier import IndependentVerifier, VerifierResult
from dataos.compiler.policy_gate import PolicyDecision, PolicyGate
from dataos.compiler.release_gate import ReleaseDecision, ReleaseGate
from dataos.compiler.validation_rule_generator import CheckSpec, ValidationRuleGenerator
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


class PlannedRun(BaseModel):
    """Result of `start_run_from_contract`: the created run plus the
    evidence (Planner envelope, Policy Gate decision) that got it there -
    Prompt Library Section 38 "Persist every agent input/output for audit"."""

    run_id: str
    planner_output: PlannerOutput
    policy_decision: PolicyDecision


class ReleaseResult(BaseModel):
    """Result of `release()`: the (possibly newly RELEASED/QUARANTINED)
    run, plus the Independent Verifier and Release Gate evidence behind
    that decision - Prompt Library Section 38 "Persist every agent
    input/output for audit"."""

    run: RunRecord
    verifier_result: VerifierResult
    release_decision: ReleaseDecision


_RELEASE_TERMINAL_STATES = frozenset({RunState.RELEASED, RunState.QUARANTINED, RunState.ROLLED_BACK})


class WorkflowOrchestrator:
    def __init__(
        self,
        registry: OperationRegistry,
        run_store: RunStore,
        artifact_store: ArtifactStore,
        planner: WorkflowPlanner | None = None,
        policy_gate: PolicyGate | None = None,
        verifier: IndependentVerifier | None = None,
        release_gate: ReleaseGate | None = None,
        explainer: ExplanationAgent | None = None,
        rule_generator: ValidationRuleGenerator | None = None,
    ) -> None:
        self._registry = registry
        self._run_store = run_store
        self._artifact_store = artifact_store
        self._planner = planner
        self._policy_gate = policy_gate or PolicyGate()
        self._verifier = verifier or IndependentVerifier()
        self._release_gate = release_gate or ReleaseGate()
        self._explainer = explainer
        self._rule_generator = rule_generator or ValidationRuleGenerator()

    def generate_validation_checks(self, workflow: Workflow, contract: RequirementContract) -> list[CheckSpec]:
        """Validation Rule Generator (Section 19). Pure and read-only - it
        produces the check manifest for a planned workflow but does not
        execute or store anything, so it can be called any time after a
        `Workflow` exists (typically right after planning, before or
        alongside `run()`)."""
        return self._rule_generator.generate(contract=contract, workflow=workflow)

    def start_run(
        self,
        workflow: Workflow,
        *,
        source_frames: dict[str, pl.DataFrame],
        source_versions: dict[str, DatasetVersion],
        contract: RequirementContract | None = None,
        granted_approvals: frozenset[str] = frozenset(),
    ) -> str:
        """Validate the DAG, create the run, lock source snapshot ids, and
        seed each declared source as an already-COMPLETED pseudo-step so a
        later `run()` (including after a real crash) never needs the
        original in-memory frames again - only the run_id.

        `contract` is optional context for the Policy/Approval Gate below -
        a caller with only a typed `Workflow` still gets the
        operation-risk and `workflow.approval_gates` checks (see
        `PolicyGate.evaluate`). This is the one place every run must pass
        through regardless of entry point (`start_run` directly, or
        `start_run_from_contract`), so it is the only place the gate is
        enforced - a hand-built `Workflow` with an ungated `export` step
        cannot skip it by bypassing `start_run_from_contract`.
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

        policy_decision = self._policy_gate.evaluate(
            workflow=workflow, contract=contract, granted_approvals=granted_approvals
        )
        if policy_decision.decision == "BLOCKED":
            raise PlatformError(
                ErrorCode.POLICY_DENIED,
                "workflow blocked by the policy/approval gate",
                evidence={
                    "risk_level": policy_decision.risk_level,
                    "required_approvals": policy_decision.required_approvals,
                    "blocking_reasons": policy_decision.blocking_reasons,
                },
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
        # Blueprint 9.2: DRAFT -> APPROVED requires "contract complete +
        # policy pass" - the policy_gate check above is exactly that pass;
        # a BLOCKED decision raises before a run is ever created.
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
        granted_approvals: frozenset[str] = frozenset(),
    ) -> PlannedRun:
        """Compose the Workflow Planner (AI Prompt Library Section 9) with
        the Policy/Approval Gate and `start_run`: Prompt Library "Agent
        pipeline" ordering is "... -> Semantic Resolver -> Planner ->
        Policy / Approval Gate -> Deterministic Executor -> ...".

        Evaluating the gate here (with the full `RequirementContract`, so
        `required_approvals`/`side_effects`/`prohibited_actions` are all in
        scope) lets a blocked plan fail fast with contract-aware evidence
        before a run row is ever created; `start_run` itself evaluates the
        same gate again (workflow-only, since it has no contract) as the
        one enforcement point no caller can bypass - see its docstring.

        Callers that already hold a typed `Workflow` (every existing
        caller/test) keep using `start_run` directly - this only exists for
        callers starting from an approved `RequirementContract`.
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

        policy_decision = self._policy_gate.evaluate(
            workflow=planner_output.workflow, contract=contract, granted_approvals=granted_approvals
        )
        if policy_decision.decision == "BLOCKED":
            raise PlatformError(
                ErrorCode.POLICY_DENIED,
                "planned workflow blocked by the policy/approval gate",
                evidence={
                    "risk_level": policy_decision.risk_level,
                    "required_approvals": policy_decision.required_approvals,
                    "blocking_reasons": policy_decision.blocking_reasons,
                },
            )

        run_id = self.start_run(
            planner_output.workflow,
            source_frames=source_frames,
            source_versions=source_versions,
            contract=contract,
            granted_approvals=granted_approvals,
        )

        return PlannedRun(run_id=run_id, planner_output=planner_output, policy_decision=policy_decision)

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

    def release(self, run_id: str, workflow: Workflow, contract: RequirementContract) -> ReleaseResult:
        """Independent Verifier (Section 21) + Release/Quarantine Gate
        (Section 23): Prompt Library "Agent pipeline" ordering is
        "... -> Validation & Reconciliation -> Independent Verifier ->
        Release / Quarantine Gate -> Explanation & Delivery". This is the
        VERIFYING -> RELEASED/QUARANTINED transition `run()` deliberately
        never makes itself - Blueprint 9.2 keeps verification and release
        as a separate step from execution, and Section 23: "You are the
        only component authorized to mark an analytical result as
        released."

        Calling this again on an already-RELEASED/QUARANTINED/ROLLED_BACK
        run recomputes the same (pure, deterministic) verdict from the
        recorded step evidence but does not attempt the transition again -
        `state_machine.transition` has no outgoing edge from a terminal
        state, so a second real transition attempt would raise.
        """
        run = self._run_store.get_run(run_id)
        if run is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no run with id '{run_id}'")

        step_runs = self._run_store.list_step_runs(run_id)
        verifier_result = self._verifier.verify(contract=contract, workflow=workflow, run=run, step_runs=step_runs)
        release_decision = self._release_gate.decide(contract=contract, run=run, verifier_result=verifier_result)

        if run.state in _RELEASE_TERMINAL_STATES:
            return ReleaseResult(run=run, verifier_result=verifier_result, release_decision=release_decision)

        new_state = RunState.RELEASED if release_decision.decision == "RELEASE" else RunState.QUARANTINED
        run = self._run_store.update_run_state(run_id, new_state)

        return ReleaseResult(run=run, verifier_result=verifier_result, release_decision=release_decision)

    def explain(
        self,
        run_id: str,
        workflow: Workflow,
        contract: RequirementContract,
        *,
        audience: Audience = "analyst",
        sample_rows: int = 20,
    ) -> ExplanationOutput:
        """Report & Explanation Agent (Section 18), the last stage of the
        pipeline: "... -> Release / Quarantine Gate -> Explanation &
        Delivery". Only ever callable on a RELEASED run - Section 18:
        "Write the user-facing answer using only RELEASED result objects
        and evidence supplied to you."

        Builds the evidence context itself (contract framing + a bounded
        sample of each covered metric's actual released output rows, read
        from the artifact store) rather than delegating that to the
        agent, matching Section 38 "Runtime context discipline": only the
        artifact-derived evidence needed to write about *this* run is
        sent, never raw source data. Independently re-runs the Verifier
        (cheap and pure) so any `unverified_claims` are force-included as
        limitations - Section 18 rule 8: "Do not hide validation
        warnings" - rather than trusted to a stale, previously-computed
        verdict.
        """
        if self._explainer is None:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                "orchestrator was constructed without an ExplanationAgent; pass one to call explain()",
            )

        run = self._run_store.get_run(run_id)
        if run is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no run with id '{run_id}'")
        if run.state != RunState.RELEASED:
            raise PlatformError(
                ErrorCode.VALIDATION_FAIL,
                "cannot explain a run that is not RELEASED",
                evidence={"run_id": run_id, "state": run.state.value},
            )

        all_step_runs = self._run_store.list_step_runs(run_id)
        step_runs = {s.step_id: s for s in all_step_runs}
        verifier_result = self._verifier.verify(contract=contract, workflow=workflow, run=run, step_runs=all_step_runs)

        metrics_context = []
        known_refs: set[str] = set()
        for metric in contract.metrics:
            covering_steps = [s for s in workflow.steps if metric.name in s.requirement_refs]
            if not covering_steps:
                continue
            step = covering_steps[0]
            record = step_runs.get(step.id)
            if record is None or record.status != "COMPLETED":
                continue
            output_df = self._artifact_store.load_by_ids(run_id, step.id)
            sample_records = output_df.head(sample_rows).to_dicts()
            metrics_context.append(
                {
                    "name": metric.name,
                    "formula": metric.formula,
                    "step_id": step.id,
                    "sample_records": sample_records,
                }
            )
            known_refs.add(metric.name)
            known_refs.add(step.id)

        evidence_context = {
            "objective": contract.objective,
            "grain": contract.grain,
            "filters": contract.filters,
            "time": contract.time.model_dump(),
            "units": contract.units.model_dump(),
            "metrics": metrics_context,
        }

        return self._explainer.explain(
            run_id=run_id,
            evidence_context=evidence_context,
            known_evidence_refs=frozenset(known_refs),
            forced_limitations=verifier_result.unverified_claims,
            audience=audience,
        )

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
