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

from dataos.compiler.automation_builder import AutomationSpec, AutomationWorkflowBuilder
from dataos.compiler.confidence_scorer import ConfidenceReport, ConfidenceScorer
from dataos.compiler.connector_planner import ConnectorWritePlanner, ExternalActionPlan, ExternalActionVerification
from dataos.compiler.explanation_agent import Audience, ExplanationAgent, ExplanationOutput
from dataos.compiler.independent_verifier import IndependentVerifier, VerifierResult
from dataos.compiler.pii_classifier import PIIClassifier
from dataos.compiler.policy_gate import PolicyDecision, PolicyGate
from dataos.compiler.reconciliation_agent import ReconciliationAgent, ReconciliationReport
from dataos.compiler.release_gate import ReleaseDecision, ReleaseGate
from dataos.compiler.schema_drift_monitor import DriftReport, SchemaDriftMonitor
from dataos.compiler.security_guard import SecurityGuard
from dataos.compiler.validation_rule_generator import CheckSpec, ValidationRuleGenerator
from dataos.compiler.workflow_planner import PlannerOutput, WorkflowPlanner
from dataos.contracts.requirement_contract import RequirementContract
from dataos.errors import ErrorCode, PlatformError
from dataos.ingestion.profiling import DatasetProfile, profile_dataset
from dataos.ingestion.snapshot import DatasetVersion
from dataos.registry.base import Operation
from dataos.registry.registry import OperationRegistry
from dataos.semantics.dictionary import SemanticDictionary
from dataos.validation.engine import ValidationReport
from dataos.validation.executor import ValidationRuleExecutor
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
    run, plus the Validation, Reconciliation, Independent Verifier,
    Release Gate, and Confidence Scorer evidence behind that decision -
    Prompt Library Section 38 "Persist every agent input/output for
    audit"."""

    run: RunRecord
    validation_report: ValidationReport
    reconciliation_report: ReconciliationReport
    verifier_result: VerifierResult
    release_decision: ReleaseDecision
    confidence_report: ConfidenceReport

    model_config = {"arbitrary_types_allowed": True}


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
        rule_executor: ValidationRuleExecutor | None = None,
        reconciliation_agent: ReconciliationAgent | None = None,
        drift_monitor: SchemaDriftMonitor | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        security_guard: SecurityGuard | None = None,
        pii_classifier: PIIClassifier | None = None,
        connector_planner: ConnectorWritePlanner | None = None,
        automation_builder: AutomationWorkflowBuilder | None = None,
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
        self._rule_executor = rule_executor or ValidationRuleExecutor()
        self._reconciliation_agent = reconciliation_agent or ReconciliationAgent()
        self._drift_monitor = drift_monitor or SchemaDriftMonitor()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._security_guard = security_guard or SecurityGuard()
        self._pii_classifier = pii_classifier or PIIClassifier()
        self._connector_planner = connector_planner
        self._automation_builder = automation_builder or AutomationWorkflowBuilder()

    def check_drift(
        self,
        *,
        source_frames: dict[str, pl.DataFrame],
        baseline_profiles: dict[str, DatasetProfile],
        baseline_dictionary: SemanticDictionary | None = None,
        current_dictionary: SemanticDictionary | None = None,
    ) -> dict[str, DriftReport]:
        """Schema and Semantic Drift Monitor (Section 25). Pure and
        read-only: profiles each source with a supplied baseline fresh and
        compares it, but does not raise or block anything itself - a
        source with no baseline supplied is simply not checked (there is
        nothing to compare against). `start_run` calls this internally and
        does enforce the result when `baseline_profiles` is passed to it.
        """
        reports: dict[str, DriftReport] = {}
        for name, baseline_profile in baseline_profiles.items():
            frame = source_frames.get(name)
            if frame is None:
                continue
            reports[name] = self._drift_monitor.compare(
                source_name=name,
                baseline_profile=baseline_profile,
                current_profile=profile_dataset(frame),
                baseline_dictionary=baseline_dictionary,
                current_dictionary=current_dictionary,
            )
        return reports

    def generate_validation_checks(self, workflow: Workflow, contract: RequirementContract) -> list[CheckSpec]:
        """Validation Rule Generator (Section 19). Pure and read-only - it
        produces the check manifest for a planned workflow but does not
        execute or store anything, so it can be called any time after a
        `Workflow` exists (typically right after planning, before or
        alongside `run()`)."""
        return self._rule_generator.generate(contract=contract, workflow=workflow)

    def plan_external_actions(self, workflow: Workflow) -> list[ExternalActionPlan]:
        """Connector / External Write Planner (Section 29). Pure and
        read-only - produces the least-privilege action plan for a
        workflow's write-capable steps but does not execute or gate
        anything itself (`PolicyGate`, enforced at `start_run`, remains
        the actual approval gate regardless of whether anyone reads this
        plan). Callable any time after a `Workflow` exists."""
        if self._connector_planner is None:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                "orchestrator was constructed without a ConnectorWritePlanner; pass one to call plan_external_actions()",
            )
        return self._connector_planner.plan(workflow)

    def verify_external_actions(self, run_id: str, workflow: Workflow) -> list[ExternalActionVerification]:
        """Independent post-write verification for every planned external
        action: re-derives its answer from the real StepRunRecords this
        run produced, never from the plan alone."""
        if self._connector_planner is None:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                "orchestrator was constructed without a ConnectorWritePlanner; pass one to call verify_external_actions()",
            )
        run = self._run_store.get_run(run_id)
        if run is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no run with id '{run_id}'")

        plans = self._connector_planner.plan(workflow)
        step_runs = {s.step_id: s for s in self._run_store.list_step_runs(run_id)}
        return self._connector_planner.verify(plans, step_runs)

    def build_automation(
        self,
        *,
        automation_id: str,
        workflow: Workflow,
        trigger: dict,
        notifications: list[str] | None = None,
        retry_policy: str = "no automatic retry - a QUARANTINED run must be reviewed and re-triggered manually",
        rollback_policy: str = "no compensating action defined for this automation's write-capable steps",
        sla: str | None = None,
    ) -> AutomationSpec:
        """Automation Workflow Builder (Section 24). Pure and read-only -
        version pins, preflight, and approval gates are read from the real
        `workflow`; `trigger`/`notifications`/`retry_policy`/
        `rollback_policy`/`sla` are always caller-supplied, never guessed.
        Call this once, against a `Workflow` that has already reached
        RELEASED, to get the `AutomationSpec` a scheduler should hold onto
        and present to every future `start_automated_run()` call."""
        return self._automation_builder.build(
            automation_id=automation_id,
            workflow=workflow,
            trigger=trigger,
            notifications=notifications,
            retry_policy=retry_policy,
            rollback_policy=rollback_policy,
            sla=sla,
        )

    def validate(self, run_id: str, workflow: Workflow, contract: RequirementContract) -> ValidationReport:
        """"Validation & Reconciliation" pipeline stage: generates the
        check manifest (Section 19) and actually runs it against the real
        artifacts this run produced.

        A check's `stage` is directly loadable when it names a real step
        id (that step's output) or a declared source ref like
        "source:orders" (used for pre-transform checks - see
        `ValidationRuleGenerator._reconciliation_checks`). A check
        generated at the "contract" or "release" stage (contract-level
        `null_policy`, `acceptance_tests`) has no single owning artifact,
        so it runs against the workflow's terminal step's output instead:
        the release candidate the check is really about. Every currently
        buildable plan (`WorkflowPlanner`'s deterministic double, and every
        hand-built test workflow) is a single linear chain, so "the last
        step in execution order" and "the terminal step" coincide; a
        genuinely branching DAG would need a real multi-sink design this
        codebase does not have yet.
        """
        run = self._run_store.get_run(run_id)
        if run is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no run with id '{run_id}'")

        rules = self._rule_generator.generate(contract=contract, workflow=workflow)

        directly_loadable_refs = {s.id for s in workflow.steps} | {f"{SOURCE_PREFIX}{s}" for s in workflow.sources}
        execution_order = workflow.execution_order()
        terminal_step_id = execution_order[-1] if execution_order else None

        dataframes: dict[str, pl.DataFrame] = {}
        for rule in rules:
            if rule.check_function is None or rule.stage in dataframes:
                continue
            source_step_id = rule.stage if rule.stage in directly_loadable_refs else terminal_step_id
            if source_step_id is None:
                continue
            if source_step_id not in dataframes:
                try:
                    dataframes[source_step_id] = self._artifact_store.load_by_ids(run_id, source_step_id)
                except PlatformError:
                    continue
            if rule.stage != source_step_id:
                dataframes[rule.stage] = dataframes[source_step_id]

        return self._rule_executor.execute(rules, dataframes)

    def reconcile(self, run_id: str, workflow: Workflow, contract: RequirementContract) -> ReconciliationReport:
        """The other half of the "Validation & Reconciliation" pipeline
        stage: Independent Reconciliation Agent (Section 20). Loads every
        artifact any step references (its own output, plus every input it
        declares) fresh from the artifact store - never trusting a step's
        own self-reported evidence dict - and hands them to
        `ReconciliationAgent`."""
        run = self._run_store.get_run(run_id)
        if run is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no run with id '{run_id}'")

        needed_refs = {ref for step in workflow.steps for ref in (*step.inputs, step.id)}
        artifacts: dict[str, pl.DataFrame] = {}
        for ref in needed_refs:
            try:
                artifacts[ref] = self._artifact_store.load_by_ids(run_id, ref)
            except PlatformError:
                continue

        return self._reconciliation_agent.reconcile(contract=contract, workflow=workflow, artifacts=artifacts)

    def start_run(
        self,
        workflow: Workflow,
        *,
        source_frames: dict[str, pl.DataFrame],
        source_versions: dict[str, DatasetVersion],
        contract: RequirementContract | None = None,
        granted_approvals: frozenset[str] = frozenset(),
        baseline_profiles: dict[str, DatasetProfile] | None = None,
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

        `baseline_profiles` is likewise optional: Blueprint 1.2 "No silent
        automation drift" only applies once a baseline exists to compare
        against - a source's first-ever run has none, and this codebase
        has no baseline store yet, so a caller (e.g. a recurring
        automation) that has kept its own baseline supplies it here to get
        the check enforced; the same `SchemaDriftMonitor` is exposed
        standalone via `check_drift()` for callers that only want the
        report, not the block.
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

        if baseline_profiles:
            drift_reports = self.check_drift(source_frames=source_frames, baseline_profiles=baseline_profiles)
            blocked = {name: r for name, r in drift_reports.items() if r.automation_action != "CONTINUE"}
            if blocked:
                raise PlatformError(
                    ErrorCode.SCHEMA_DRIFT,
                    "one or more sources drifted from their approved baseline",
                    evidence={name: report.model_dump() for name, report in blocked.items()},
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

    def start_automated_run(
        self,
        spec: AutomationSpec,
        workflow: Workflow,
        *,
        source_frames: dict[str, pl.DataFrame],
        source_versions: dict[str, DatasetVersion],
        baseline_profiles: dict[str, DatasetProfile],
        contract: RequirementContract | None = None,
        granted_approvals: frozenset[str] = frozenset(),
    ) -> str:
        """The recurring-run entry point Section 24 exists for: "Never
        auto-adapt to a material schema or semantic change. Drift must
        stop or quarantine the run until reviewed."

        Refuses outright - before anything else runs - if `workflow`'s own
        `workflow_version`, `requirement_contract_id`, or any step's
        `operation_version` no longer matches `spec.version_pins` exactly.
        A changed pin is exactly the "material change" this automation was
        built against; silently accepting a different workflow under the
        same `AutomationSpec` is precisely the auto-adaptation Section 24
        forbids - a human must rebuild the automation (call
        `build_automation()` again) instead.

        `baseline_profiles` is required here (unlike `start_run`'s own
        optional parameter): a scheduled, unattended run is exactly the
        case where schema-drift checking is not optional.
        """
        mismatches: list[str] = []
        if workflow.workflow_version != spec.version_pins.workflow_version:
            mismatches.append(
                f"workflow_version changed: pinned {spec.version_pins.workflow_version}, "
                f"got {workflow.workflow_version}"
            )
        if workflow.requirement_contract_id != spec.version_pins.requirement_contract_id:
            mismatches.append(
                f"requirement_contract_id changed: pinned '{spec.version_pins.requirement_contract_id}', "
                f"got '{workflow.requirement_contract_id}'"
            )
        for step in workflow.steps:
            pinned_version = spec.version_pins.operation_versions.get(step.operation_id)
            if pinned_version is not None and pinned_version != step.operation_version:
                mismatches.append(
                    f"operation '{step.operation_id}' version changed: pinned {pinned_version}, "
                    f"got {step.operation_version}"
                )

        if mismatches:
            raise PlatformError(
                ErrorCode.VALIDATION_FAIL,
                "automated run refused: workflow no longer matches the approved automation's version pins",
                evidence={"automation_id": spec.automation_id, "mismatches": mismatches},
            )

        return self.start_run(
            workflow,
            source_frames=source_frames,
            source_versions=source_versions,
            contract=contract,
            granted_approvals=granted_approvals,
            baseline_profiles=baseline_profiles,
        )

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

            if step.operation_id == "export":
                # Section 27 Security Guard, defense in depth ahead of
                # ExportOperation's own "'..'" precondition check: also
                # rejects an unapproved destination scheme, and defuses
                # CSV/Excel formula-injection payloads in the data itself
                # before it is ever written to a file a user might open.
                destination_path = step.params.get("destination_path", "") if isinstance(step.params, dict) else ""
                path_report = self._security_guard.scan_destination_path(str(destination_path))
                if path_report.decision == "BLOCK":
                    return self._fail_step_and_quarantine(
                        run_id,
                        step_id,
                        operation_full_id=step.full_operation_id,
                        reason=f"export destination blocked by the security guard: {destination_path!r}",
                        error_code=ErrorCode.POLICY_DENIED,
                        evidence={"flags": [f.model_dump() for f in path_report.flags]},
                    )
                input_df = self._security_guard.defuse_formula_injection(input_df)

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
        """Validation & Reconciliation (`validate()` + `reconcile()`) +
        Independent Verifier (Section 21) + Release/Quarantine Gate
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

        validation_report = self.validate(run_id, workflow, contract)
        reconciliation_report = self.reconcile(run_id, workflow, contract)
        step_runs = self._run_store.list_step_runs(run_id)
        verifier_result = self._verifier.verify(contract=contract, workflow=workflow, run=run, step_runs=step_runs)
        release_decision = self._release_gate.decide(
            contract=contract,
            run=run,
            verifier_result=verifier_result,
            validation_report=validation_report,
            reconciliation_report=reconciliation_report,
        )
        confidence_report = self._confidence_scorer.score(
            contract=contract,
            run=run,
            workflow=workflow,
            step_runs=step_runs,
            validation_report=validation_report,
            reconciliation_report=reconciliation_report,
            verifier_result=verifier_result,
            release_decision=release_decision,
        )

        if run.state in _RELEASE_TERMINAL_STATES:
            return ReleaseResult(
                run=run,
                validation_report=validation_report,
                reconciliation_report=reconciliation_report,
                verifier_result=verifier_result,
                release_decision=release_decision,
                confidence_report=confidence_report,
            )

        new_state = RunState.RELEASED if release_decision.decision == "RELEASE" else RunState.QUARANTINED
        run = self._run_store.update_run_state(run_id, new_state)

        return ReleaseResult(
            run=run,
            validation_report=validation_report,
            reconciliation_report=reconciliation_report,
            verifier_result=verifier_result,
            release_decision=release_decision,
            confidence_report=confidence_report,
        )

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
        sanitized_locations: list[str] = []
        masked_fields: list[str] = []
        for metric in contract.metrics:
            covering_steps = [s for s in workflow.steps if metric.name in s.requirement_refs]
            if not covering_steps:
                continue
            step = covering_steps[0]
            record = step_runs.get(step.id)
            if record is None or record.status != "COMPLETED":
                continue
            output_df = self._artifact_store.load_by_ids(run_id, step.id)
            sample_df = output_df.head(sample_rows)

            # Section 28 PII Classifier: applied first, before any other
            # processing, so a sensitive value is masked/dropped before
            # anything downstream (including the security scan below) has
            # a chance to see it - "applies privacy controls before data
            # is sent to models."
            pii_report = self._pii_classifier.classify_dataframe(sample_df)
            fields_needing_masking = [f.field for f in pii_report.fields if f.model_access != "ALLOW"]
            if fields_needing_masking:
                sample_df = self._pii_classifier.mask_for_model(sample_df, pii_report)
                masked_fields.extend(f"{step.id}.{field}" for field in fields_needing_masking)

            # Section 27 Security Guard: this is the only place actual
            # data cell values are placed into an LLM prompt in this
            # codebase - any value that reads like an instruction gets
            # wrapped in explicit untrusted-content delimiters
            # (Constitution 8.1) before it is ever sent, never dropped or
            # rewritten otherwise.
            security_report = self._security_guard.scan_dataframe(sample_df, location_prefix=f"{step.id}.")
            if security_report.decision != "ALLOW":
                flagged_columns = {loc.split(".", 1)[1] for loc in security_report.sanitized_refs}
                sample_df = self._security_guard.mark_untrusted_columns(sample_df, flagged_columns)
                sanitized_locations.extend(security_report.sanitized_refs)

            metrics_context.append(
                {
                    "name": metric.name,
                    "formula": metric.formula,
                    "step_id": step.id,
                    "sample_records": sample_df.to_dicts(),
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

        forced_limitations = list(verifier_result.unverified_claims)
        if masked_fields:
            forced_limitations.append(
                "some released fields were classified as sensitive and masked or withheld before being "
                f"sent to the explanation model: {masked_fields}"
            )
        if sanitized_locations:
            forced_limitations.append(
                "some released data values matched an instruction-like or code-payload pattern and were "
                f"marked as untrusted content before being sent to the explanation model: {sanitized_locations}"
            )

        return self._explainer.explain(
            run_id=run_id,
            evidence_context=evidence_context,
            known_evidence_refs=frozenset(known_refs),
            forced_limitations=forced_limitations,
            audience=audience,
        )

    def _fail_step_and_quarantine(
        self,
        run_id: str,
        step_id: str,
        *,
        operation_full_id: str,
        reason: str,
        error_code: ErrorCode = ErrorCode.VALIDATION_FAIL,
        evidence: dict | None = None,
    ) -> RunRecord:
        timestamp = now_iso()
        self._run_store.upsert_step_run(
            StepRunRecord(
                run_id=run_id,
                step_id=step_id,
                status="FAILED",
                operation_full_id=operation_full_id,
                error={"code": error_code.value, "reason": reason, "evidence": evidence or {}},
                started_at=timestamp,
                completed_at=timestamp,
            )
        )
        return self._run_store.update_run_state(run_id, RunState.QUARANTINED)
