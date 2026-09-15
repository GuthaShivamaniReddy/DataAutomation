"""Workflow Planner orchestration.

Source of truth: AI Prompt Library Section 9 "Workflow Planner" and the
"Agent pipeline" ordering: ... -> Semantic Resolver -> Planner -> Policy /
Approval Gate -> ... This is the next real LLM-calling stage after
`RequirementCompiler` (Section 3): it turns an *approved*
`RequirementContract` into a typed `Workflow` DAG, using only operations
already present in the `OperationRegistry`.

Section 9's own "Mandatory stop / quarantine conditions" are enforced here
deterministically rather than trusted to prompt discipline, the same
pattern used throughout this package:
  - "A material business definition remains unresolved" -> refuse to plan
    against a contract that is not `ready_for_planning` (Blueprint 1.2
    "No guessing"; this is also Section 4's Ambiguity Gate's job already
    done upstream - the Planner must not re-decide it).
  - "A required step cannot be represented with an approved operation/tool"
    -> every proposed step's `operation_id@operation_version` is resolved
    against the real `OperationRegistry` (this is also Section 10
    "Operation Registry Selector"'s rule: "You are not allowed to invent
    an unregistered operation... If no registered operation satisfies the
    step, return NO_SAFE_OPERATION").
The resulting `Workflow` is additionally run through `validate_dag()`
(structural cycle/duplicate-id/unknown-reference checks) before being
returned, so a malformed plan can never reach the orchestrator.
"""

from __future__ import annotations

import json
import uuid

from pydantic import BaseModel

from dataos.compiler.constitution import GLOBAL_CONSTITUTION, AgentEnvelope
from dataos.compiler.prompts import WORKFLOW_PLANNER_SYSTEM_PROMPT
from dataos.compiler.raw_plan import RawPlan
from dataos.contracts.requirement_contract import RequirementContract
from dataos.errors import ErrorCode, PlatformError
from dataos.llm.client import LLMClient
from dataos.registry.registry import OperationRegistry
from dataos.workflow.dsl import ReleasePolicy, Workflow, WorkflowStep

WORKFLOW_PLANNER_PROMPT_VERSION = "1.0"


class PlannerOutput(BaseModel):
    workflow: Workflow
    envelope: AgentEnvelope


class WorkflowPlanner:
    def __init__(self, llm_client: LLMClient, registry: OperationRegistry) -> None:
        self._llm_client = llm_client
        self._registry = registry

    def plan(
        self,
        *,
        contract: RequirementContract,
        contract_id: str,
        workflow_version: int = 1,
    ) -> PlannerOutput:
        if not contract.ready_for_planning:
            raise PlatformError(
                ErrorCode.VALIDATION_FAIL,
                "cannot plan against a requirement contract that is not ready_for_planning",
                evidence={"contract_id": contract_id, "blocking_reasons": contract.blocking_reasons},
            )

        user_prompt = _render_planning_context(contract, self._registry)

        raw_plan = self._llm_client.complete_structured(
            system_prompt=f"{GLOBAL_CONSTITUTION}\n\n{WORKFLOW_PLANNER_SYSTEM_PROMPT}",
            user_prompt=user_prompt,
            response_model=RawPlan,
        )

        unresolved = []
        steps: list[WorkflowStep] = []
        for raw_step in raw_plan.steps:
            try:
                self._registry.get(raw_step.operation_id, raw_step.operation_version)
            except PlatformError:
                unresolved.append({"step_id": raw_step.id, "operation_id": raw_step.operation_id, "operation_version": raw_step.operation_version})
                continue
            steps.append(
                WorkflowStep(
                    id=raw_step.id,
                    operation_id=raw_step.operation_id,
                    operation_version=raw_step.operation_version,
                    inputs=raw_step.inputs,
                    params=raw_step.params,
                    on_failure=raw_step.on_failure,
                    requirement_refs=raw_step.requirement_refs,
                )
            )

        if unresolved:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                "plan references operation(s) not present in the registry",
                evidence={"unresolved_steps": unresolved, "available_operations": self._registry.list_operations()},
            )

        workflow = Workflow(
            workflow_version=workflow_version,
            requirement_contract_id=contract_id,
            sources=[s.name for s in contract.sources],
            steps=steps,
            release_policy=ReleasePolicy(),
            final_acceptance_tests=raw_plan.final_acceptance_tests or list(contract.acceptance_tests),
            approval_gates=raw_plan.approval_gates,
        )
        workflow.validate_dag()

        envelope = AgentEnvelope(
            run_id=str(uuid.uuid4()),
            agent="workflow_planner",
            agent_prompt_version=WORKFLOW_PLANNER_PROMPT_VERSION,
            model_id=getattr(self._llm_client, "model_id", type(self._llm_client).__name__),
            requirement_contract_id=contract_id,
            status="OK",
            result={"plan_id": raw_plan.plan_id, "step_types": {s.id: s.type for s in raw_plan.steps}},
            next_action="await policy/approval gate before execution",
        )

        return PlannerOutput(workflow=workflow, envelope=envelope)


def _render_planning_context(contract: RequirementContract, registry: OperationRegistry) -> str:
    """Runtime context discipline (Prompt Library Section 38): send only
    the fields the Planner actually needs, as compact structured JSON -
    never the full contract's audit-trail fields (assumptions,
    clarifications, status) and never raw source data."""
    context = {
        "objective": contract.objective,
        "sources": [s.name for s in contract.sources],
        "grain": contract.grain,
        "population": contract.population,
        "metrics": [
            {"name": m.name, "formula": m.formula, "source_fields": m.source_fields} for m in contract.metrics
        ],
        "dimensions": contract.dimensions,
        "filters": contract.filters,
        "group_by": contract.group_by,
        "time": contract.time.model_dump(),
        "units": contract.units.model_dump(),
        "cleaning_rules": contract.cleaning_rules,
        "outputs": contract.outputs,
        "acceptance_tests": contract.acceptance_tests,
        "available_operations": registry.list_operations(),
    }
    return json.dumps(context)
