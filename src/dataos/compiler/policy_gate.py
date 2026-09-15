"""Policy / Approval Gate.

Source of truth: AI Prompt Library Section 1 "Global AI Constitution" rule
7 ("Never approve an external write, deletion, payment, payroll action,
customer update, regulatory filing, or irreversible side effect without
the required authorization gate"), Section 2 "Master Orchestrator"
procedure step 6 ("Apply policy gates. Any destructive, regulated,
financial, privacy-sensitive, or external-write step must have the
required approval token") and its mandatory stop condition "A high-risk
action lacks approval", and Section 38's "External write" prompt-chaining
pattern: "Requirement Compiler -> Planner -> Policy Gate -> Dry-run/Preview
-> User/Policy Approval -> Write Tool -> Post-write Reconciliation ->
Release/Receipt".

This is deterministic code, not an LLM call, for the same reason
`AmbiguityGate` is: Section 8.1 "Tool permissions are enforced outside the
model" and the Global Constitution's Correctness Hierarchy explicitly
ranks "Model interpretation / suggestion" (F) below deterministic results
- an authorization decision must never depend on a model's judgment call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.contracts.requirement_contract import RequirementContract
from dataos.workflow.dsl import Workflow

# AI Prompt Library Section 7's operation table + Section 29 "Connector /
# External Write Planner": "Default to read-only. Any write must reference
# an approved workflow step and authorization token." The only Phase 1
# registry operation that performs an external side effect (writing to a
# destination path) is `export` - every other registered operation is a
# pure in-memory dataframe transform with no side effect to gate.
DEFAULT_APPROVAL_REQUIRED_OPERATIONS: frozenset[str] = frozenset({"export"})


class PolicyDecision(BaseModel):
    decision: Literal["PROCEED", "BLOCKED"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    required_approvals: list[str] = Field(default_factory=list)
    """Every approval token this workflow needs, granted or not - callers
    can present this list to a human/policy approver."""
    blocking_reasons: list[str] = Field(default_factory=list)


class PolicyGate:
    def __init__(
        self,
        approval_required_operations: frozenset[str] = DEFAULT_APPROVAL_REQUIRED_OPERATIONS,
    ) -> None:
        self._approval_required_operations = approval_required_operations

    def evaluate(
        self,
        *,
        workflow: Workflow,
        contract: RequirementContract | None = None,
        granted_approvals: frozenset[str] = frozenset(),
    ) -> PolicyDecision:
        """`contract` is optional: callers that only hold a typed `Workflow`
        (e.g. `WorkflowOrchestrator.start_run` with no requirement contract
        in scope) still get the operation-risk and `workflow.approval_gates`
        checks below - just without the contract-level
        `required_approvals`/`side_effects`/`prohibited_actions` checks,
        which need the contract to exist."""
        contract_required_approvals = set(contract.required_approvals) if contract else set()
        contract_side_effects = set(contract.side_effects) if contract else set()
        contract_prohibited_actions = set(contract.prohibited_actions) if contract else set()

        blocking_reasons: list[str] = []

        # Constitution rule 7 / Blueprint "No destructive default": a
        # prohibited action can never be approved away - it always blocks,
        # regardless of what `granted_approvals` contains.
        gate_names = (
            contract_required_approvals
            | contract_side_effects
            | set(workflow.approval_gates)
            | {s.operation_id for s in workflow.steps}
        )
        prohibited_hits = sorted(a for a in contract_prohibited_actions if a in gate_names)
        for hit in prohibited_hits:
            blocking_reasons.append(f"'{hit}' is a prohibited_action on this requirement contract")

        required: set[str] = set()
        for step in workflow.steps:
            if step.operation_id in self._approval_required_operations:
                required.add(step.operation_id)
        required |= contract_required_approvals
        # A declared side effect with no separately-named approval still
        # needs *some* explicit grant - the side-effect text itself is its
        # own required-approval token so it can never be silently skipped.
        required |= contract_side_effects
        required |= set(workflow.approval_gates)

        missing = sorted(required - granted_approvals)
        for name in missing:
            blocking_reasons.append(f"required approval not granted: '{name}'")

        if prohibited_hits:
            risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "CRITICAL"
        elif required:
            risk_level = "HIGH"
        elif contract_side_effects:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        decision: Literal["PROCEED", "BLOCKED"] = "BLOCKED" if blocking_reasons else "PROCEED"
        return PolicyDecision(
            decision=decision,
            risk_level=risk_level,
            required_approvals=sorted(required),
            blocking_reasons=blocking_reasons,
        )
