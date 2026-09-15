"""Requirement Contract schema.

Source of truth:
  - Reliability-First Master Blueprint, Section 3.1 "Requirement Contract"
    and Appendix A "Requirement Contract Field Checklist".
  - AI Prompt Library, Section 3 "Requirement Compiler" and Appendix A
    "Reusable Requirement Contract Template".

The contract is the authoritative, machine-checkable definition of what
"correct" means for a run (Blueprint 3.1). No planning or execution may
begin against a contract that is not ready for planning.

The central non-negotiable rule encoded here (Blueprint 1.2 "No guessing",
Prompt Library Section 3 "Ambiguity policy") is:

    Only EXPLICIT, GOVERNED, and safely INFERRED_UNIQUE items may proceed
    automatically. AMBIGUOUS or MISSING material items must create
    clarification questions and block execution.

This is enforced structurally below, not left to prompt discipline.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class DefinitionStatus(str, Enum):
    """Confidence classification for a single requirement field.

    Source: AI Prompt Library, Section 3 "Ambiguity policy".
    """

    EXPLICIT = "EXPLICIT"
    GOVERNED = "GOVERNED"
    INFERRED_UNIQUE = "INFERRED_UNIQUE"
    AMBIGUOUS = "AMBIGUOUS"
    MISSING = "MISSING"

    @property
    def is_blocking(self) -> bool:
        return self in (DefinitionStatus.AMBIGUOUS, DefinitionStatus.MISSING)


class RequirementStatus(str, Enum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"


class Source(BaseModel):
    """A required input dataset/system for this contract."""

    name: str
    required: bool = True
    snapshot_id: str | None = None
    """Bound to an immutable DatasetVersion once ingestion has run."""


class Metric(BaseModel):
    """A single requested or governed metric definition.

    Every metric must declare its own confidence status. A contract cannot
    be marked ready for planning while any metric's status is AMBIGUOUS or
    MISSING (see RequirementContract's model validator below).
    """

    name: str
    definition: str | None = None
    formula: str | None = None
    source_fields: list[str] = Field(default_factory=list)
    definition_status: DefinitionStatus = DefinitionStatus.MISSING


class TimeSpec(BaseModel):
    range: str | None = None
    timezone: str | None = None
    comparison: str | None = None
    fiscal_calendar: str | None = None


class UnitsPolicy(BaseModel):
    currency: str | None = None
    measurement: str | None = None


class Clarification(BaseModel):
    """A concrete blocking question raised for the user (Blueprint 3.2)."""

    issue: str
    why_material: str
    options: list[str] = Field(default_factory=list)
    question: str


class RequirementContract(BaseModel):
    contract_version: int = 1
    objective: str

    sources: list[Source] = Field(default_factory=list)
    authoritative_semantics: dict[str, str] = Field(default_factory=dict)
    """Concept -> exact formula/definition text, e.g. {"revenue": "sum(orders.net_amount) - sum(refunds.amount)"}."""

    grain: str | None = None
    population: str | None = None

    metrics: list[Metric] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)

    time: TimeSpec = Field(default_factory=TimeSpec)
    units: UnitsPolicy = Field(default_factory=UnitsPolicy)
    tolerances: dict[str, float] = Field(default_factory=dict)
    null_policy: dict[str, str] = Field(default_factory=dict)
    """Field -> "error" | "ignore" | "fill:<value>" | domain-specific rule. No implicit default."""

    cleaning_rules: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    destination: str | None = None
    schedule: str | None = None

    side_effects: list[str] = Field(default_factory=list)
    required_approvals: list[str] = Field(default_factory=list)
    prohibited_actions: list[str] = Field(default_factory=list)

    acceptance_tests: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)

    status: RequirementStatus = RequirementStatus.DRAFT
    ready_for_planning: bool = False

    @model_validator(mode="after")
    def _enforce_no_guessing(self) -> "RequirementContract":
        """Blueprint 1.2 "No guessing" as code, not as prompt discipline.

        Any AMBIGUOUS/MISSING material metric, or any unresolved
        Clarification, forces the contract to BLOCKED / not-ready
        regardless of what the caller passed in. A contract can only
        reach ready_for_planning=True by actually resolving every
        blocking item (removing/upgrading it), never by asserting the
        flag directly.
        """
        blocking_metrics = [m.name for m in self.metrics if m.definition_status.is_blocking]
        has_open_clarifications = len(self.clarifications) > 0
        has_no_objective_sources = len(self.sources) == 0

        is_blocked = bool(blocking_metrics) or has_open_clarifications or has_no_objective_sources

        if is_blocked:
            object.__setattr__(self, "status", RequirementStatus.BLOCKED)
            object.__setattr__(self, "ready_for_planning", False)
        elif self.status == RequirementStatus.BLOCKED:
            # Previously blocked but no blocking condition remains: leave
            # status as-is (caller/compiler must explicitly re-approve),
            # but never silently flip ready_for_planning true on our own.
            object.__setattr__(self, "ready_for_planning", False)
        else:
            object.__setattr__(
                self,
                "ready_for_planning",
                self.status == RequirementStatus.APPROVED,
            )

        return self

    @property
    def blocking_reasons(self) -> list[str]:
        reasons = [
            f"metric '{m.name}' has definition_status={m.definition_status.value}"
            for m in self.metrics
            if m.definition_status.is_blocking
        ]
        reasons += [f"unresolved clarification: {c.question}" for c in self.clarifications]
        if not self.sources:
            reasons.append("no sources declared")
        return reasons
