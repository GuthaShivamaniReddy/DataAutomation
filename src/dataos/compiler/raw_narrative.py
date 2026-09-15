"""Shared structured-output schema for the optional LLM narrative layer
on otherwise-deterministic gates/generators/agents in this package
(`ValidationRuleGenerator`, `ReconciliationAgent`, `IndependentVerifier`,
`ConfidenceScorer`, `ReleaseGate`, `SchemaDriftMonitor`).

Every one of those components keeps its actual decision (a check's
severity, a verdict, a classification, a pass/fail, a release/quarantine
call) entirely deterministic, computed before this schema is ever used.
An LLM, called only when the component was explicitly constructed with an
`LLMClient`, may supply human-readable prose *in addition to* that
decision - never in place of it. This is Section 8.1 "Tool permissions
are enforced outside the model" applied uniformly to every narrative call
site in this package, rather than re-litigated per component.

`items` is a list rather than a single string so one call can narrate
several already-computed facts at once (e.g. `ValidationRuleGenerator`
narrating several `CheckSpec`s); a component that only ever needs one
narrative uses a single item with `ref="summary"`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RawNarrativeItem(BaseModel):
    ref: str
    narrative: str


class RawNarrative(BaseModel):
    items: list[RawNarrativeItem] = Field(default_factory=list)
