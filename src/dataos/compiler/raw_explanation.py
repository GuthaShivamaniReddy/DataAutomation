"""The structured-output schema requested from the LLM by the Report &
Explanation Agent.

Source of truth: AI Prompt Library Section 18 "Report & Explanation
Agent" - "Required output contract". Unlike `RawExtraction` ->
`RequirementContract` or `RawPlan` -> `Workflow`, there is no further
deterministic assembly step here: the explanation itself is the final
deliverable. What deterministic code adds instead is validation - see
`ExplanationAgent`, which rejects any finding whose `evidence_refs` cite
something the model was never actually given.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    statement: str
    type: Literal["FACT", "INTERPRETATION", "ESTIMATE"]
    evidence_refs: list[str] = Field(default_factory=list)


class RawExplanation(BaseModel):
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    method_note: str = ""
