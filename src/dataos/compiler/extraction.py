"""The structured-output schema requested from the LLM by the Requirement
Compiler.

Source of truth: AI Prompt Library Section 3 "Requirement Compiler" -
"FOR EACH REQUEST EXTRACT" list. This is deliberately narrower than the
full RequirementContract: the model's job is raw extraction only ("Do not
plan implementation yet"). Everything else (semantic governance, ambiguity
blocking, final contract assembly) happens in deterministic code that
this extraction feeds into.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedMetric(BaseModel):
    term: str
    formula: str | None = None
    """Only set when the user's own request text stated an explicit formula/definition."""


class RawExtraction(BaseModel):
    objective: str
    metrics: list[ExtractedMetric] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    time_range: str | None = None
    timezone: str | None = None
