"""Semantic layer object models.

Source of truth: Reliability-First Master Blueprint Section 6 "Semantic
Layer and Business-Rule Management": "It stores definitions for metrics,
dimensions, entities, calendars, currencies, identifiers, valid values,
ownership, and authoritative sources. Definitions are versioned and can
require approval."
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SemanticMetric(BaseModel):
    semantic_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    formula: str
    grain: str | None = None
    filters: list[str] = Field(default_factory=list)
    source_fields: list[str] = Field(default_factory=list)
    currency: str | None = None
    rounding: str | None = None
    owner: str | None = None
    version: int = 1
    effective_from: str | None = None

    def matches(self, term: str) -> bool:
        normalized = _normalize(term)
        candidates = {_normalize(self.name), *[_normalize(a) for a in self.aliases]}
        return normalized in candidates


class SemanticDimension(BaseModel):
    semantic_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    source_field: str
    allowed_values: list[str] | None = None
    hierarchy: list[str] = Field(default_factory=list)

    def matches(self, term: str) -> bool:
        normalized = _normalize(term)
        candidates = {_normalize(self.name), *[_normalize(a) for a in self.aliases]}
        return normalized in candidates


class SemanticEntity(BaseModel):
    semantic_id: str
    name: str
    key_fields: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class SemanticCalendar(BaseModel):
    timezone: str
    week_start: str = "Monday"
    fiscal_year_start_month: int = 1


def _normalize(term: str) -> str:
    return " ".join(term.strip().lower().replace("_", " ").split())
