"""Business Semantic Resolver.

Source of truth: AI Prompt Library Section 5 "Business Semantic
Resolver": "Resolve user terms against the organization semantic layer.
Do not create new business definitions silently... Never substitute
gross sales for net revenue, bookings for recognized revenue, users for
accounts, events for unique users, or similar near-synonyms without
evidence."

This is pure dictionary lookup, not an LLM call - "evidence" here means
an actual alias match in the governed SemanticDictionary, never a
plausible-sounding guess.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.semantics.dictionary import SemanticDictionary
from dataos.semantics.models import SemanticMetric


class ResolvedTerm(BaseModel):
    term: str
    status: Literal["GOVERNED", "AMBIGUOUS", "UNGOVERNED"]
    metric: SemanticMetric | None = None
    """Set only when status == GOVERNED."""
    candidate_ids: list[str] = Field(default_factory=list)
    """Set only when status == AMBIGUOUS - the competing semantic_ids, never auto-picked."""


class SemanticResolver:
    def __init__(self, dictionary: SemanticDictionary) -> None:
        self._dictionary = dictionary

    def resolve_metric(self, term: str) -> ResolvedTerm:
        candidates = self._dictionary.find_metric_candidates(term)

        if len(candidates) == 1:
            return ResolvedTerm(term=term, status="GOVERNED", metric=candidates[0])
        if len(candidates) > 1:
            return ResolvedTerm(
                term=term,
                status="AMBIGUOUS",
                candidate_ids=[c.semantic_id for c in candidates],
            )
        return ResolvedTerm(term=term, status="UNGOVERNED")
