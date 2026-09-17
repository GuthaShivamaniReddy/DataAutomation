"""Schema Mapping Agent.

Source of truth: AI Prompt Library Section 7 "Schema Mapping Agent":
"Given a Requirement Contract, governed semantics, schemas, and profiling
evidence, map required concepts to concrete fields... Prefer explicit
metadata and governed semantic mappings over name similarity. Do not map
two different business concepts to the same column unless the semantic
layer explicitly allows it... If a required concept has no unambiguous
source, mark it BLOCKING."

Deterministic code, not an LLM call - same reasoning as every other
gate/generator in this package: whether a declared source field ("dataset.
column") or a plain concept name resolves to a real, unambiguous column is
a provable fact about `DatasetProfile` evidence and the governed
`SemanticDictionary`, not a judgment call.

This sits between the Semantic Resolver (Section 5, which only decides
whether a *term* has a governed formula) and the Workflow Planner
(Section 9, which needs concrete columns to build steps against). A
`Metric.source_fields`/`SemanticDimension.source_field` entry is only ever
trusted metadata (correctness hierarchy tiers B/C); this agent's job is to
prove that metadata still points at a real column in the profiled data
before anything downstream builds on it - it never invents a mapping that
metadata and profiling evidence do not already support.

Two structural checks beyond plain existence:
  - a resolved column whose *concept* name reads as a monetary amount
    (Section 7: "For money, identify amount field... decimal scale") but
    whose profiled dtype is not numeric is not usable as a money field -
    that downgrades the mapping to MISSING rather than reporting a mapping
    that cannot actually be computed against.
  - two distinct concepts resolving to the exact same (dataset, column)
    pair is exactly Section 7's "do not map two different business
    concepts to the same column" case; this codebase has no semantic-layer
    "explicitly allows it" exception table, so it is never assumed and
    always surfaces as a blocking item.

Identifier null-characteristics and per-field date-role labeling are
informational evidence only (never a blocking mapping status here):
whether a null identifier or an unclassified date column is *acceptable*
is a data-quality judgment against the declared grain, which is the Data
Quality Assessor's job (Section 8), not this agent's.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import SCHEMA_MAPPING_AGENT_SYSTEM_PROMPT
from dataos.contracts.requirement_contract import RequirementContract
from dataos.ingestion.profiling import ColumnProfile, DatasetProfile
from dataos.llm.client import LLMClient
from dataos.semantics.dictionary import SemanticDictionary

MappingStatus = Literal["MAPPED", "AMBIGUOUS", "MISSING"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]

_MONEY_TERM_PATTERN = re.compile(
    r"(revenue|amount|sales|price|cost|total|balance|payment|spend|margin|profit)", re.I
)
_NUMERIC_DTYPE_PATTERN = re.compile(r"(Int|UInt|Float|Decimal)")
_IDENTIFIER_SUFFIX_PATTERN = re.compile(r"(^id$|_id$)", re.I)
_DATE_DTYPE_PATTERN = re.compile("Date")

# Checked in this order; the first match wins. Purely descriptive labeling
# of an already-resolved date/datetime column - see module docstring.
_TIME_ROLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("created_time", re.compile(r"created|create[_-]?date", re.I)),
    ("updated_time", re.compile(r"updated|modified", re.I)),
    ("posting_time", re.compile(r"posting|posted|book(ing)?[_-]?date", re.I)),
    ("effective_time", re.compile(r"effective", re.I)),
    ("event_time", re.compile(r"event|order[_-]?date|ship[_-]?date|transaction[_-]?date|occurred", re.I)),
]


class SchemaMapping(BaseModel):
    concept: str
    dataset: str | None = None
    column: str | None = None
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)
    status: MappingStatus


class MappingBlockingItem(BaseModel):
    concept: str
    reason: str
    candidates: list[str] = Field(default_factory=list)


class SchemaMappingResult(BaseModel):
    mappings: list[SchemaMapping] = Field(default_factory=list)
    blocking_items: list[MappingBlockingItem] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `mappings`/`blocking_items` remain
    the decision."""


class SchemaMappingAgent:
    def __init__(
        self,
        dictionary: SemanticDictionary | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._dictionary = dictionary
        self._llm_client = llm_client

    def map(
        self,
        *,
        contract: RequirementContract,
        profiles: dict[str, DatasetProfile],
    ) -> SchemaMappingResult:
        mappings: list[SchemaMapping] = []

        for metric in contract.metrics:
            if metric.definition_status.is_blocking or not metric.source_fields:
                # Already blocked upstream (Ambiguity Gate / contract
                # validator), or nothing structured to map yet (an
                # EXPLICIT metric's free-form formula has no declared
                # source_fields) - not this agent's concern either way.
                continue
            for raw_field in metric.source_fields:
                mappings.append(self._resolve_field(metric.name, raw_field, profiles))

        for term in (*contract.dimensions, *contract.group_by):
            mappings.append(self._resolve_dimension(term, profiles))

        blocking_items = self._blocking_items(mappings)

        narrative_text = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=SCHEMA_MAPPING_AGENT_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {
                            "mapped_count": sum(1 for m in mappings if m.status == "MAPPED"),
                            "unmapped_count": sum(1 for m in mappings if m.status != "MAPPED"),
                            "blocking_items": [b.model_dump() for b in blocking_items],
                        },
                    }
                ],
            )
            narrative_text = narratives.get("summary")

        return SchemaMappingResult(mappings=mappings, blocking_items=blocking_items, narrative=narrative_text)

    def _resolve_field(self, concept: str, raw_field: str, profiles: dict[str, DatasetProfile]) -> SchemaMapping:
        if "." in raw_field:
            dataset, column = raw_field.split(".", 1)
            profile = profiles.get(dataset)
            if profile is None:
                return SchemaMapping(
                    concept=concept, dataset=dataset, column=column, confidence="LOW", status="MISSING",
                    evidence=[f"dataset '{dataset}' is not among the profiled sources"],
                )
            col = profile.column(column)
            if col is None:
                return SchemaMapping(
                    concept=concept, dataset=dataset, column=column, confidence="LOW", status="MISSING",
                    evidence=[f"column '{column}' does not exist in dataset '{dataset}'"],
                )
            evidence = [f"governed source field '{raw_field}' confirmed by profile: dtype={col.dtype}, null_rate={col.null_rate:.3f}"]
            return self._finalize(concept, dataset, column, col, confidence="HIGH", evidence=evidence)

        hits = [(name, p.column(raw_field)) for name, p in profiles.items() if p.column(raw_field) is not None]
        if not hits:
            return SchemaMapping(
                concept=concept, column=raw_field, confidence="LOW", status="MISSING",
                evidence=[f"no profiled dataset contains a column named '{raw_field}'"],
            )
        if len(hits) > 1:
            return SchemaMapping(
                concept=concept, column=raw_field, confidence="LOW", status="AMBIGUOUS",
                evidence=[f"column '{raw_field}' exists in multiple datasets: {[n for n, _ in hits]}"],
            )
        dataset, col = hits[0]
        assert col is not None
        evidence = [f"column-name match in dataset '{dataset}': dtype={col.dtype}, null_rate={col.null_rate:.3f}"]
        return self._finalize(concept, dataset, raw_field, col, confidence="MEDIUM", evidence=evidence)

    def _resolve_dimension(self, term: str, profiles: dict[str, DatasetProfile]) -> SchemaMapping:
        if self._dictionary is not None:
            candidates = self._dictionary.find_dimension_candidates(term)
            if len(candidates) > 1:
                return SchemaMapping(
                    concept=term, confidence="LOW", status="AMBIGUOUS",
                    evidence=[f"multiple governed dimensions match '{term}': {[c.semantic_id for c in candidates]}"],
                )
            if len(candidates) == 1:
                dim = candidates[0]
                mapping = self._resolve_field(term, dim.source_field, profiles)
                mapping.evidence.insert(0, f"governed dimension '{dim.semantic_id}' declares source_field '{dim.source_field}'")
                if mapping.status == "MAPPED":
                    mapping.confidence = "HIGH"
                return mapping
        return self._resolve_field(term, term, profiles)

    def _finalize(
        self,
        concept: str,
        dataset: str,
        column: str,
        col: ColumnProfile,
        *,
        confidence: Confidence,
        evidence: list[str],
    ) -> SchemaMapping:
        """A resolved (dataset, column) pair still has to satisfy the
        structural shape its concept implies - Section 7's money rule is
        the one case that is provable from the profile alone (see module
        docstring); everything else is descriptive-only evidence."""
        if _MONEY_TERM_PATTERN.search(concept) and not _NUMERIC_DTYPE_PATTERN.search(col.dtype):
            return SchemaMapping(
                concept=concept, dataset=dataset, column=column, confidence="LOW", status="MISSING",
                evidence=[
                    *evidence,
                    f"'{concept}' reads as a monetary concept but resolves to non-numeric dtype '{col.dtype}' - not usable as an amount field",
                ],
            )

        if (_IDENTIFIER_SUFFIX_PATTERN.search(concept) or _IDENTIFIER_SUFFIX_PATTERN.search(column)) and col.null_count > 0:
            evidence.append(
                f"identifier-shaped column has {col.null_count} null value(s) out of its profiled rows "
                "- verify null-handling against the declared grain (Data Quality Assessor)"
            )

        if _DATE_DTYPE_PATTERN.search(col.dtype):
            for role, pattern in _TIME_ROLE_PATTERNS:
                if pattern.search(column):
                    evidence.append(f"time_role={role}")
                    break

        return SchemaMapping(concept=concept, dataset=dataset, column=column, confidence=confidence, evidence=evidence, status="MAPPED")

    def _blocking_items(self, mappings: list[SchemaMapping]) -> list[MappingBlockingItem]:
        items: list[MappingBlockingItem] = []
        for m in mappings:
            if m.status == "MISSING":
                items.append(
                    MappingBlockingItem(
                        concept=m.concept,
                        reason=f"no unambiguous source field found for '{m.concept}': {'; '.join(m.evidence)}",
                    )
                )
            elif m.status == "AMBIGUOUS":
                items.append(
                    MappingBlockingItem(
                        concept=m.concept,
                        reason=f"'{m.concept}' matches more than one possible source",
                        candidates=list(m.evidence),
                    )
                )

        by_column: dict[tuple[str, str], set[str]] = {}
        for m in mappings:
            if m.status == "MAPPED" and m.dataset and m.column:
                by_column.setdefault((m.dataset, m.column), set()).add(m.concept)
        for (dataset, column), concepts in by_column.items():
            if len(concepts) > 1:
                items.append(
                    MappingBlockingItem(
                        concept="/".join(sorted(concepts)),
                        reason=(
                            f"column '{dataset}.{column}' is mapped to multiple distinct concepts "
                            f"{sorted(concepts)} without semantic-layer confirmation that this is intentional"
                        ),
                        candidates=sorted(concepts),
                    )
                )

        return items
