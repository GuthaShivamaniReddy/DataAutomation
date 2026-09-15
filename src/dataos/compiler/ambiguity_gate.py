"""Ambiguity & Missing-Information Gate.

Source of truth: AI Prompt Library Section 4 "Ambiguity and
Missing-Information Gate": "Review the Requirement Contract and available
metadata. Your only job is to decide whether execution can safely
begin... Ask the fewest questions that fully resolve the blocking
uncertainty."

This is deterministic code, not a second LLM call - the checks below are
exactly the subset of Section 4's list that can be decided from the
RawExtraction, the SemanticResolver's results, and (optionally) a
dataset profile, without needing further model judgment. It exists as a
second, independent layer on top of RequirementContract's own validator
(Blueprint 1.2 "No guessing") - Global Constitution rule: "Never allow a
downstream agent to weaken these rules."
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.constitution import MATERIAL_TERMS
from dataos.compiler.extraction import RawExtraction
from dataos.compiler.semantic_resolver import ResolvedTerm
from dataos.contracts.requirement_contract import Clarification
from dataos.ingestion.profiling import DatasetProfile

_DATE_BUCKET_WORDS = {"day", "week", "month", "quarter", "year"}


def _normalize(term: str) -> str:
    return " ".join(term.strip().lower().replace("_", " ").split())


class GateResult(BaseModel):
    decision: Literal["PROCEED", "CLARIFY", "BLOCK"]
    clarifications: list[Clarification] = Field(default_factory=list)


class AmbiguityGate:
    def evaluate(
        self,
        *,
        extraction: RawExtraction,
        resolved_metrics: list[ResolvedTerm],
        sources: list[str],
        profile: DatasetProfile | None = None,
    ) -> GateResult:
        clarifications: list[Clarification] = []

        formula_by_term = {m.term: m.formula for m in extraction.metrics}

        for resolved in resolved_metrics:
            if resolved.status == "AMBIGUOUS":
                clarifications.append(
                    Clarification(
                        issue=f"metric term '{resolved.term}' matches multiple governed definitions",
                        why_material="the chosen definition changes the released numeric result",
                        options=resolved.candidate_ids,
                        question=(
                            f"Which governed definition should '{resolved.term}' use: "
                            f"{', '.join(resolved.candidate_ids)}?"
                        ),
                    )
                )
            elif resolved.status == "UNGOVERNED":
                has_explicit_formula = bool(formula_by_term.get(resolved.term))
                is_material_term = _normalize(resolved.term) in MATERIAL_TERMS
                if is_material_term or not has_explicit_formula:
                    clarifications.append(
                        Clarification(
                            issue=f"metric term '{resolved.term}' has no governed definition",
                            why_material=(
                                "terms like this have no accepted meaning without an explicit "
                                "or governed definition"
                                if is_material_term
                                else "an ungoverned metric requires an explicit formula before it can be computed"
                            ),
                            options=[],
                            question=f"What exact definition/formula should '{resolved.term}' use?",
                        )
                    )

        if not sources:
            clarifications.append(
                Clarification(
                    issue="no source dataset(s) specified",
                    why_material="the platform must never guess which uploaded/connected dataset to use",
                    options=[],
                    question="Which exact dataset(s)/snapshot(s) should this request run against?",
                )
            )

        date_bucket_terms = {t.lower() for t in (*extraction.group_by, *extraction.dimensions)} & _DATE_BUCKET_WORDS
        if date_bucket_terms and not extraction.timezone:
            clarifications.append(
                Clarification(
                    issue=f"date-bucketed grouping ({', '.join(sorted(date_bucket_terms))}) with no timezone specified",
                    why_material="the reporting timezone changes which calendar period a boundary timestamp falls into",
                    options=[],
                    question="What timezone/fiscal calendar should be used for date bucketing?",
                )
            )

        if profile is not None:
            column_names = [c.name for c in profile.columns]
            for resolved in resolved_metrics:
                if resolved.status != "UNGOVERNED":
                    continue
                term_norm = _normalize(resolved.term)
                candidate_columns = [c for c in column_names if term_norm in _normalize(c)]
                if len(candidate_columns) >= 2:
                    clarifications.append(
                        Clarification(
                            issue=f"multiple columns plausibly represent '{resolved.term}'",
                            why_material="using the wrong column silently changes the released result",
                            options=candidate_columns,
                            question=(
                                f"Which column is authoritative for '{resolved.term}': "
                                f"{', '.join(candidate_columns)}?"
                            ),
                        )
                    )

        decision = "BLOCK" if clarifications else "PROCEED"
        return GateResult(decision=decision, clarifications=clarifications)
