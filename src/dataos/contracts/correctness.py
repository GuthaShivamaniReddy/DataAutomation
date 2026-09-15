"""Correctness classes and their minimum release evidence.

Source of truth: Reliability-First Master Blueprint, Section 1.3 "Correctness
classes". This table is what the (future) validation and release-gate phases
consult to decide whether a run has produced enough evidence to be released.
Nothing here is optional or negotiable per-run - the evidence requirement is
fixed by the correctness class of the request.
"""

from __future__ import annotations

from enum import Enum


class CorrectnessClass(str, Enum):
    C0_INFORMATIONAL = "C0"
    """profile, counts, schema overview"""

    C1_TRANSFORMATIONAL = "C1"
    """clean, map, join, aggregate"""

    C2_ANALYTICAL = "C2"
    """KPIs, variance, cohorts, statistics"""

    C3_PREDICTIVE = "C3"
    """forecast, classification, clustering"""

    C4_AUTHORITATIVE = "C4"
    """posting, payroll, regulatory, production writes"""


# Minimum release evidence per class, per Blueprint Section 1.3. Each class's
# list is *additive* on top of the classes before it (C2 requires everything
# C1 requires, plus its own additions, etc.) - REQUIRED_EVIDENCE already
# expands that so each entry is the full, non-relative list.
REQUIRED_EVIDENCE: dict[CorrectnessClass, list[str]] = {
    CorrectnessClass.C0_INFORMATIONAL: [
        "parse_success",
        "source_hash",
        "schema_evidence",
    ],
    CorrectnessClass.C1_TRANSFORMATIONAL: [
        "parse_success",
        "source_hash",
        "schema_evidence",
        "input_output_invariants",
        "row_impact_report",
        "key_checks",
    ],
    CorrectnessClass.C2_ANALYTICAL: [
        "parse_success",
        "source_hash",
        "schema_evidence",
        "input_output_invariants",
        "row_impact_report",
        "key_checks",
        "metric_definitions",
        "independent_spot_or_aggregate_checks",
    ],
    CorrectnessClass.C3_PREDICTIVE: [
        "parse_success",
        "source_hash",
        "schema_evidence",
        "input_output_invariants",
        "row_impact_report",
        "key_checks",
        "metric_definitions",
        "independent_spot_or_aggregate_checks",
        "suitability_checks",
        "holdout_or_backtest",
        "uncertainty",
    ],
    CorrectnessClass.C4_AUTHORITATIVE: [
        "parse_success",
        "source_hash",
        "schema_evidence",
        "input_output_invariants",
        "row_impact_report",
        "key_checks",
        "metric_definitions",
        "independent_spot_or_aggregate_checks",
        "suitability_checks",
        "holdout_or_backtest",
        "uncertainty",
        "reconciliation",
        "human_approval",
        "rollback_plan",
        "domain_controls",
    ],
}


def required_evidence_for(correctness_class: CorrectnessClass) -> list[str]:
    """Return the exact evidence keys that must be present before release."""
    return REQUIRED_EVIDENCE[correctness_class]
