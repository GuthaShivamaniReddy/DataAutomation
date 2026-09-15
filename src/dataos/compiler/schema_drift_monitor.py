"""Schema and Semantic Drift Monitor.

Source of truth: AI Prompt Library Section 25 "Schema Drift Monitor":
"Compare the current run's source metadata and governed definitions with
the last approved baseline... Classify each change as COMPATIBLE,
REVIEW_REQUIRED, or BREAKING. Never automatically remap a missing or
renamed material field solely by similarity." Also Blueprint Section 14
"No silent automation drift: Recurring workflows compare schema,
distributions, semantics, and rules to the approved baseline before
running."

Deterministic code, not an LLM call - same reasoning as every other
gate/generator/agent in this package: comparing two already-computed,
already-typed `DatasetProfile`/`SemanticDictionary` snapshots is
mechanical, and "never remap a renamed field solely by similarity" is
best enforced by construction - this module never tries to match a
removed column to an added one at all; it reports both as independent
changes.

Section 25's list also covers "enum/domain changes" and "source version
changes" - `DatasetProfile` has no enum/domain field and this codebase has
no separate source-version identifier distinct from the profile itself,
so this monitor produces nothing for either rather than inventing a
signal that was never computed (the same "no field, no check" discipline
`ValidationRuleGenerator` and `ReconciliationAgent` already follow).

When constructed with an `LLMClient`, `compare()` additionally asks it to
narrate the already-computed `DriftReport` into `DriftReport.narrative` -
never to decide a change's `classification`, `drift_status`, or
`automation_action`, all of which are already fixed by the time the model
is ever called.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import SCHEMA_DRIFT_MONITOR_SYSTEM_PROMPT
from dataos.ingestion.profiling import ColumnProfile, DatasetProfile
from dataos.llm.client import LLMClient
from dataos.semantics.dictionary import SemanticDictionary

Classification = Literal["COMPATIBLE", "REVIEW_REQUIRED", "BREAKING"]
DriftStatus = Literal["NONE", "COMPATIBLE", "REVIEW_REQUIRED", "BREAKING"]
AutomationAction = Literal["CONTINUE", "QUARANTINE", "STOP"]

_AUTOMATION_ACTION: dict[DriftStatus, AutomationAction] = {
    "NONE": "CONTINUE",
    "COMPATIBLE": "CONTINUE",
    "REVIEW_REQUIRED": "QUARANTINE",
    "BREAKING": "STOP",
}

_TEMPORAL_DTYPE_MARKERS = ("Date", "Datetime", "Time")


class DriftChange(BaseModel):
    kind: str
    column: str | None = None
    detail: str
    classification: Classification


class DriftReport(BaseModel):
    source_name: str
    drift_status: DriftStatus
    changes: list[DriftChange] = Field(default_factory=list)
    automation_action: AutomationAction
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see class
    docstring) - never authoritative; `drift_status` remains the
    decision."""


class SchemaDriftMonitor:
    def __init__(
        self,
        *,
        null_rate_review_threshold: float = 0.05,
        row_count_review_ratio: float = 0.5,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._null_rate_review_threshold = null_rate_review_threshold
        self._row_count_review_ratio = row_count_review_ratio
        self._llm_client = llm_client

    def compare(
        self,
        *,
        source_name: str,
        baseline_profile: DatasetProfile,
        current_profile: DatasetProfile,
        baseline_dictionary: SemanticDictionary | None = None,
        current_dictionary: SemanticDictionary | None = None,
    ) -> DriftReport:
        changes: list[DriftChange] = []

        baseline_columns = {c.name: c for c in baseline_profile.columns}
        current_columns = {c.name: c for c in current_profile.columns}

        for name in sorted(set(baseline_columns) - set(current_columns)):
            changes.append(
                DriftChange(
                    kind="column_removed",
                    column=name,
                    detail=f"column '{name}' is present in the baseline but missing from the current source",
                    classification="BREAKING",
                )
            )

        for name in sorted(set(current_columns) - set(baseline_columns)):
            changes.append(
                DriftChange(
                    kind="column_added",
                    column=name,
                    detail=f"column '{name}' is new since the baseline",
                    classification="COMPATIBLE",
                )
            )

        for name in sorted(set(baseline_columns) & set(current_columns)):
            changes.extend(self._column_changes(name, baseline_columns[name], current_columns[name]))

        if baseline_profile.row_count > 0:
            row_count_ratio = (
                abs(current_profile.row_count - baseline_profile.row_count) / baseline_profile.row_count
            )
            if row_count_ratio > self._row_count_review_ratio:
                changes.append(
                    DriftChange(
                        kind="row_count_shift",
                        detail=(
                            f"row count shifted from {baseline_profile.row_count} to "
                            f"{current_profile.row_count} ({row_count_ratio:.0%} change)"
                        ),
                        classification="REVIEW_REQUIRED",
                    )
                )

        if baseline_dictionary is not None and current_dictionary is not None:
            changes.extend(self._semantic_changes(baseline_dictionary, current_dictionary))

        status = _aggregate_status(changes)
        automation_action = _AUTOMATION_ACTION[status]

        narrative = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=SCHEMA_DRIFT_MONITOR_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {
                            "source_name": source_name,
                            "drift_status": status,
                            "automation_action": automation_action,
                            "changes": [c.model_dump() for c in changes],
                        },
                    }
                ],
            )
            narrative = narratives.get("summary")

        return DriftReport(
            source_name=source_name,
            drift_status=status,
            changes=changes,
            automation_action=automation_action,
            narrative=narrative,
        )

    def _column_changes(self, name: str, baseline: ColumnProfile, current: ColumnProfile) -> list[DriftChange]:
        changes: list[DriftChange] = []

        if baseline.dtype != current.dtype:
            changes.append(
                DriftChange(
                    kind="dtype_changed",
                    column=name,
                    detail=f"column '{name}' dtype changed from {baseline.dtype} to {current.dtype}",
                    classification="BREAKING",
                )
            )

        if baseline.is_candidate_key and not current.is_candidate_key:
            changes.append(
                DriftChange(
                    kind="key_uniqueness_lost",
                    column=name,
                    detail=(
                        f"column '{name}' was a candidate key in the baseline but is no longer unique/complete"
                    ),
                    classification="BREAKING",
                )
            )

        null_rate_delta = current.null_rate - baseline.null_rate
        if null_rate_delta > self._null_rate_review_threshold:
            changes.append(
                DriftChange(
                    kind="null_rate_increased",
                    column=name,
                    detail=(
                        f"column '{name}' null rate increased from {baseline.null_rate:.3f} "
                        f"to {current.null_rate:.3f}"
                    ),
                    classification="REVIEW_REQUIRED",
                )
            )

        if any(marker in baseline.dtype for marker in _TEMPORAL_DTYPE_MARKERS):
            if baseline.max_value is not None and current.max_value is not None:
                try:
                    regressed = current.max_value < baseline.max_value
                except TypeError:
                    regressed = False
                if regressed:
                    changes.append(
                        DriftChange(
                            kind="date_coverage_regressed",
                            column=name,
                            detail=(
                                f"column '{name}' max observed value regressed from "
                                f"{baseline.max_value} to {current.max_value}"
                            ),
                            classification="REVIEW_REQUIRED",
                        )
                    )

        return changes

    def _semantic_changes(
        self, baseline: SemanticDictionary, current: SemanticDictionary
    ) -> list[DriftChange]:
        changes: list[DriftChange] = []

        for semantic_id in sorted(set(baseline.metrics) - set(current.metrics)):
            changes.append(
                DriftChange(
                    kind="semantic_definition_removed",
                    detail=f"governed metric '{semantic_id}' no longer exists in the semantic dictionary",
                    classification="BREAKING",
                )
            )

        for semantic_id in sorted(set(baseline.metrics) & set(current.metrics)):
            before, after = baseline.metrics[semantic_id], current.metrics[semantic_id]
            if before.version != after.version or before.formula != after.formula:
                changes.append(
                    DriftChange(
                        kind="semantic_definition_changed",
                        detail=(
                            f"governed metric '{semantic_id}' definition changed "
                            f"(version {before.version} -> {after.version})"
                        ),
                        classification="REVIEW_REQUIRED",
                    )
                )

        if baseline.calendar is not None and current.calendar is not None:
            if baseline.calendar.model_dump() != current.calendar.model_dump():
                changes.append(
                    DriftChange(
                        kind="calendar_policy_changed",
                        detail="fiscal/calendar policy changed since the baseline",
                        classification="REVIEW_REQUIRED",
                    )
                )

        return changes


def _aggregate_status(changes: list[DriftChange]) -> DriftStatus:
    if not changes:
        return "NONE"
    classifications = {c.classification for c in changes}
    if "BREAKING" in classifications:
        return "BREAKING"
    if "REVIEW_REQUIRED" in classifications:
        return "REVIEW_REQUIRED"
    return "COMPATIBLE"
