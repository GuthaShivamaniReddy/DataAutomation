"""Data Quality Assessor.

Source of truth: AI Prompt Library Section 8 "Data Quality Assessor":
"Evaluate whether the source data is fit for the specific requirement.
Quality is contextual: a dataset can be adequate for one request and
inadequate for another... For each issue, quantify impact and classify as
BLOCKING, REQUIRES_RULE, WARNING, or ACCEPTABLE. Do not propose dropping
records unless the Requirement Contract permits it."

Deterministic code, not an LLM call - same reasoning as the other
gates/generators in this package: fitness for a requirement is a provable
fact about `DatasetProfile` evidence (Section 6) and, where supplied, the
Schema Mapping Agent's already-resolved concept->column links (Section
7), not a judgment call.

Section 8's checklist is long; several bullets are deliberately left to
the component that already owns them rather than re-implemented here,
so two components can never silently disagree about the same fact:
  - "referential integrity for joins" -> Join Safety Reviewer (Section 12).
  - "distribution shifts" / "schema drift" -> Schema Drift Monitor
    (Section 25), which already compares against a baseline profile.
  - "reconciliation anchors when available" -> Reconciliation Agent
    (Section 20).
  - "impossible / invalid values" against a business rule (a range, an
    enum) -> Validation Rule Generator (Section 19) /
    `validation/checks/data_quality.py`, which run those checks against
    real execution artifacts, not pre-execution profiles.

What remains, and is implemented here, is exactly what a `DatasetProfile`
plus an optional `SchemaMappingResult` can prove before any workflow is
built: completeness of the fields a mapping actually resolved, whether a
dataset has any provable candidate key at all, duplicate rows, a column
name whose dtype disagrees across sources, an undeclared currency policy
for a monetary concept, and a declared time range with no date column
anywhere to apply it to.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import DATA_QUALITY_ASSESSOR_SYSTEM_PROMPT
from dataos.compiler.schema_mapping_agent import SchemaMappingResult
from dataos.contracts.requirement_contract import RequirementContract
from dataos.ingestion.profiling import DatasetProfile
from dataos.llm.client import LLMClient

Severity = Literal["BLOCKING", "REQUIRES_RULE", "WARNING", "ACCEPTABLE"]
Fitness = Literal["PASS", "CONDITIONAL", "FAIL"]

_MONEY_TERM_PATTERN = re.compile(
    r"(revenue|amount|sales|price|cost|total|balance|payment|spend|margin|profit)", re.I
)
_DATE_DTYPE_PATTERN = re.compile("Date")

# A field with any null at all requires an explicit rule; above this
# fraction the source is not fit for the requirement at all regardless of
# a rule, since too little real data would remain.
_NULL_RATE_BLOCKING_THRESHOLD = 0.5
_DUPLICATE_RATE_BLOCKING_THRESHOLD = 0.5


class QualityIssue(BaseModel):
    severity: Severity
    rule: str
    observed: str
    impact: str
    required_action: str


class DataQualityReport(BaseModel):
    fitness: Fitness
    issues: list[QualityIssue] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `fitness`/`issues` remain the
    decision."""


class DataQualityAssessor:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def assess(
        self,
        *,
        contract: RequirementContract,
        profiles: dict[str, DatasetProfile],
        schema_mapping: SchemaMappingResult | None = None,
    ) -> DataQualityReport:
        issues: list[QualityIssue] = [
            *self._completeness_issues(schema_mapping, profiles),
            *self._uniqueness_issues(profiles),
            *self._duplicate_issues(profiles),
            *self._cross_source_dtype_issues(profiles),
            *self._currency_policy_issues(contract, schema_mapping),
            *self._time_coverage_issues(contract, profiles),
            *self._text_formatting_issues(schema_mapping, profiles),
        ]

        fitness = self._fitness(issues)
        evidence_refs = [f"profile:{name}" for name in profiles]

        narrative_text = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=DATA_QUALITY_ASSESSOR_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {"fitness": fitness, "issues": [i.model_dump() for i in issues]},
                    }
                ],
            )
            narrative_text = narratives.get("summary")

        return DataQualityReport(fitness=fitness, issues=issues, evidence_refs=evidence_refs, narrative=narrative_text)

    def _completeness_issues(
        self, schema_mapping: SchemaMappingResult | None, profiles: dict[str, DatasetProfile]
    ) -> list[QualityIssue]:
        if schema_mapping is None:
            return []
        issues: list[QualityIssue] = []
        for mapping in schema_mapping.mappings:
            if mapping.status != "MAPPED" or not mapping.dataset or not mapping.column:
                continue
            profile = profiles.get(mapping.dataset)
            if profile is None:
                continue
            col = profile.column(mapping.column)
            if col is None or col.null_rate <= 0:
                continue

            blocking = col.null_rate > _NULL_RATE_BLOCKING_THRESHOLD
            issues.append(
                QualityIssue(
                    severity="BLOCKING" if blocking else "REQUIRES_RULE",
                    rule=f"completeness:{mapping.dataset}.{mapping.column}",
                    observed=f"{col.null_rate:.1%} null ({col.null_count} of {profile.row_count} row(s))",
                    impact=f"'{mapping.concept}' cannot be computed for rows missing '{mapping.dataset}.{mapping.column}'",
                    required_action=(
                        "source is not fit for this requirement until the missing-data rate is remediated"
                        if blocking
                        else "define an explicit null-handling rule (Cleaning Strategy Agent) before this field is used"
                    ),
                )
            )
        return issues

    def _text_formatting_issues(
        self, schema_mapping: SchemaMappingResult | None, profiles: dict[str, DatasetProfile]
    ) -> list[QualityIssue]:
        if schema_mapping is None:
            return []
        issues: list[QualityIssue] = []
        for mapping in schema_mapping.mappings:
            if mapping.status != "MAPPED" or not mapping.dataset or not mapping.column:
                continue
            profile = profiles.get(mapping.dataset)
            if profile is None:
                continue
            col = profile.column(mapping.column)
            if col is None or col.trimmed_lowercase_distinct_count is None:
                continue
            if col.trimmed_lowercase_distinct_count >= col.distinct_count:
                continue

            collapsed = col.distinct_count - col.trimmed_lowercase_distinct_count
            issues.append(
                QualityIssue(
                    severity="REQUIRES_RULE",
                    rule=f"text_formatting:{mapping.dataset}.{mapping.column}",
                    observed=(
                        f"{col.distinct_count} distinct value(s) fold to {col.trimmed_lowercase_distinct_count} "
                        f"after trim+lowercase ({collapsed} case/whitespace-only variant(s))"
                    ),
                    impact=f"'{mapping.concept}' may double-count or mis-group rows that only differ by case/whitespace in '{mapping.dataset}.{mapping.column}'",
                    required_action="define an explicit text-normalization rule (Cleaning Strategy Agent) before this field is used for grouping/joining",
                )
            )
        return issues

    def _uniqueness_issues(self, profiles: dict[str, DatasetProfile]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for name, profile in profiles.items():
            if profile.row_count > 0 and not profile.candidate_key_columns:
                issues.append(
                    QualityIssue(
                        severity="WARNING",
                        rule=f"uniqueness:{name}",
                        observed="no single column is fully populated and unique",
                        impact=f"row-level uniqueness/grain for '{name}' cannot be verified from profiling evidence alone",
                        required_action="confirm the intended grain and unique key explicitly, or supply a composite-key check",
                    )
                )
        return issues

    def _duplicate_issues(self, profiles: dict[str, DatasetProfile]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for name, profile in profiles.items():
            if profile.duplicate_row_count <= 0:
                continue
            rate = profile.duplicate_row_count / profile.row_count if profile.row_count else 0.0
            issues.append(
                QualityIssue(
                    severity="BLOCKING" if rate > _DUPLICATE_RATE_BLOCKING_THRESHOLD else "REQUIRES_RULE",
                    rule=f"duplicates:{name}",
                    observed=f"{profile.duplicate_row_count} duplicate row(s) ({rate:.1%} of {profile.row_count})",
                    impact="duplicate rows silently inflate any additive aggregate computed over this dataset",
                    required_action="define an explicit deduplication rule before aggregating; never drop duplicates implicitly",
                )
            )
        return issues

    def _cross_source_dtype_issues(self, profiles: dict[str, DatasetProfile]) -> list[QualityIssue]:
        dtypes_by_column: dict[str, dict[str, str]] = {}
        for dataset, profile in profiles.items():
            for col in profile.columns:
                dtypes_by_column.setdefault(col.name, {})[dataset] = col.dtype

        issues: list[QualityIssue] = []
        for column, by_dataset in dtypes_by_column.items():
            if len(by_dataset) > 1 and len(set(by_dataset.values())) > 1:
                issues.append(
                    QualityIssue(
                        severity="WARNING",
                        rule=f"cross_source_dtype:{column}",
                        observed=f"dtype differs across sources: {by_dataset}",
                        impact=f"joining or comparing '{column}' across these sources without an explicit cast could silently fail or misbehave",
                        required_action="confirm whether these columns represent the same concept and add an explicit cast/mapping rule if so",
                    )
                )
        return issues

    def _currency_policy_issues(
        self, contract: RequirementContract, schema_mapping: SchemaMappingResult | None
    ) -> list[QualityIssue]:
        if contract.units.currency:
            return []

        money_concepts: set[str] = set()
        if schema_mapping is not None:
            money_concepts = {
                m.concept for m in schema_mapping.mappings if m.status == "MAPPED" and _MONEY_TERM_PATTERN.search(m.concept)
            }
        if not money_concepts:
            money_concepts = {m.name for m in contract.metrics if _MONEY_TERM_PATTERN.search(m.name)}

        if not money_concepts:
            return []
        return [
            QualityIssue(
                severity="REQUIRES_RULE",
                rule="currency_policy",
                observed=f"no currency declared in contract.units.currency for monetary concept(s) {sorted(money_concepts)}",
                impact="mixing currencies without an FX/consolidation policy silently produces an incorrect total",
                required_action="declare contract.units.currency or an explicit FX policy before aggregating these concepts",
            )
        ]

    def _time_coverage_issues(self, contract: RequirementContract, profiles: dict[str, DatasetProfile]) -> list[QualityIssue]:
        if not contract.time.range:
            return []
        has_date_column = any(
            _DATE_DTYPE_PATTERN.search(col.dtype) for profile in profiles.values() for col in profile.columns
        )
        if has_date_column:
            return []
        return [
            QualityIssue(
                severity="BLOCKING",
                rule="time_coverage",
                observed=f"time range '{contract.time.range}' was requested but no date/datetime column exists in any profiled source",
                impact="the requested time filter/period cannot be applied to this data at all",
                required_action="identify the correct date field or source before this requirement can proceed",
            )
        ]

    def _fitness(self, issues: list[QualityIssue]) -> Fitness:
        if any(i.severity == "BLOCKING" for i in issues):
            return "FAIL"
        if any(i.severity == "REQUIRES_RULE" for i in issues):
            return "CONDITIONAL"
        return "PASS"
