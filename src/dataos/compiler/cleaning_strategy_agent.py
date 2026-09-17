"""Data Cleaning Strategy Agent.

Source of truth: AI Prompt Library Section 11 "Cleaning Strategy Agent":
"Propose cleaning rules only when required by the Requirement Contract or
data quality findings... NEVER automatically: delete outliers merely
because they are extreme; impute business-critical values without a
governed rule; merge near-duplicate entities without entity-resolution
evidence; coerce failed parses to null and continue silently; normalize
identifiers in ways that can change identity."

Deterministic code, not an LLM call - same reasoning as every other
gate/generator in this package: whether a cleaning action is *proposed*
(never applied) and whether it requires approval is fully determined by
the Requirement Contract's own `null_policy` and a `DataQualityReport`'s
issues, not a judgment call.

Two independent, non-overlapping trigger sources, exactly matching
Section 11's own "propose ... only when required by":
  - `RequirementContract.null_policy` - a column already governed
    ("error" | "ignore" | "fill:<value>") always yields a rule, since the
    contract itself is the required authorization; `validation_rule_
    generator.py`'s own comment ("'ignore' / 'fill:<value>' are cleaning
    rules, not validation assertions") is exactly the gap this class
    fills.
  - `DataQualityReport.issues` (Section 8) whose severity is
    REQUIRES_RULE or BLOCKING - a WARNING alone is not "required" by
    anything and is left alone.

A completeness issue for a column with no governed `null_policy` entry
never gets an invented rule - Constitution rule 4 ("Never silently drop
rows... nulls... without an explicit change") means the honest output is
a blocking item asking for that policy, not a guessed fill value.
Similarly, `cross_source_dtype`/`currency_policy`/`time_coverage` issues
need a contract-level decision (which source is authoritative, what
currency/FX policy applies, which date field to use) this agent has no
authority to make, so they always surface as blocking items instead of a
proposed rule.

`approval_required` is true for every rule that is lossy (a fill or a
deduplication) and false only for a rule that merely restates an
already-governed, already-approved contract policy ("error"/"ignore") -
Section 11's own "business approval requirement" field, decided by the
same lossy/governed distinction throughout this codebase (see
`pii_classifier.py`'s analogous "never relax toward the permissive
reading" discipline).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from dataos.compiler.data_quality_assessor import DataQualityReport, QualityIssue
from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import CLEANING_STRATEGY_AGENT_SYSTEM_PROMPT
from dataos.contracts.requirement_contract import RequirementContract
from dataos.llm.client import LLMClient

_CONTRACT_LEVEL_ISSUE_PREFIXES = ("cross_source_dtype:", "currency_policy", "time_coverage")


class CleaningRule(BaseModel):
    rule_id: str
    condition: str
    action: str
    lossy: bool
    expected_impact: str | None = None
    validation: str
    approval_required: bool


class CleaningBlockingItem(BaseModel):
    field: str
    reason: str


class CleaningPlan(BaseModel):
    rules: list[CleaningRule] = Field(default_factory=list)
    blocking_items: list[CleaningBlockingItem] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `rules`/`blocking_items` remain the
    decision."""


class CleaningStrategyAgent:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def propose(
        self,
        *,
        contract: RequirementContract,
        quality_report: DataQualityReport | None = None,
    ) -> CleaningPlan:
        rules: list[CleaningRule] = []
        blocking_items: list[CleaningBlockingItem] = []

        for column, policy in contract.null_policy.items():
            rule = self._rule_from_null_policy(column, policy)
            if rule is not None:
                rules.append(rule)
            else:
                blocking_items.append(
                    CleaningBlockingItem(
                        field=column,
                        reason=(
                            f"null_policy '{policy}' for '{column}' is not one of 'error' | 'ignore' | "
                            "'fill:<value>' - no cleaning rule can be derived from it"
                        ),
                    )
                )

        if quality_report is not None:
            for issue in quality_report.issues:
                if issue.severity not in ("REQUIRES_RULE", "BLOCKING"):
                    continue
                if issue.rule.startswith("completeness:"):
                    self._handle_completeness_issue(issue, contract, blocking_items)
                elif issue.rule.startswith("duplicates:"):
                    rules.append(self._rule_from_duplicates_issue(issue))
                elif issue.rule.startswith("text_formatting:"):
                    rules.append(self._rule_from_text_formatting_issue(issue))
                elif issue.rule.startswith(_CONTRACT_LEVEL_ISSUE_PREFIXES):
                    blocking_items.append(
                        CleaningBlockingItem(
                            field=issue.rule,
                            reason=(
                                f"{issue.impact} - this requires a contract-level decision, not a cleaning rule"
                            ),
                        )
                    )

        narrative_text = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=CLEANING_STRATEGY_AGENT_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {
                            "rules": [r.model_dump() for r in rules],
                            "blocking_items": [b.model_dump() for b in blocking_items],
                        },
                    }
                ],
            )
            narrative_text = narratives.get("summary")

        return CleaningPlan(rules=rules, blocking_items=blocking_items, narrative=narrative_text)

    def _rule_from_null_policy(self, column: str, policy: str) -> CleaningRule | None:
        if policy == "error":
            return CleaningRule(
                rule_id=f"null_policy:{column}",
                condition=f"null in '{column}'",
                action="reject - do not fill or drop; any null in this field fails the run",
                lossy=False,
                validation=f"null_rate({column}) <= 0.0",
                approval_required=False,
            )
        if policy == "ignore":
            return CleaningRule(
                rule_id=f"null_policy:{column}",
                condition=f"null in '{column}'",
                action="ignore - proceed with the null value; no imputation is performed",
                lossy=False,
                validation="downstream aggregates over this field must be null-safe (skip nulls, never coerce to 0 implicitly)",
                approval_required=False,
            )
        if policy.startswith("fill:"):
            value = policy.split(":", 1)[1]
            return CleaningRule(
                rule_id=f"null_policy:{column}",
                condition=f"null in '{column}'",
                action=f"fill with the governed literal value '{value}'",
                lossy=True,
                expected_impact=(
                    f"every null in '{column}' is replaced with '{value}'; the fact that a value was "
                    "originally missing is not retained unless a companion indicator column is added"
                ),
                validation=f"post-fill null_rate({column}) == 0.0; original null positions must remain "
                "reconstructable from the run's lineage/audit evidence",
                approval_required=True,
            )
        return None

    def _handle_completeness_issue(
        self, issue: QualityIssue, contract: RequirementContract, blocking_items: list[CleaningBlockingItem]
    ) -> None:
        ref = issue.rule.split(":", 1)[1]  # "<dataset>.<column>"
        _, _, column = ref.partition(".")
        if column in contract.null_policy:
            return  # already produced a rule from the null_policy loop above
        blocking_items.append(
            CleaningBlockingItem(
                field=ref,
                reason=(
                    f"'{ref}' has observed nulls ({issue.observed}) but no null_policy is declared for "
                    f"'{column}' - a cleaning rule is never invented without a governed policy"
                ),
            )
        )

    def _rule_from_duplicates_issue(self, issue: QualityIssue) -> CleaningRule:
        dataset = issue.rule.split(":", 1)[1]
        return CleaningRule(
            rule_id=f"dedup:{dataset}",
            condition=f"exact duplicate rows in '{dataset}' - observed {issue.observed}",
            action=f"deduplicate '{dataset}' via the registered 'deduplicate' operation, keeping the first occurrence per full-row match",
            lossy=True,
            expected_impact=issue.observed,
            validation="post-dedup row_count must reconcile against the pre-dedup row_count minus the removed duplicate count",
            approval_required=True,
        )

    def _rule_from_text_formatting_issue(self, issue: QualityIssue) -> CleaningRule:
        ref = issue.rule.split(":", 1)[1]  # "<dataset>.<column>"
        _, _, column = ref.partition(".")
        return CleaningRule(
            rule_id=f"text_clean:{ref}",
            condition=f"case/whitespace-only variants in '{ref}' - observed {issue.observed}",
            action=(
                f"normalize '{column}' via the registered 'text_clean' operation "
                "(normalize: trim, lower) with allow_identity_collapse=True"
            ),
            lossy=True,
            expected_impact=issue.observed,
            validation=(
                f"post-clean distinct_count for '{column}' must equal the pre-clean trimmed_lowercase_distinct_count "
                "reported by profiling; re-profile to confirm no further variants remain"
            ),
            approval_required=True,
        )
