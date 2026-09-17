"""Join Safety Reviewer.

Source of truth: AI Prompt Library Section 12 "Join Safety Reviewer":
"Review every proposed join before execution... Require: left and right
datasets, join columns and business meaning, normalized datatype/format
compatibility, uniqueness/null statistics for keys, expected relationship
(1:1, 1:N, N:1, or explicitly allowed N:N), expected unmatched behavior,
pre-join row counts, post-join cardinality bounds, duplicate amplification
thresholds, reconciliation checks for additive measures. Reject accidental
many-to-many joins. Reject joins based only on similar column names. If
multiple keys could work, require clarification or governed mapping."

Deterministic code, not an LLM call - same reasoning as every other
gate/generator in this package (see `ambiguity_gate.py`,
`reconciliation_agent.py`): whether a join is safe is a provable fact
about the two datasets' real key columns, not a judgment call. This is the
planning-time counterpart to two things that already exist:
  - `registry/operations/join.py`'s `JoinOperation`, which enforces
    uniqueness/cardinality at *execution* time and raises; this reviewer
    runs the same checks earlier, against `DatasetProfile` evidence
    (Section 6) when the real dataframes aren't available yet, so an
    unsafe join can be caught (and CLARIFY/REJECT returned) before it is
    ever added to an approved workflow, not just when it happens to run.
  - `validation/checks/join_safety.py`'s reusable match-rate/
    multiplication-factor checks, reused here verbatim when dataframes are
    supplied.

Section 12's "Reject joins based only on similar column names" is about
*why* a key was chosen, not the data itself - nothing inspectable from a
profile or dataframe can prove that on its own. Callers must state it via
`key_evidence`; there is deliberately no default, since guessing "this key
was chosen well" is exactly the kind of silent assumption the Global
Constitution forbids.

When constructed with an `LLMClient`, `review()` additionally asks it to
narrate the already-computed `JoinReviewResult` into
`JoinReviewResult.narrative` - never to decide `decision`, `join_contract`,
or `issues`, all of which are already fixed by the time the model is ever
called (same discipline as `narrative.py`'s other five callers).
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import JOIN_SAFETY_REVIEWER_SYSTEM_PROMPT
from dataos.ingestion.profiling import DatasetProfile
from dataos.llm.client import LLMClient
from dataos.validation.checks.join_safety import check_join_match_rate, check_join_multiplication_factor

JoinType = Literal["inner", "left", "right", "outer"]
Cardinality = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
KeyEvidence = Literal["GOVERNED_MAPPING", "EXPLICIT_USER_MAPPING", "NAME_SIMILARITY_ONLY"]
Severity = Literal["BLOCKING", "NEEDS_CLARIFICATION", "WARNING"]

_DEFAULT_MIN_MATCH_RATE = 0.5
_DEFAULT_MAX_MULTIPLICATION_FACTOR = 2.0

_UNMATCHED_POLICY_BY_HOW: dict[JoinType, str] = {
    "inner": "drop rows unmatched on either side",
    "left": "keep every left row; an unmatched right side becomes null",
    "right": "keep every right row; an unmatched left side becomes null",
    "outer": "keep every row from both sides; an unmatched side becomes null",
}


class JoinIssue(BaseModel):
    severity: Severity
    code: str
    message: str


class JoinKey(BaseModel):
    left: str
    right: str


class JoinContractSpec(BaseModel):
    type: JoinType
    keys: list[JoinKey]
    expected_cardinality: Cardinality | None
    unmatched_policy: str
    prechecks: list[str] = Field(default_factory=list)
    postchecks: list[str] = Field(default_factory=list)


class JoinReviewResult(BaseModel):
    decision: Literal["APPROVE", "REJECT", "CLARIFY"]
    join_contract: JoinContractSpec | None = None
    issues: list[JoinIssue] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `decision` remains the decision."""


class JoinSafetyReviewer:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def review(
        self,
        *,
        left_name: str,
        right_name: str,
        left_keys: list[str],
        right_keys: list[str],
        how: JoinType,
        key_evidence: KeyEvidence,
        left_profile: DatasetProfile,
        right_profile: DatasetProfile,
        expected_cardinality: Cardinality | None = None,
        allow_many_to_many: bool = False,
        ambiguous_key_candidates: list[str] | None = None,
        left_frame: pl.DataFrame | None = None,
        right_frame: pl.DataFrame | None = None,
        min_match_rate: float = _DEFAULT_MIN_MATCH_RATE,
        max_multiplication_factor: float = _DEFAULT_MAX_MULTIPLICATION_FACTOR,
    ) -> JoinReviewResult:
        issues: list[JoinIssue] = []

        if not left_keys or not right_keys:
            issues.append(
                JoinIssue(
                    severity="BLOCKING",
                    code="NO_JOIN_KEYS",
                    message="a join must declare at least one key column on each side",
                )
            )
        elif len(left_keys) != len(right_keys):
            issues.append(
                JoinIssue(
                    severity="BLOCKING",
                    code="KEY_COUNT_MISMATCH",
                    message=f"left declares {len(left_keys)} key(s) but right declares {len(right_keys)}",
                )
            )

        left_columns = {c.name: c for c in left_profile.columns}
        right_columns = {c.name: c for c in right_profile.columns}
        missing_left = [k for k in left_keys if k not in left_columns]
        missing_right = [k for k in right_keys if k not in right_columns]
        if missing_left:
            issues.append(
                JoinIssue(
                    severity="BLOCKING",
                    code="KEY_MISSING",
                    message=f"left join key(s) do not exist in '{left_name}': {missing_left}",
                )
            )
        if missing_right:
            issues.append(
                JoinIssue(
                    severity="BLOCKING",
                    code="KEY_MISSING",
                    message=f"right join key(s) do not exist in '{right_name}': {missing_right}",
                )
            )

        keys_resolved = bool(left_keys) and len(left_keys) == len(right_keys) and not missing_left and not missing_right

        if key_evidence == "NAME_SIMILARITY_ONLY":
            issues.append(
                JoinIssue(
                    severity="BLOCKING",
                    code="NAME_SIMILARITY_ONLY",
                    message=(
                        f"join key(s) {list(zip(left_keys, right_keys))} were selected only because the "
                        "column names look similar - this requires a governed mapping or explicit user "
                        "confirmation, not name matching alone"
                    ),
                )
            )

        if keys_resolved:
            for lk, rk in zip(left_keys, right_keys):
                left_dtype = left_columns[lk].dtype
                right_dtype = right_columns[rk].dtype
                if left_dtype != right_dtype:
                    issues.append(
                        JoinIssue(
                            severity="BLOCKING",
                            code="KEY_DTYPE_MISMATCH",
                            message=(
                                f"key dtype mismatch: {left_name}.{lk} ({left_dtype}) vs "
                                f"{right_name}.{rk} ({right_dtype})"
                            ),
                        )
                    )

        if ambiguous_key_candidates:
            issues.append(
                JoinIssue(
                    severity="NEEDS_CLARIFICATION",
                    code="AMBIGUOUS_KEY_CANDIDATES",
                    message=(
                        f"more than one column could plausibly serve as the join key: "
                        f"{ambiguous_key_candidates} - confirm which one is authoritative"
                    ),
                )
            )

        if expected_cardinality is None:
            issues.append(
                JoinIssue(
                    severity="NEEDS_CLARIFICATION",
                    code="CARDINALITY_UNDECLARED",
                    message="no expected join cardinality was declared (one_to_one/one_to_many/many_to_one/many_to_many)",
                )
            )
        elif expected_cardinality == "many_to_many" and not allow_many_to_many:
            issues.append(
                JoinIssue(
                    severity="BLOCKING",
                    code="UNAPPROVED_MANY_TO_MANY",
                    message=(
                        "expected_cardinality is many_to_many but allow_many_to_many is not set - "
                        "accidental many-to-many joins are rejected"
                    ),
                )
            )

        if keys_resolved and expected_cardinality is not None:
            left_must_be_unique = expected_cardinality in ("one_to_one", "one_to_many")
            right_must_be_unique = expected_cardinality in ("one_to_one", "many_to_one")
            issues.extend(
                self._uniqueness_issues(
                    side="left", name=left_name, keys=left_keys, must_be_unique=left_must_be_unique,
                    profile=left_profile, frame=left_frame,
                )
            )
            issues.extend(
                self._uniqueness_issues(
                    side="right", name=right_name, keys=right_keys, must_be_unique=right_must_be_unique,
                    profile=right_profile, frame=right_frame,
                )
            )

        prechecks = [
            f"{left_name}.{left_keys} and {right_name}.{right_keys} exist with compatible dtypes",
            f"pre-join row counts: {left_name}={left_profile.row_count}, {right_name}={right_profile.row_count}",
        ]
        postchecks: list[str] = []
        if keys_resolved and left_frame is not None and right_frame is not None:
            match_result = check_join_match_rate(
                left_frame, right_frame, left_keys=left_keys, right_keys=right_keys,
                min_match_rate=min_match_rate, severity="WARNING",
            )
            postchecks.append(f"match_rate >= {min_match_rate} (observed {match_result.observed:.3f})")
            if not match_result.passed:
                issues.append(
                    JoinIssue(
                        severity="WARNING",
                        code="LOW_MATCH_RATE",
                        message=(
                            f"only {match_result.observed:.1%} of '{left_name}' rows matched a '{right_name}' "
                            f"row (threshold {min_match_rate:.1%})"
                        ),
                    )
                )

            mult_result = check_join_multiplication_factor(
                left_frame, right_frame, left_keys=left_keys, right_keys=right_keys,
                max_factor=max_multiplication_factor, how=how, severity="WARNING",
            )
            postchecks.append(
                f"row_multiplication_factor <= {max_multiplication_factor} (observed {mult_result.observed:.3f})"
            )
            if not mult_result.passed:
                issues.append(
                    JoinIssue(
                        severity="WARNING",
                        code="AMPLIFICATION_RISK",
                        message=(
                            f"joining '{left_name}' to '{right_name}' multiplies row count by "
                            f"{mult_result.observed:.2f}x (threshold {max_multiplication_factor}x) - verify this "
                            "fan-out is expected before trusting downstream sums"
                        ),
                    )
                )
            postchecks.append("reconcile additive measures computed pre- and post-join within tolerance")
        else:
            postchecks.append(
                "match_rate and row_multiplication_factor unavailable - no dataframes were supplied to the reviewer"
            )

        blocking = [i for i in issues if i.severity == "BLOCKING"]
        needs_clarification = [i for i in issues if i.severity == "NEEDS_CLARIFICATION"]

        if blocking:
            decision: Literal["APPROVE", "REJECT", "CLARIFY"] = "REJECT"
        elif needs_clarification:
            decision = "CLARIFY"
        else:
            decision = "APPROVE"

        join_contract = None
        if keys_resolved:
            join_contract = JoinContractSpec(
                type=how,
                keys=[JoinKey(left=lk, right=rk) for lk, rk in zip(left_keys, right_keys)],
                expected_cardinality=expected_cardinality,
                unmatched_policy=_UNMATCHED_POLICY_BY_HOW[how],
                prechecks=prechecks,
                postchecks=postchecks,
            )

        narrative_text = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=JOIN_SAFETY_REVIEWER_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {
                            "decision": decision,
                            "left": left_name,
                            "right": right_name,
                            "issues": [i.model_dump() for i in issues],
                        },
                    }
                ],
            )
            narrative_text = narratives.get("summary")

        return JoinReviewResult(decision=decision, join_contract=join_contract, issues=issues, narrative=narrative_text)

    def _uniqueness_issues(
        self,
        *,
        side: str,
        name: str,
        keys: list[str],
        must_be_unique: bool,
        profile: DatasetProfile,
        frame: pl.DataFrame | None,
    ) -> list[JoinIssue]:
        if not must_be_unique:
            return []

        if frame is not None:
            duplicate_groups = frame.group_by(keys).len(name="count").filter(pl.col("count") > 1)
            if duplicate_groups.height > 0:
                return [
                    JoinIssue(
                        severity="BLOCKING",
                        code="DUPLICATE_KEY",
                        message=(
                            f"'{name}' is declared unique on {keys} but {duplicate_groups.height} duplicate key "
                            "group(s) were found in the real data - this join would silently multiply rows"
                        ),
                    )
                ]
            return []

        if len(keys) == 1:
            column = profile.column(keys[0])
            if column is not None and column.is_candidate_key:
                return []

        return [
            JoinIssue(
                severity="NEEDS_CLARIFICATION",
                code="UNIQUENESS_UNVERIFIED",
                message=(
                    f"'{name}' must be unique on {keys} for the declared cardinality, but this cannot be proven "
                    f"from profiling evidence alone (side={side}) - supply the dataframe for a duplicate-key "
                    "check or confirm via governed mapping"
                ),
            )
        ]
