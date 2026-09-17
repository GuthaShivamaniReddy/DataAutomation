"""Forecasting / ML Suitability Gate.

Source of truth: AI Prompt Library Section 16 "Forecasting and ML
Suitability Gate": "Decide whether a predictive or forecasting request is
valid before any model is trained... Never promise exact future values.
Predictions must remain labeled as estimates with uncertainty and model
version."

Deterministic code, not an LLM call - same reasoning as every other gate
in this package: whether a target/feature column exists, whether the
target leaks into the features, whether there is enough labeled data, and
whether a time column is present for a time-ordered split are all
provable facts about `DatasetProfile` evidence (Section 6), not a
judgment call. Class-imbalance can only be measured from real label
values, so it is checked only when a dataframe is supplied - exactly the
same "profile first, dataframe optional for a deeper check" convention
`JoinSafetyReviewer` already uses for match-rate/multiplication-factor
checks.

Section 16's checklist mixes two genuinely different questions this
codebase already keeps separate elsewhere, so this gate does too:
  - "is the labeled data itself suitable for training a model" - this is
    what `decision` actually measures, from real evidence, and can
    honestly be PROCEED.
  - "can this platform train a model at all" - it cannot: no model-
    training primitive exists anywhere in the `OperationRegistry`
    (`OperationRegistrySelector`, Section 10, has no operation_id for
    one). That fact is always surfaced as a standing risk, but never
    forces `decision` to REJECT by itself - a REJECT here must always be
    traceable to a real defect in the data, the same discipline
    `JoinSafetyReviewer.review()` already applies when it can APPROVE a
    join contract this codebase's executor cannot run yet either.

Split strategy, baseline, evaluation metrics, and uncertainty method are
fixed, reviewed methodology defaults keyed only by `task` (a closed,
non-domain-specific choice, unlike a business metric definition) - never
an invented number or business-specific choice.
"""

from __future__ import annotations

import re
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import ML_SUITABILITY_GATE_SYSTEM_PROMPT
from dataos.ingestion.profiling import DatasetProfile
from dataos.llm.client import LLMClient

Task = Literal["classification", "regression", "forecasting"]
Decision = Literal["PROCEED", "LIMITED", "REJECT"]

_MIN_SAMPLE_SIZE = 100
_MAX_TARGET_NULL_RATE = 0.2
_MIN_MINORITY_CLASS_RATE = 0.05
_DATE_DTYPE_PATTERN = re.compile("Date")

_PROTECTED_ATTRIBUTE_PATTERN = re.compile(
    r"(gender|\bsex\b|race|ethnicit|religio|disabilit|marital|pregnan|sexual[_-]?orientation|\bage\b)", re.I
)

_STRUCTURAL_RISK = (
    "no model-training primitive is registered in this platform's Operation Registry yet - even a PROCEED "
    "verdict here only certifies that the data is suitable for training in principle, not that this platform "
    "can execute it"
)
_RETRAINING_POLICY_NOTE = (
    "a retraining/drift policy is not decided by this gate - any recurring use of a trained model must go "
    "through the Automation Workflow Builder (Section 24) and be checked by the Schema Drift Monitor (Section 25) "
    "on every run"
)

_SPLIT_STRATEGY: dict[Task, str] = {
    "classification": "stratified train/validation/test split; never split in a way that separates rows of the same entity across sets",
    "regression": "random train/validation/test split, grouped by entity when the grain has repeated entities",
    "forecasting": "time-ordered train/validation/test split with no future data in the training window",
}
_BASELINE: dict[Task, str] = {
    "classification": "majority-class baseline",
    "regression": "mean/median-of-target baseline",
    "forecasting": "naive persistence (last-value) or seasonal-naive baseline",
}
_METRICS: dict[Task, list[str]] = {
    "classification": ["accuracy", "precision", "recall", "f1", "AUC (if binary)"],
    "regression": ["MAE", "RMSE", "R2"],
    "forecasting": ["MAE", "RMSE", "MAPE (if target is strictly positive)"],
}
_UNCERTAINTY_METHOD: dict[Task, str] = {
    "classification": "calibrated predicted probabilities (reliability curve / Brier score)",
    "regression": "prediction intervals from the residual distribution or quantile regression",
    "forecasting": "prediction intervals per horizon step, widening with horizon",
}


class MLSuitabilityRequest(BaseModel):
    task: Task
    target_column: str
    feature_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    """Required when task == "forecasting" for a time-ordered split."""


class MLSuitabilityResult(BaseModel):
    decision: Decision
    task: Task
    target: str
    split_strategy: str
    baseline: str
    metrics: list[str] = Field(default_factory=list)
    uncertainty_method: str
    risks: list[str] = Field(default_factory=list)
    reason: str
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `decision`/`risks` remain the
    decision."""


class MLSuitabilityGate:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def evaluate(
        self,
        *,
        request: MLSuitabilityRequest,
        profile: DatasetProfile,
        frame: pl.DataFrame | None = None,
    ) -> MLSuitabilityResult:
        blocking: list[str] = []
        risks: list[str] = []

        target_col = profile.column(request.target_column)
        if target_col is None:
            blocking.append(f"target column '{request.target_column}' does not exist in the profiled data")
        else:
            if target_col.null_rate > _MAX_TARGET_NULL_RATE:
                blocking.append(
                    f"target '{request.target_column}' is {target_col.null_rate:.1%} null - insufficient label availability"
                )
            elif target_col.null_rate > 0:
                risks.append(
                    f"target has some missing labels ({target_col.null_rate:.1%}); rows with a missing label "
                    "must be excluded from training, never imputed"
                )

        if request.target_column in request.feature_columns:
            blocking.append(f"target column '{request.target_column}' is also listed as a feature - direct label leakage")

        missing_features = [f for f in request.feature_columns if profile.column(f) is None]
        if missing_features:
            blocking.append(f"feature column(s) do not exist in the profiled data: {missing_features}")

        if profile.row_count < _MIN_SAMPLE_SIZE:
            blocking.append(
                f"sample size ({profile.row_count} row(s)) is below the minimum ({_MIN_SAMPLE_SIZE}) needed for "
                "a defensible train/validation/test split"
            )

        if request.task == "forecasting":
            if not request.time_column:
                blocking.append("forecasting requires an explicit time_column for a time-ordered split - none was declared")
            else:
                time_col = profile.column(request.time_column)
                if time_col is None:
                    blocking.append(f"time_column '{request.time_column}' does not exist in the profiled data")
                elif not _DATE_DTYPE_PATTERN.search(time_col.dtype):
                    blocking.append(f"time_column '{request.time_column}' is not a date/datetime column (dtype={time_col.dtype})")

        if request.task == "classification" and target_col is not None and not blocking:
            if frame is not None:
                labels = frame[request.target_column].drop_nulls()
                if labels.len() > 0:
                    minority_rate = labels.value_counts()["count"].min() / labels.len()
                    if minority_rate < _MIN_MINORITY_CLASS_RATE:
                        risks.append(f"severe class imbalance observed: the minority class is {minority_rate:.1%} of labeled rows")
            else:
                risks.append("class imbalance could not be verified from profiling evidence alone - supply the dataframe for a real distribution check")

        protected_features = [f for f in request.feature_columns if _PROTECTED_ATTRIBUTE_PATTERN.search(f)]
        if protected_features:
            risks.append(
                f"feature(s) {protected_features} may be a protected attribute - confirm fairness/legal review before training on them"
            )

        risks.append(_RETRAINING_POLICY_NOTE)
        risks.append(_STRUCTURAL_RISK)

        if blocking:
            decision: Decision = "REJECT"
            reason = "rejected: " + "; ".join(blocking)
        elif risks[:-2]:  # anything besides the two always-present standing notes
            decision = "LIMITED"
            reason = "limited: the data is usable but carries risk(s) that must be reviewed before training - see risks"
        else:
            decision = "PROCEED"
            reason = "all data-suitability checks passed; risks lists standing platform-level notes that still apply"

        result = MLSuitabilityResult(
            decision=decision,
            task=request.task,
            target=request.target_column,
            split_strategy=_SPLIT_STRATEGY[request.task],
            baseline=_BASELINE[request.task],
            metrics=_METRICS[request.task],
            uncertainty_method=_UNCERTAINTY_METHOD[request.task],
            risks=blocking + risks,
            reason=reason,
        )

        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=ML_SUITABILITY_GATE_SYSTEM_PROMPT,
                items=[{"ref": "summary", "facts": result.model_dump(exclude={"narrative"})}],
            )
            result = result.model_copy(update={"narrative": narratives.get("summary")})

        return result
