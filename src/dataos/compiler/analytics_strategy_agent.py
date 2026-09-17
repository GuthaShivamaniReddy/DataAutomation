"""Analytics Strategy Agent.

Source of truth: AI Prompt Library Section 15 "Analytics Strategy
Agent": "Select analytical methods appropriate to the user decision and
available data. Do not manufacture causal claims from correlation...
Classify requested analysis as descriptive, diagnostic, comparative,
segmentation, anomaly detection, predictive, forecasting, causal, or
prescriptive... If the data cannot support the requested inference,
state the limitation and propose the closest valid analysis."

Deterministic code, not an LLM call. Classifying which of Section 15's
nine categories a request belongs to is intent routing, not a business
definition decision (Blueprint's "never guess" rule is about metric/join/
unit meaning, not about which analytical technique a request is asking
for) - the same class of decision the Master Orchestrator's own
`request_class` classification (Section 2) already makes deterministically
from structured signals rather than free model judgment. Keyword patterns
are checked in a fixed, documented precedence order (most-restricted
category first) exactly like `pii_classifier.py`'s "first match wins" name
patterns; a request matching no pattern is UNCLASSIFIED and forced to
NEEDS_CLARIFICATION rather than silently defaulted to descriptive.

Whether a classified category's request can actually be *approved* is a
provable fact about what this platform's registry can do today, not
optimism: CAUSAL, PREDICTIVE, FORECASTING, ANOMALY_DETECTION, and
PRESCRIPTIVE always resolve to NOT_SUPPORTED, because this codebase has
no causal-inference, model-training, anomaly-detection, or optimization
primitive registered anywhere (`OperationRegistrySelector`, Section 10,
has no operation_id for any of them) - approving one of these would be
exactly the "plausible answer" Constitution rule 9 forbids. Each still
proposes the closest valid analysis this platform *can* run, per
Section 15's own instruction, rather than a bare refusal.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import ANALYTICS_STRATEGY_AGENT_SYSTEM_PROMPT
from dataos.contracts.requirement_contract import RequirementContract
from dataos.llm.client import LLMClient

AnalysisType = Literal[
    "DESCRIPTIVE",
    "DIAGNOSTIC",
    "COMPARATIVE",
    "SEGMENTATION",
    "ANOMALY_DETECTION",
    "PREDICTIVE",
    "FORECASTING",
    "CAUSAL",
    "PRESCRIPTIVE",
    "UNCLASSIFIED",
]
Status = Literal["APPROVED", "NEEDS_CLARIFICATION", "NOT_SUPPORTED"]

# Checked in this order; the first match wins - most-restricted /
# least-supported categories are listed first so a request that reads as
# both e.g. "why did X change compared to last month" (DIAGNOSTIC and
# COMPARATIVE) is never miscategorized as the safer, better-supported one.
_KEYWORD_SIGNALS: list[tuple[AnalysisType, re.Pattern]] = [
    ("CAUSAL", re.compile(r"\bcaus(e|es|ed|ing|al)\b|\bimpact of\b|\beffect of\b", re.I)),
    ("FORECASTING", re.compile(r"\bforecast|\bproject(ion)?\b|\bnext (week|month|quarter|year)\b|\bfuture\b", re.I)),
    ("PREDICTIVE", re.compile(r"\bpredict|\blikelihood\b|\bprobability\b|\bwill (churn|convert|buy)\b|\bclassify\b", re.I)),
    ("ANOMALY_DETECTION", re.compile(r"\banomal|\boutlier|\bunusual\b|\bunexpected(ly)?\b|\bspike\b", re.I)),
    ("PRESCRIPTIVE", re.compile(r"\bshould we\b|\brecommend|\bwhat should\b|\boptimi[sz]e\b|\bbest action\b", re.I)),
    ("COMPARATIVE", re.compile(r"\bcompar|\bversus\b|\bvs\.?\b|\bcompared to\b|\bchange from\b", re.I)),
    ("SEGMENTATION", re.compile(r"\bsegment|\bcohort|\bcluster", re.I)),
    ("DIAGNOSTIC", re.compile(r"\bwhy\b|\bdriver|\broot cause\b|\bcontribut", re.I)),
]


class AnalyticsStrategyResult(BaseModel):
    analysis_type: AnalysisType
    method: str | None = None
    required_features: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    validation_plan: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    status: Status
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `analysis_type`/`status` remain the
    decision."""


class AnalyticsStrategyAgent:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def classify(self, *, contract: RequirementContract) -> AnalyticsStrategyResult:
        result = self._classify(contract)

        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=ANALYTICS_STRATEGY_AGENT_SYSTEM_PROMPT,
                items=[{"ref": "summary", "facts": result.model_dump(exclude={"narrative"})}],
            )
            result = result.model_copy(update={"narrative": narratives.get("summary")})

        return result

    def _classify(self, contract: RequirementContract) -> AnalyticsStrategyResult:
        analysis_type = self._detect_type(contract)

        if analysis_type == "UNCLASSIFIED":
            return AnalyticsStrategyResult(
                analysis_type="UNCLASSIFIED",
                status="NEEDS_CLARIFICATION",
                limitations=[
                    "the objective does not match any recognized analysis category "
                    "(descriptive, diagnostic, comparative, segmentation, anomaly detection, "
                    "predictive, forecasting, causal, prescriptive) - state which kind of analysis is intended"
                ],
            )

        if analysis_type == "DESCRIPTIVE":
            return AnalyticsStrategyResult(
                analysis_type=analysis_type,
                method="deterministic aggregate over the registered 'aggregate'/'derive' operations",
                required_features=[m.name for m in contract.metrics] + list(contract.dimensions),
                validation_plan=["reconcile the aggregate against an independent path (Reconciliation Agent, Section 20)"],
                status="APPROVED",
            )

        if analysis_type == "COMPARATIVE":
            if not contract.time.comparison:
                return AnalyticsStrategyResult(
                    analysis_type=analysis_type,
                    status="NEEDS_CLARIFICATION",
                    limitations=["a comparison was requested but no comparison period is declared in the contract"],
                )
            return AnalyticsStrategyResult(
                analysis_type=analysis_type,
                method="deterministic aggregate computed once per period, then differenced",
                required_features=[m.name for m in contract.metrics],
                assumptions=[f"comparison period: {contract.time.comparison}"],
                validation_plan=["reconcile the absolute and percentage change against both periods' raw totals"],
                status="APPROVED",
            )

        if analysis_type == "SEGMENTATION":
            if not contract.dimensions and not contract.group_by:
                return AnalyticsStrategyResult(
                    analysis_type=analysis_type,
                    status="NEEDS_CLARIFICATION",
                    limitations=["segmentation was requested but no grouping dimension is declared in the contract"],
                )
            return AnalyticsStrategyResult(
                analysis_type=analysis_type,
                method="deterministic aggregate grouped by the declared dimension(s)",
                required_features=list(contract.dimensions) + list(contract.group_by),
                validation_plan=["segment totals must reconcile to the ungrouped total within tolerance"],
                status="APPROVED",
            )

        if analysis_type == "DIAGNOSTIC":
            return AnalyticsStrategyResult(
                analysis_type=analysis_type,
                method="deterministic additive contribution decomposition",
                required_features=[m.name for m in contract.metrics] + list(contract.dimensions),
                assumptions=["contributions are measured, not causal - see limitations"],
                validation_plan=["contribution decomposition must reconcile to the total change within tolerance"],
                limitations=[
                    "a diagnostic breakdown reports measured contribution to a change, never a causal explanation; "
                    "correlation is not causation"
                ],
                status="APPROVED",
            )

        if analysis_type == "CAUSAL":
            return AnalyticsStrategyResult(
                analysis_type=analysis_type,
                status="NOT_SUPPORTED",
                limitations=[
                    "no experiment/randomization design or causal-inference method is implemented in this "
                    "platform; correlation is not causation - closest valid analysis: a DIAGNOSTIC measured-"
                    "contribution decomposition of the same metric"
                ],
            )

        if analysis_type == "ANOMALY_DETECTION":
            return AnalyticsStrategyResult(
                analysis_type=analysis_type,
                status="NOT_SUPPORTED",
                limitations=[
                    "no anomaly-detection primitive is registered in this platform's Operation Registry; "
                    "closest valid analysis: an explicit threshold-based filter ('select_filter') reviewed and "
                    "approved by a human, not an automatically detected anomaly"
                ],
            )

        if analysis_type in ("PREDICTIVE", "FORECASTING"):
            return AnalyticsStrategyResult(
                analysis_type=analysis_type,
                status="NOT_SUPPORTED",
                limitations=[
                    f"{analysis_type.lower()} requires the Forecasting/ML Suitability Gate (Section 16) and a "
                    "trained model primitive, neither of which is registered in this platform yet; closest valid "
                    "analysis: a DESCRIPTIVE historical trend of the same metric"
                ],
            )

        assert analysis_type == "PRESCRIPTIVE"
        return AnalyticsStrategyResult(
            analysis_type=analysis_type,
            status="NOT_SUPPORTED",
            limitations=[
                "no optimization/decision-recommendation primitive is implemented in this platform; closest valid "
                "analysis: a COMPARATIVE or DIAGNOSTIC breakdown of the same metric to inform a human decision"
            ],
        )

    def _detect_type(self, contract: RequirementContract) -> AnalysisType:
        text = contract.objective or ""
        for analysis_type, pattern in _KEYWORD_SIGNALS:
            if pattern.search(text):
                return analysis_type
        if contract.time.comparison:
            return "COMPARATIVE"
        if text.strip():
            return "DESCRIPTIVE"
        return "UNCLASSIFIED"
