"""A deterministic, rule-based reference LLMClient.

This is explicitly a test double, not a real natural-language
understanding engine. It exists so the compiler pipeline (Requirement
Compiler -> Ambiguity Gate -> Semantic Resolver -> Workflow Planner) can
be built and tested today, with zero network access, before any real
model is wired in (AnthropicLLMClient, added alongside this, is
unused/untested until an API key is supplied).

Deliberately "dumb" for both response models it supports:
  - `RawExtraction`: every extracted metric term is reported with no
    claimed confidence beyond bare extraction. It never decides whether a
    term is governed, ambiguous, or safe to proceed with - that decision
    belongs entirely to the deterministic SemanticResolver/AmbiguityGate
    downstream (AI Prompt Library Section 3: "Do not replace the user's
    business meaning with generic analytics defaults").
  - `RawPlan`: only proposes a single `aggregate` step, and only for
    metrics whose already-resolved formula matches a plain `fn(table.col)`
    shape - it does not attempt joins, cleaning, or multi-step plans.
Using genuinely naive rules here keeps the tests honest: they exercise the
pipeline's safety logic (registry validation, DAG validation, ambiguity
blocking), not a hand-tuned fake AI.
"""

from __future__ import annotations

import json
import re
import uuid

from dataos.compiler.extraction import ExtractedMetric, RawExtraction
from dataos.compiler.raw_plan import RawPlan, RawPlanStep
from dataos.errors import ErrorCode, PlatformError
from dataos.llm.client import LLMClient, T
from dataos.workflow.dsl import SOURCE_PREFIX

_STOPWORDS = {
    "the", "a", "an", "of", "for", "by", "and", "or", "to", "in", "on",
    "show", "report", "give", "me", "please", "get", "list", "calculate",
    "what", "is", "are", "was", "were", "our", "per",
}

# A metric noun is heuristically anything after "show/report/calculate/get"
# up to a "by"/"for"/end-of-sentence, or any known business-metric-ish
# word. This is intentionally shallow - see module docstring.
_METRIC_TRIGGER = re.compile(
    r"\b(?:show|report|give me|calculate|get|list)\s+(?:the\s+)?(.+?)(?:\s+by\b|\s+for\b|\s*$)",
    re.IGNORECASE,
)
_BY_CLAUSE = re.compile(r"\bby\s+([a-zA-Z_][\w\- ]*)", re.IGNORECASE)
_TIMEZONE_HINT = re.compile(r"\b(UTC|GMT|[A-Za-z_]+/[A-Za-z_]+)\b")

# Only a bare `fn(table.column)` formula is understood - compound formulas
# like "sum(orders.net_amount) - sum(refunds.amount)" are left unplanned,
# the same "leave it to a real model" boundary the extraction side draws.
_SIMPLE_METRIC_FORMULA = re.compile(r"^(sum|count|mean|min|max|n_unique)\(\w+\.(\w+)\)$")

_SUPPORTED_RESPONSE_MODELS = (RawExtraction, RawPlan)


class DeterministicLLMClient(LLMClient):
    def complete_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        if response_model is RawExtraction:
            return self._extract(user_prompt)  # type: ignore[return-value]
        if response_model is RawPlan:
            return self._plan(user_prompt)  # type: ignore[return-value]
        raise PlatformError(
            ErrorCode.MODEL_OUTPUT_INVALID,
            (
                f"DeterministicLLMClient is a reference/test double and only supports "
                f"response_model in {[m.__name__ for m in _SUPPORTED_RESPONSE_MODELS]}, got {response_model!r}"
            ),
        )

    def _extract(self, text: str) -> RawExtraction:
        objective = text.strip()

        metric_terms: list[str] = []
        trigger_match = _METRIC_TRIGGER.search(text)
        if trigger_match:
            candidate = trigger_match.group(1).strip().lower()
            words = [w for w in re.findall(r"[a-zA-Z_]+", candidate) if w not in _STOPWORDS]
            if words:
                metric_terms.append(" ".join(words))

        group_by = [g.strip().lower() for g in _BY_CLAUSE.findall(text)]

        timezone_match = _TIMEZONE_HINT.search(text)
        timezone = timezone_match.group(1) if timezone_match else None

        return RawExtraction(
            objective=objective,
            metrics=[ExtractedMetric(term=t) for t in metric_terms],
            dimensions=[],
            filters=[],
            group_by=group_by,
            time_range=None,
            timezone=timezone,
        )

    def _plan(self, context_json: str) -> RawPlan:
        """`context_json` is the compact JSON rendered by
        `WorkflowPlanner._render_planning_context` - not free NL text, since
        planning is a structured-input task, unlike raw extraction."""
        context = json.loads(context_json)
        sources = context.get("sources") or []

        metric_specs = []
        for metric in context.get("metrics", []):
            formula = (metric.get("formula") or "").strip()
            match = _SIMPLE_METRIC_FORMULA.match(formula)
            if not match:
                continue
            fn, column = match.groups()
            metric_specs.append({"name": metric["name"], "column": column, "fn": fn})

        plan_id = str(uuid.uuid4())
        if not sources or not metric_specs:
            return RawPlan(plan_id=plan_id, steps=[])

        step = RawPlanStep(
            id="s1",
            type="AGGREGATE",
            operation_id="aggregate",
            operation_version="1.0",
            inputs=[f"{SOURCE_PREFIX}{sources[0]}"],
            params={"group_by": context.get("group_by", []), "metrics": metric_specs},
            requirement_refs=[m["name"] for m in metric_specs],
        )
        return RawPlan(
            plan_id=plan_id,
            steps=[step],
            final_acceptance_tests=context.get("acceptance_tests", []),
        )
