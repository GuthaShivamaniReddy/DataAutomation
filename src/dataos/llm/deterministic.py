"""A deterministic, rule-based reference LLMClient.

This is explicitly a test double, not a real natural-language
understanding engine. It exists so the compiler pipeline (Requirement
Compiler -> Ambiguity Gate -> Semantic Resolver) can be built and tested
today, with zero network access, before any real model is wired in
(AnthropicLLMClient, added alongside this, is unused/untested until an
API key is supplied).

Deliberately "dumb": every extracted metric term is reported with no
claimed confidence beyond bare extraction. It never decides whether a
term is governed, ambiguous, or safe to proceed with - that decision
belongs entirely to the deterministic SemanticResolver/AmbiguityGate
downstream (AI Prompt Library Section 3: "Do not replace the user's
business meaning with generic analytics defaults"). Using a genuinely
naive extractor here keeps the tests honest: they exercise the
pipeline's safety logic, not a hand-tuned fake AI.
"""

from __future__ import annotations

import re

from dataos.compiler.extraction import ExtractedMetric, RawExtraction
from dataos.errors import ErrorCode, PlatformError
from dataos.llm.client import LLMClient, T

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


class DeterministicLLMClient(LLMClient):
    def complete_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        if response_model is not RawExtraction:
            raise PlatformError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                (
                    f"DeterministicLLMClient is a reference/test double and only supports "
                    f"response_model=RawExtraction, got {response_model!r}"
                ),
            )
        extraction = self._extract(user_prompt)
        return extraction  # type: ignore[return-value]

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
