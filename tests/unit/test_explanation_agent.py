import pytest

from dataos.compiler.explanation_agent import ExplanationAgent
from dataos.compiler.raw_explanation import Finding, RawExplanation
from dataos.errors import ErrorCode, PlatformError
from dataos.llm.client import LLMClient
from dataos.llm.deterministic import DeterministicLLMClient


def _evidence_context() -> dict:
    return {
        "objective": "show order count",
        "grain": None,
        "filters": [],
        "time": {"range": None, "timezone": None, "comparison": None, "fiscal_calendar": None},
        "units": {"currency": None, "measurement": None},
        "metrics": [
            {
                "name": "order_count",
                "formula": "count(orders.order_id)",
                "step_id": "s1",
                "sample_records": [{"order_count": 5}],
            }
        ],
    }


def test_deterministic_double_produces_a_grounded_finding():
    agent = ExplanationAgent(DeterministicLLMClient())
    output = agent.explain(
        run_id="run_1",
        evidence_context=_evidence_context(),
        known_evidence_refs=frozenset({"order_count", "s1"}),
    )

    assert output.explanation.findings[0].statement == "order_count is 5."
    assert set(output.explanation.findings[0].evidence_refs) == {"order_count", "s1"}
    assert output.envelope.status == "OK"
    assert output.envelope.result["workflow_run_id"] == "run_1"


def test_forced_limitations_are_appended_even_if_model_omits_them():
    agent = ExplanationAgent(DeterministicLLMClient())
    output = agent.explain(
        run_id="run_1",
        evidence_context=_evidence_context(),
        known_evidence_refs=frozenset({"order_count", "s1"}),
        forced_limitations=["some independent check could not be verified"],
    )

    assert "some independent check could not be verified" in output.explanation.limitations


class _FakeLLMClient(LLMClient):
    """Returns a fixed RawExplanation citing evidence it was never given,
    to exercise the deterministic evidence_refs guardrail."""

    model_id = "fake/hallucinating-explainer"

    def complete_structured(self, *, system_prompt, user_prompt, response_model):
        return RawExplanation(
            summary="Revenue grew 20%.",
            findings=[Finding(statement="Revenue grew 20%.", type="FACT", evidence_refs=["revenue_growth"])],
        )


def test_finding_citing_unsupplied_evidence_is_rejected():
    agent = ExplanationAgent(_FakeLLMClient())

    with pytest.raises(PlatformError) as excinfo:
        agent.explain(
            run_id="run_1",
            evidence_context=_evidence_context(),
            known_evidence_refs=frozenset({"order_count", "s1"}),
        )

    assert excinfo.value.code == ErrorCode.MODEL_OUTPUT_INVALID
    assert excinfo.value.evidence["unknown_evidence_refs"] == ["revenue_growth"]
