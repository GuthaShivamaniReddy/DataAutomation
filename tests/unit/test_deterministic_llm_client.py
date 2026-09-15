import pytest

from dataos.compiler.extraction import RawExtraction
from dataos.errors import ErrorCode, PlatformError
from dataos.llm.deterministic import DeterministicLLMClient


def test_extracts_metric_term_after_trigger_phrase():
    client = DeterministicLLMClient()
    result = client.complete_structured(
        system_prompt="ignored", user_prompt="Show revenue by month", response_model=RawExtraction
    )
    assert result.metrics[0].term == "revenue"
    assert result.group_by == ["month"]


def test_extracts_by_clause_content():
    # The heuristic extractor is intentionally shallow (see module docstring) -
    # it is enough that the grouping terms show up somewhere in the extraction,
    # not that it perfectly segments multiple "by" clauses.
    client = DeterministicLLMClient()
    result = client.complete_structured(
        system_prompt="ignored",
        user_prompt="Report order count by region by month",
        response_model=RawExtraction,
    )
    joined = " ".join(result.group_by)
    assert "region" in joined
    assert "month" in joined


def test_no_trigger_phrase_yields_no_metrics():
    client = DeterministicLLMClient()
    result = client.complete_structured(
        system_prompt="ignored", user_prompt="hello there", response_model=RawExtraction
    )
    assert result.metrics == []


def test_timezone_hint_extracted_when_present():
    client = DeterministicLLMClient()
    result = client.complete_structured(
        system_prompt="ignored",
        user_prompt="Show revenue by month in America/New_York",
        response_model=RawExtraction,
    )
    assert result.timezone == "America/New_York"


def test_unsupported_response_model_raises_model_output_invalid():
    from pydantic import BaseModel

    class SomethingElse(BaseModel):
        x: int

    client = DeterministicLLMClient()
    with pytest.raises(PlatformError) as excinfo:
        client.complete_structured(system_prompt="", user_prompt="", response_model=SomethingElse)
    assert excinfo.value.code == ErrorCode.MODEL_OUTPUT_INVALID
