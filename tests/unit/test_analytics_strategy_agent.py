from dataos.compiler.analytics_strategy_agent import AnalyticsStrategyAgent
from dataos.contracts.requirement_contract import RequirementContract, Source, TimeSpec


def _contract(**kwargs) -> RequirementContract:
    kwargs.setdefault("objective", "show revenue by month")
    kwargs.setdefault("sources", [Source(name="orders")])
    return RequirementContract(**kwargs)


def test_plain_show_request_is_descriptive_and_approved():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="show total revenue"))

    assert result.analysis_type == "DESCRIPTIVE"
    assert result.status == "APPROVED"


def test_why_question_is_diagnostic_and_flags_correlation_limitation():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="why did revenue drop last month"))

    assert result.analysis_type == "DIAGNOSTIC"
    assert result.status == "APPROVED"
    assert any("causal" in lim.lower() for lim in result.limitations)


def test_comparison_request_with_declared_period_is_approved():
    result = AnalyticsStrategyAgent().classify(
        contract=_contract(objective="compare revenue to last quarter", time=TimeSpec(comparison="QoQ"))
    )

    assert result.analysis_type == "COMPARATIVE"
    assert result.status == "APPROVED"


def test_comparison_request_without_declared_period_needs_clarification():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="compare revenue versus last quarter"))

    assert result.analysis_type == "COMPARATIVE"
    assert result.status == "NEEDS_CLARIFICATION"


def test_segmentation_request_with_dimensions_is_approved():
    result = AnalyticsStrategyAgent().classify(
        contract=_contract(objective="segment customers by cohort", dimensions=["region"])
    )

    assert result.analysis_type == "SEGMENTATION"
    assert result.status == "APPROVED"


def test_segmentation_request_without_dimensions_needs_clarification():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="segment customers into cohorts"))

    assert result.analysis_type == "SEGMENTATION"
    assert result.status == "NEEDS_CLARIFICATION"


def test_causal_request_is_not_supported_with_a_fallback_suggestion():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="what is the effect of price on churn"))

    assert result.analysis_type == "CAUSAL"
    assert result.status == "NOT_SUPPORTED"
    assert any("DIAGNOSTIC" in lim for lim in result.limitations)


def test_forecast_request_is_not_supported_with_a_fallback_suggestion():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="forecast next quarter revenue"))

    assert result.analysis_type == "FORECASTING"
    assert result.status == "NOT_SUPPORTED"
    assert any("DESCRIPTIVE" in lim for lim in result.limitations)


def test_predictive_request_is_not_supported():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="predict which customers will churn"))

    assert result.analysis_type == "PREDICTIVE"
    assert result.status == "NOT_SUPPORTED"


def test_anomaly_request_is_not_supported_with_a_fallback_suggestion():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="find unusual spikes in daily signups"))

    assert result.analysis_type == "ANOMALY_DETECTION"
    assert result.status == "NOT_SUPPORTED"
    assert any("select_filter" in lim for lim in result.limitations)


def test_prescriptive_request_is_not_supported():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective="what should we do to increase revenue"))

    assert result.analysis_type == "PRESCRIPTIVE"
    assert result.status == "NOT_SUPPORTED"


def test_causal_signal_takes_precedence_over_comparative_signal():
    result = AnalyticsStrategyAgent().classify(
        contract=_contract(objective="what is the causal effect of price compared to last quarter")
    )

    assert result.analysis_type == "CAUSAL"


def test_empty_objective_is_unclassified_and_needs_clarification():
    result = AnalyticsStrategyAgent().classify(contract=_contract(objective=""))

    assert result.analysis_type == "UNCLASSIFIED"
    assert result.status == "NEEDS_CLARIFICATION"


def test_narrative_is_none_without_an_llm_client():
    result = AnalyticsStrategyAgent().classify(contract=_contract())

    assert result.narrative is None


def test_narrative_is_populated_without_changing_the_classification_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    agent = AnalyticsStrategyAgent(llm_client=DeterministicLLMClient())
    result = agent.classify(contract=_contract())

    assert result.narrative is not None
    assert result.analysis_type == "DESCRIPTIVE"
