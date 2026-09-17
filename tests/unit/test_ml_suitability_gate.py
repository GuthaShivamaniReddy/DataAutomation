from datetime import datetime, timedelta

import polars as pl

from dataos.compiler.ml_suitability_gate import MLSuitabilityGate, MLSuitabilityRequest
from dataos.ingestion.profiling import profile_dataset


def _big_frame(n: int = 150) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": list(range(n)),
            "tenure_days": [i % 30 for i in range(n)],
            "churned": [1 if i % 10 == 0 else 0 for i in range(n)],
        }
    )


def test_missing_target_column_is_rejected():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="does_not_exist", feature_columns=["tenure_days"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "REJECT"
    assert any("does not exist" in r for r in result.risks)


def test_target_used_as_a_feature_is_rejected_as_leakage():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["churned", "tenure_days"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "REJECT"
    assert any("leakage" in r for r in result.risks)


def test_missing_feature_column_is_rejected():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["nonexistent_feature"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "REJECT"
    assert any("nonexistent_feature" in r for r in result.risks)


def test_small_sample_size_is_rejected():
    df = _big_frame(n=10)
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["tenure_days"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "REJECT"
    assert any("sample size" in r for r in result.risks)


def test_high_target_null_rate_is_rejected():
    n = 150
    df = pl.DataFrame({"id": list(range(n)), "label": [None if i % 2 == 0 else 1 for i in range(n)]})
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="label", feature_columns=["id"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "REJECT"
    assert any("insufficient label availability" in r for r in result.risks)


def test_forecasting_without_time_column_is_rejected():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="forecasting", target_column="churned", feature_columns=["tenure_days"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "REJECT"
    assert any("time_column" in r for r in result.risks)


def test_forecasting_with_a_non_date_time_column_is_rejected():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(
        task="forecasting", target_column="churned", feature_columns=["tenure_days"], time_column="customer_id"
    )

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "REJECT"
    assert any("not a date/datetime column" in r for r in result.risks)


def test_forecasting_with_a_real_date_column_can_proceed():
    n = 150
    start = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "day": [start + timedelta(days=i) for i in range(n)],
            "revenue": [float(i) for i in range(n)],
        }
    )
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="forecasting", target_column="revenue", feature_columns=["day"], time_column="day")

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision in ("PROCEED", "LIMITED")


def test_class_imbalance_check_without_a_frame_is_a_limited_risk_not_a_rejection():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["tenure_days"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.decision == "LIMITED"
    assert any("class imbalance could not be verified" in r for r in result.risks)


def test_severe_class_imbalance_is_detected_with_a_real_frame():
    n = 200
    df = pl.DataFrame({"id": list(range(n)), "label": [1 if i == 0 else 0 for i in range(n)]})
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="label", feature_columns=["id"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile, frame=df)

    assert result.decision == "LIMITED"
    assert any("severe class imbalance" in r for r in result.risks)


def test_protected_attribute_feature_is_flagged():
    df = _big_frame()
    profile = profile_dataset(df.with_columns(pl.lit("F").alias("gender")))
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["tenure_days", "gender"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile, frame=df.with_columns(pl.lit("F").alias("gender")))

    assert any("protected attribute" in r for r in result.risks)


def test_structural_no_training_primitive_risk_is_always_present():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["tenure_days"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile, frame=df)

    assert any("no model-training primitive" in r for r in result.risks)


def test_narrative_is_none_without_an_llm_client():
    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["tenure_days"])

    result = MLSuitabilityGate().evaluate(request=request, profile=profile)

    assert result.narrative is None


def test_narrative_is_populated_without_changing_the_decision_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    df = _big_frame()
    profile = profile_dataset(df)
    request = MLSuitabilityRequest(task="classification", target_column="churned", feature_columns=["tenure_days"])

    gate = MLSuitabilityGate(llm_client=DeterministicLLMClient())
    result = gate.evaluate(request=request, profile=profile)

    assert result.narrative is not None
    assert result.decision == "LIMITED"
