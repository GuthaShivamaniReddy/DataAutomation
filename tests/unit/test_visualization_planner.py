from datetime import datetime, timedelta

import polars as pl

from dataos.compiler.visualization_planner import VisualizationPlanner
from dataos.contracts.requirement_contract import Metric, RequirementContract, Source, TimeSpec, UnitsPolicy
from dataos.ingestion.profiling import profile_dataset


def _contract(**kwargs) -> RequirementContract:
    kwargs.setdefault("objective", "x")
    kwargs.setdefault("sources", [Source(name="orders")])
    return RequirementContract(**kwargs)


def test_time_series_output_becomes_a_line_chart():
    start = datetime(2024, 1, 1)
    df = pl.DataFrame({"day": [start + timedelta(days=i) for i in range(5)], "revenue": [1.0, 2.0, 3.0, 4.0, 5.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    visual = result.visuals[0]
    assert visual.chart_type == "line"
    assert visual.x == "day"
    assert visual.y == ["revenue"]


def test_categorical_output_becomes_a_bar_chart():
    df = pl.DataFrame({"region": ["East", "West"], "revenue": [10.0, 20.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    visual = result.visuals[0]
    assert visual.chart_type == "bar"
    assert visual.x == "region"


def test_scalar_output_becomes_a_stat_tile():
    df = pl.DataFrame({"revenue": [42.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    visual = result.visuals[0]
    assert visual.chart_type == "stat"
    assert visual.x is None


def test_second_categorical_column_becomes_the_series():
    df = pl.DataFrame({"region": ["East", "West"], "channel": ["web", "store"], "revenue": [10.0, 20.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    assert result.visuals[0].series == "channel"


def test_annotations_include_period_currency_and_population():
    df = pl.DataFrame({"revenue": [42.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(
        metrics=[Metric(name="revenue")],
        time=TimeSpec(range="last 30 days"),
        units=UnitsPolicy(currency="USD"),
        population="active customers",
    )

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    annotations = result.visuals[0].annotations
    assert any("last 30 days" in a for a in annotations)
    assert any("USD" in a for a in annotations)
    assert any("active customers" in a for a in annotations)


def test_large_row_count_warns_about_an_unverified_aggregate():
    df = pl.DataFrame({"revenue": [float(i) for i in range(6000)]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    assert any("verified aggregate" in w for w in result.warnings)


def test_money_metric_without_currency_warns():
    df = pl.DataFrame({"revenue": [42.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    assert any("currency" in w for w in result.warnings)


def test_forecast_column_without_uncertainty_bound_warns():
    df = pl.DataFrame({"day": [datetime(2024, 1, 1)], "forecast_revenue": [42.0]})
    profiles = {"forecast_revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="forecast_revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    assert any("uncertainty-bound" in w for w in result.warnings)


def test_forecast_column_with_uncertainty_bound_does_not_warn():
    df = pl.DataFrame(
        {
            "day": [datetime(2024, 1, 1)],
            "forecast_revenue": [42.0],
            "lower_bound": [40.0],
            "upper_bound": [44.0],
        }
    )
    profiles = {"forecast_revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="forecast_revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    assert not any("uncertainty-bound" in w for w in result.warnings)


def test_narrative_is_none_without_an_llm_client():
    df = pl.DataFrame({"revenue": [42.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    result = VisualizationPlanner().plan(contract=contract, profiles=profiles)

    assert result.narrative is None


def test_narrative_is_populated_without_changing_visuals_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    df = pl.DataFrame({"revenue": [42.0]})
    profiles = {"revenue": profile_dataset(df)}
    contract = _contract(metrics=[Metric(name="revenue")])

    planner = VisualizationPlanner(llm_client=DeterministicLLMClient())
    result = planner.plan(contract=contract, profiles=profiles)

    assert result.narrative is not None
    assert result.visuals[0].chart_type == "stat"
