"""Visualization Planner.

Source of truth: AI Prompt Library Section 17 "Visualization Planner":
"Design visualizations that faithfully represent verified data and the
user's decision context... Never alter or recompute verified metric
values... Avoid misleading axes/scales, truncated baselines when
inappropriate, 3D distortion, and unnecessary dual axes... If a visual
requires an aggregate not already verified, request a deterministic
computation step rather than calculating it yourself."

Deterministic code, not an LLM call - which chart type fits a metric's
already-profiled output shape (a date column present -> a time series; a
categorical column present -> a comparison; neither -> a single value) is
a provable fact about `DatasetProfile` evidence (Section 6), never a
judgment call, and this planner never touches the underlying values at
all - it only ever reads column names/dtypes/row_count off a profile that
was itself computed from a RELEASED run's output artifact.

Only three chart types are ever emitted (`line`, `bar`, `stat`), and every
visual is exactly one metric's own output - "avoid ... 3D distortion, and
unnecessary dual axes" is satisfied by construction rather than by a
runtime check, since nothing here ever proposes a 3D chart or combines
two metrics' scales onto one axis.

Two checks flag a risk without ever fabricating the fix (Constitution
rule 8, "never convert uncertainty into false precision"):
  - a profile with an implausibly large row count for a metric's
    *verified aggregate* output is exactly Section 17's "requires an
    aggregate not already verified" case - flagged as a warning asking
    for an explicit AGGREGATE step, never aggregated here.
  - a column that reads as a forecast/prediction with no paired
    uncertainty-bound column violates "forecasts must ... include
    uncertainty when available" - flagged, never silently accepted.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from dataos.compiler.narrative import narrate
from dataos.compiler.prompts import VISUALIZATION_PLANNER_SYSTEM_PROMPT
from dataos.contracts.requirement_contract import RequirementContract
from dataos.ingestion.profiling import DatasetProfile
from dataos.llm.client import LLMClient

ChartType = Literal["line", "bar", "stat"]

_DATE_DTYPE_PATTERN = re.compile("Date")
_NUMERIC_DTYPE_PATTERN = re.compile(r"(Int|UInt|Float|Decimal)")
_MONEY_TERM_PATTERN = re.compile(
    r"(revenue|amount|sales|price|cost|total|balance|payment|spend|margin|profit)", re.I
)
_FORECAST_COLUMN_PATTERN = re.compile(r"(forecast|predicted)", re.I)
_UNCERTAINTY_COLUMN_PATTERN = re.compile(r"(lower[_-]?bound|upper[_-]?bound|confidence|interval|_ci$)", re.I)

# Above this many rows, a metric's output no longer looks like a
# released, verified aggregate result - it looks like raw/unaggregated
# detail that was never actually rolled up. Purely a sanity threshold,
# documented as heuristic; it only ever produces a warning, never blocks.
_LIKELY_UNAGGREGATED_ROW_COUNT = 5000


class VisualSpec(BaseModel):
    title: str
    chart_type: ChartType
    dataset_artifact: str
    x: str | None = None
    y: list[str] = Field(default_factory=list)
    series: str | None = None
    filters: list[str] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    accessibility_note: str


class VisualizationPlanResult(BaseModel):
    visuals: list[VisualSpec] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    narrative: str | None = None
    """Optional human-readable elaboration from an LLM (see module
    docstring) - never authoritative; `visuals`/`warnings` remain the
    decision."""


class VisualizationPlanner:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    def plan(
        self,
        *,
        contract: RequirementContract,
        profiles: dict[str, DatasetProfile],
    ) -> VisualizationPlanResult:
        """`profiles` keys are metric names, mapped to a `DatasetProfile`
        of that metric's already-RELEASED output artifact - never a raw
        source. A metric with no entry is simply not visualized, rather
        than guessed at."""
        visuals: list[VisualSpec] = []
        warnings: list[str] = []

        metrics_by_name = {m.name: m for m in contract.metrics}
        for dataset_artifact, profile in profiles.items():
            metric = metrics_by_name.get(dataset_artifact)
            visual, metric_warnings = self._plan_one(dataset_artifact, profile, contract, metric)
            visuals.append(visual)
            warnings.extend(metric_warnings)

        narrative_text = None
        if self._llm_client is not None:
            narratives = narrate(
                self._llm_client,
                system_prompt=VISUALIZATION_PLANNER_SYSTEM_PROMPT,
                items=[
                    {
                        "ref": "summary",
                        "facts": {"visuals": [v.model_dump() for v in visuals], "warnings": warnings},
                    }
                ],
            )
            narrative_text = narratives.get("summary")

        return VisualizationPlanResult(visuals=visuals, warnings=warnings, narrative=narrative_text)

    def _plan_one(
        self,
        dataset_artifact: str,
        profile: DatasetProfile,
        contract: RequirementContract,
        metric,
    ) -> tuple[VisualSpec, list[str]]:
        warnings: list[str] = []

        date_columns = [c.name for c in profile.columns if _DATE_DTYPE_PATTERN.search(c.dtype)]
        numeric_columns = [c.name for c in profile.columns if _NUMERIC_DTYPE_PATTERN.search(c.dtype)]
        categorical_columns = [
            c.name for c in profile.columns if c.name not in date_columns and c.name not in numeric_columns
        ]

        if metric is not None and metric.name in numeric_columns:
            y = [metric.name]
        else:
            y = numeric_columns

        if date_columns:
            chart_type: ChartType = "line"
            x: str | None = date_columns[0]
        elif categorical_columns:
            chart_type = "bar"
            x = categorical_columns[0]
        else:
            chart_type = "stat"
            x = None

        series = categorical_columns[1] if x in categorical_columns and len(categorical_columns) > 1 else None

        annotations = [
            f"period: {contract.time.range or 'not specified'}",
            f"timezone: {contract.time.timezone or 'not specified'}",
            f"currency: {contract.units.currency or 'not specified'}",
            f"population: {contract.population or 'not specified'}",
        ]

        if profile.row_count > _LIKELY_UNAGGREGATED_ROW_COUNT:
            warnings.append(
                f"'{dataset_artifact}' has {profile.row_count} row(s) - this does not look like a verified "
                "aggregate; request an explicit AGGREGATE step before charting it, rather than treating raw rows as the metric"
            )

        if metric is not None and _MONEY_TERM_PATTERN.search(metric.name) and not contract.units.currency:
            warnings.append(f"'{metric.name}' reads as a monetary metric but no currency is declared - label the axis/legend once one is set")

        forecast_columns = [c for c in numeric_columns if _FORECAST_COLUMN_PATTERN.search(c)]
        uncertainty_columns = [c.name for c in profile.columns if _UNCERTAINTY_COLUMN_PATTERN.search(c.name)]
        if forecast_columns and not uncertainty_columns:
            warnings.append(
                f"'{dataset_artifact}' has forecast/predicted column(s) {forecast_columns} but no paired "
                "uncertainty-bound column - forecasts must include uncertainty when available"
            )

        accessibility_note = (
            f"provide alt text describing the {chart_type} "
            f"{'trend over time' if chart_type == 'line' else 'comparison across categories' if chart_type == 'bar' else 'value'} "
            "and a data table fallback for screen readers"
        )

        visual = VisualSpec(
            title=metric.name if metric is not None else dataset_artifact,
            chart_type=chart_type,
            dataset_artifact=dataset_artifact,
            x=x,
            y=y,
            series=series,
            filters=list(contract.filters),
            annotations=annotations,
            accessibility_note=accessibility_note,
        )
        return visual, warnings
