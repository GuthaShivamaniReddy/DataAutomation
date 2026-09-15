"""Deterministic dataset profiling.

Source of truth: Reliability-First Master Blueprint Section 5 ("Schema:
physical type, logical type, nullability, candidate keys, cardinality,
ranges" / "Quality: missingness, duplicates, outliers, invalid categories,
malformed dates, impossible values") and AI Prompt Library Section 6
"Dataset Profiling Agent".

This is pure deterministic code - no LLM involved, per the blueprint's own
build order (Section 28.1): profiling exists before any AI planner does.
A later phase's Dataset Profiling Agent prompt consumes exactly this
output as "OBSERVED facts" and is barred from inferring statistics that
were not computed here.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_rate: float
    distinct_count: int
    is_candidate_key: bool
    min_value: str | None = None
    max_value: str | None = None


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    duplicate_row_count: int
    candidate_key_columns: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


_NUMERIC_OR_TEMPORAL_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
    pl.Date, pl.Datetime, pl.Time,
)


def profile_dataset(df: pl.DataFrame) -> DatasetProfile:
    row_count = df.height
    columns: list[ColumnProfile] = []
    candidate_keys: list[str] = []

    for name in df.columns:
        series = df[name]
        null_count = int(series.null_count())
        null_rate = (null_count / row_count) if row_count else 0.0
        distinct_count = int(series.n_unique())
        # Candidate key: unique and fully populated at this grain. An
        # entirely-empty column is never a candidate key.
        is_candidate_key = row_count > 0 and null_count == 0 and distinct_count == row_count

        min_value = max_value = None
        if isinstance(series.dtype, _NUMERIC_OR_TEMPORAL_DTYPES) and null_count < row_count:
            mn, mx = series.min(), series.max()
            min_value = str(mn) if mn is not None else None
            max_value = str(mx) if mx is not None else None

        columns.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                null_count=null_count,
                null_rate=null_rate,
                distinct_count=distinct_count,
                is_candidate_key=is_candidate_key,
                min_value=min_value,
                max_value=max_value,
            )
        )
        if is_candidate_key:
            candidate_keys.append(name)

    duplicate_row_count = 0
    if row_count > 0 and df.columns:
        duplicate_row_count = row_count - df.unique().height

    return DatasetProfile(
        row_count=row_count,
        column_count=len(df.columns),
        columns=columns,
        duplicate_row_count=duplicate_row_count,
        candidate_key_columns=candidate_keys,
    )
