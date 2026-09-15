"""File parsing with explicit strict/permissive modes.

Source of truth: Reliability-First Master Blueprint Section 5 ("Parsing:
encoding detection, delimiter/sheet discovery, strict and permissive parse
modes with error report") and Section 11's "Silent partial file parse"
prevention pattern: "parse-error count + rejected-row artifact + release
threshold".

The rule enforced here: a parse can never silently drop malformed rows and
claim a full-data result. In strict mode any malformed row is a hard
failure (DATA_QUALITY_BLOCK). In permissive mode malformed rows are
excluded from the returned frame but are always counted and sampled in the
ParseReport, so a downstream caller can never mistake a partial parse for
a complete one.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from dataos.errors import ErrorCode, PlatformError

ParseMode = Literal["strict", "permissive"]


class ParseReport(BaseModel):
    source_format: str
    rows_seen: int
    rows_parsed: int
    rows_rejected: int = 0
    rejected_sample: list[dict] = Field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        return self.rows_rejected > 0


_REJECTED_SAMPLE_CAP = 20


def parse_csv(path: str | Path, mode: ParseMode = "strict") -> tuple[pl.DataFrame, ParseReport]:
    """Parse a CSV file, validating row shape against the header.

    A row whose field count does not match the header is malformed. Strict
    mode raises; permissive mode excludes it from the frame but reports it.
    """
    path = Path(path)
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        ncols = len(header)

        good_rows: list[list[str]] = []
        rejected: list[dict] = []
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue  # trailing blank line - not a data row, not a malformation
            if ncols == 0 or len(row) != ncols:
                rejected.append({"line": line_no, "raw": row})
            else:
                good_rows.append(row)

    rows_seen = len(good_rows) + len(rejected)

    if rejected and mode == "strict":
        raise PlatformError(
            ErrorCode.DATA_QUALITY_BLOCK,
            f"{len(rejected)} malformed row(s) found in strict parse mode; refusing partial result",
            evidence={"rows_rejected": len(rejected), "rejected_sample": rejected[:_REJECTED_SAMPLE_CAP]},
        )

    if good_rows:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        writer.writerows(good_rows)
        buf.seek(0)
        df = pl.read_csv(buf)
    else:
        df = pl.DataFrame({h: [] for h in header}) if header else pl.DataFrame()

    return df, ParseReport(
        source_format="csv",
        rows_seen=rows_seen,
        rows_parsed=df.height,
        rows_rejected=len(rejected),
        rejected_sample=rejected[:_REJECTED_SAMPLE_CAP],
    )


def parse_xlsx(path: str | Path, mode: ParseMode = "strict", sheet_name: str | None = None) -> tuple[pl.DataFrame, ParseReport]:
    """Parse an XLSX file. `mode` is accepted for interface symmetry with
    parse_csv; XLSX has no ragged-row failure mode in this implementation,
    so strict/permissive behave identically for row shape. A future
    revision can add cell-level validation here.
    """
    df = pl.read_excel(str(path), sheet_name=sheet_name, engine="openpyxl") if sheet_name else pl.read_excel(str(path), engine="openpyxl")
    return df, ParseReport(source_format="xlsx", rows_seen=df.height, rows_parsed=df.height, rows_rejected=0)


def parse_parquet(path: str | Path, mode: ParseMode = "strict") -> tuple[pl.DataFrame, ParseReport]:
    """Parse a Parquet file. Parquet is a typed columnar format with no
    row-shape ambiguity, so there is no malformed-row concept to enforce.
    """
    df = pl.read_parquet(str(path))
    return df, ParseReport(source_format="parquet", rows_seen=df.height, rows_parsed=df.height, rows_rejected=0)
