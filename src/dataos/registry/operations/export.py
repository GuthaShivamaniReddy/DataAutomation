"""export operation.

Blueprint Section 7: preconditions "approved format/destination"; mandatory
evidence "checksum, row count, schema manifest".
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import polars as pl

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact
from dataos.ingestion.hashing import sha256_file
from dataos.registry.base import Operation, OperationResult

_ALLOWED_FORMATS = {"csv", "parquet", "json"}


class ExportOperation(Operation):
    operation_id: ClassVar[str] = "export"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        format: Literal["csv", "parquet", "json"]
        destination_path: str

    def check_preconditions(self, df: pl.DataFrame, params: "ExportOperation.Params") -> list[str]:
        violations = []
        if params.format not in _ALLOWED_FORMATS:
            violations.append(f"unapproved export format: {params.format}")
        if ".." in Path(params.destination_path).parts:
            violations.append("destination_path must not contain '..' path traversal segments")
        return violations

    def run(self, df: pl.DataFrame, params: "ExportOperation.Params") -> OperationResult:
        dest = Path(params.destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if params.format == "csv":
            df.write_csv(dest)
        elif params.format == "parquet":
            df.write_parquet(dest)
        else:
            df.write_json(dest)

        if not dest.exists():
            raise PlatformError(
                ErrorCode.EVIDENCE_INSUFFICIENT,
                f"export claimed success but destination file was not created: {dest}",
            )

        checksum = sha256_file(dest)
        schema_manifest = {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)}

        evidence = {
            "format": params.format,
            "destination_path": str(dest),
            "checksum": checksum,
            "row_count": df.height,
            "schema_manifest": schema_manifest,
        }
        row_impact = RowImpact(rows_in=df.height, rows_out=df.height, rows_rejected=0)
        return OperationResult(output=df, evidence=evidence, row_impact=row_impact)
