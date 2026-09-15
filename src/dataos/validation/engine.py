"""Validation Engine.

Source of truth: `validation/__init__.py`'s module docstring - this
package's checks are plain functions returning `ValidationResult` rather
than a declarative "CheckSpec" DSL, so the engine's only job is to
aggregate results callers already computed into one `ValidationReport`.
Blueprint Section 10's pyramid and Section 9.2's state table ("VERIFYING
all mandatory checks pass RELEASED / any mandatory check fails
QUARANTINED") are what a later phase's Independent Verifier / Release
Gate reads a `ValidationReport` to decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataos.evidence.models import ValidationResult


@dataclass
class ValidationReport:
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[ValidationResult]:
        return [r for r in self.results if r.severity == "BLOCKING" and not r.passed]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [r for r in self.results if r.severity == "WARNING" and not r.passed]

    @property
    def passed(self) -> bool:
        return len(self.blocking_failures) == 0


class ValidationEngine:
    """Collects `ValidationResult`s produced by the plain check functions
    in `validation/checks/*` into a single report. It does not know how to
    run any specific check - callers invoke those functions themselves and
    hand the results here.
    """

    def __init__(self) -> None:
        self._results: list[ValidationResult] = []

    def record(self, result: ValidationResult) -> None:
        self._results.append(result)

    def record_all(self, results: list[ValidationResult]) -> None:
        self._results.extend(results)

    def report(self) -> ValidationReport:
        return ValidationReport(results=list(self._results))
