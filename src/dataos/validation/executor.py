"""Validation Rule Executor.

Executes the `CheckSpec` manifest `ValidationRuleGenerator` (Prompt
Library Section 19) produces against the real per-stage dataframes a run
actually computed, folding every result into a `ValidationEngine` report -
closing the loop from "generate the rules" to "prove they held," and
implementing the "Validation & Reconciliation" stage of the Prompt
Library's "Agent pipeline" (between the Deterministic Executor and the
Independent Verifier).

A `CheckSpec.check_function` (when set) is always one of the plain
functions in `dataos.validation.checks`, resolved here by name - never
imported ad hoc, so the set of things this executor can ever run is
exactly what `ValidationRuleGenerator` already validated exists. Only the
single-dataframe-shaped functions the generator actually produces today
(`check_null_rate`, `check_no_duplicate_rows`, `check_dual_computation`)
are supported; anything else raises rather than being called with the
wrong argument shape by guesswork - `check_conservation`,
`check_reconciliation`, and the join-safety checks all need a *pair* of
inputs (before/after frames, or two floats) that a single `stage ->
dataframe` mapping cannot supply, so wiring those in needs a real design
decision, not a silent best-effort call.

A `CheckSpec` with `check_function=None` names a check a registry
Operation already enforces structurally (e.g. join-key uniqueness) -
execution never reaches a COMPLETED step whose structural guarantee was
violated, so this executor records it as a passing, self-evidencing check
rather than re-running it as a second, possibly-divergent implementation.
"""

from __future__ import annotations

import polars as pl

import dataos.validation.checks as checks
from dataos.compiler.validation_rule_generator import CheckSpec
from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import ValidationResult
from dataos.validation.engine import ValidationEngine, ValidationReport

_SINGLE_FRAME_CHECKS = frozenset({"check_null_rate", "check_no_duplicate_rows", "check_dual_computation"})


class ValidationRuleExecutor:
    def execute(self, rules: list[CheckSpec], dataframes: dict[str, pl.DataFrame]) -> ValidationReport:
        engine = ValidationEngine()
        for rule in rules:
            engine.record(self._run_one(rule, dataframes))
        return engine.report()

    def _run_one(self, rule: CheckSpec, dataframes: dict[str, pl.DataFrame]) -> ValidationResult:
        if rule.check_function is None:
            return ValidationResult(
                check_id=rule.check_id,
                observed="enforced structurally by the operation that produced this stage",
                expected="enforced structurally by the operation that produced this stage",
                passed=True,
                severity=rule.severity,
            )

        if rule.check_function not in _SINGLE_FRAME_CHECKS:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                (
                    f"ValidationRuleExecutor does not (yet) support running "
                    f"'{rule.check_function}' - it needs more than one dataframe input"
                ),
                evidence={"check_id": rule.check_id, "check_function": rule.check_function},
            )

        df = dataframes.get(rule.stage)
        if df is None:
            return ValidationResult(
                check_id=rule.check_id,
                observed=None,
                expected="a dataframe for this check's stage",
                passed=False,
                severity=rule.severity,
            )

        check_fn = getattr(checks, rule.check_function)
        try:
            return check_fn(df, check_id=rule.check_id, severity=rule.severity, **rule.check_args)
        except Exception as exc:  # noqa: BLE001 - one bad check must never crash the whole validation pass
            return ValidationResult(
                check_id=rule.check_id,
                observed=f"check raised {type(exc).__name__}: {exc}",
                expected="the check to run without error",
                passed=False,
                severity=rule.severity,
            )
