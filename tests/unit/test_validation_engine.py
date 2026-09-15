from dataos.evidence.models import ValidationResult
from dataos.validation.engine import ValidationEngine


def test_report_passes_when_no_blocking_failures():
    engine = ValidationEngine()
    engine.record(ValidationResult(check_id="a", observed=1, expected=1, passed=True, severity="BLOCKING"))
    engine.record(ValidationResult(check_id="b", observed=2, expected=1, passed=False, severity="WARNING"))

    report = engine.report()

    assert report.passed is True
    assert report.blocking_failures == []
    assert len(report.warnings) == 1


def test_report_fails_on_blocking_failure():
    engine = ValidationEngine()
    engine.record_all(
        [
            ValidationResult(check_id="a", observed=1, expected=1, passed=True, severity="BLOCKING"),
            ValidationResult(check_id="b", observed=0, expected=1, passed=False, severity="BLOCKING"),
        ]
    )

    report = engine.report()

    assert report.passed is False
    assert [r.check_id for r in report.blocking_failures] == ["b"]
