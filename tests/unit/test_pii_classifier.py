import polars as pl

from dataos.compiler.pii_classifier import PIIClassifier


def _classify(df: pl.DataFrame):
    report = PIIClassifier().classify_dataframe(df)
    return {f.field: f for f in report.fields}


def test_public_column_allows_model_access():
    fields = _classify(pl.DataFrame({"region": ["East", "West"]}))
    f = fields["region"]
    assert f.classification == "public_non_sensitive"
    assert f.model_access == "ALLOW"


def test_ssn_by_name_and_value_is_high_confidence_deny():
    fields = _classify(pl.DataFrame({"ssn": ["123-45-6789", "987-65-4321"]}))
    f = fields["ssn"]
    assert f.classification == "government_identifier"
    assert f.confidence == "HIGH"
    assert f.evidence_type == "schema_metadata+value_pattern"
    assert f.model_access == "DENY"


def test_password_column_name_alone_is_deny():
    fields = _classify(pl.DataFrame({"password": ["not-shaped-like-anything-special"]}))
    f = fields["password"]
    assert f.classification == "authentication_secret"
    assert f.model_access == "DENY"
    assert f.confidence == "MEDIUM"
    assert f.evidence_type == "schema_metadata"


def test_email_shaped_values_with_unrelated_column_name_is_low_confidence_mask():
    fields = _classify(pl.DataFrame({"contact_info": ["a@example.com", "b@example.com"]}))
    f = fields["contact_info"]
    assert f.classification == "contact_data"
    assert f.confidence == "LOW"
    assert f.evidence_type == "value_pattern"
    assert f.model_access == "MASK"


def test_confidence_never_relaxes_model_access_below_mask_for_low_confidence_sensitive_field():
    # Even at LOW confidence, a sensitive classification must never map to ALLOW.
    fields = _classify(pl.DataFrame({"misc": ["a@example.com"]}))
    assert fields["misc"].model_access in ("MASK", "DENY")


def test_conflicting_name_and_value_signals_pick_the_more_restrictive_classification():
    # Column named like contact data (MASK) but values shaped like an SSN (DENY).
    fields = _classify(pl.DataFrame({"phone": ["123-45-6789", "987-65-4321"]}))
    f = fields["phone"]
    assert f.classification == "government_identifier"
    assert f.model_access == "DENY"
    assert f.confidence == "MEDIUM"


def test_non_string_column_is_not_value_matched():
    fields = _classify(pl.DataFrame({"amount": [1.0, 2.0, 3.0]}))
    assert fields["amount"].classification == "public_non_sensitive"


def test_credit_card_shaped_value_is_flagged_payment_data():
    fields = _classify(pl.DataFrame({"reference": ["4111111111111111", "4222222222222"]}))
    f = fields["reference"]
    assert f.classification == "payment_data"
    assert f.model_access == "DENY"


def test_report_blocking_flags_list_deny_fields():
    df = pl.DataFrame({"password": ["x"], "region": ["East"]})
    report = PIIClassifier().classify_dataframe(df)
    assert report.blocking_flags
    assert "password" in report.blocking_flags[0]


def test_mask_for_model_drops_deny_and_masks_mask_fields():
    df = pl.DataFrame(
        {
            "ssn": ["123-45-6789"],
            "email": ["a@example.com"],
            "region": ["East"],
        }
    )
    classifier = PIIClassifier()
    report = classifier.classify_dataframe(df)
    masked = classifier.mask_for_model(df, report)

    assert "ssn" not in masked.columns  # DENY -> dropped entirely
    assert masked["email"].to_list() == ["***MASKED***"]  # MASK -> placeholder, never the raw value
    assert masked["region"].to_list() == ["East"]  # ALLOW -> untouched


def test_mask_for_model_never_leaks_the_raw_sensitive_value():
    df = pl.DataFrame({"password": ["hunter2"]})
    classifier = PIIClassifier()
    report = classifier.classify_dataframe(df)
    masked = classifier.mask_for_model(df, report)
    assert "password" not in masked.columns
