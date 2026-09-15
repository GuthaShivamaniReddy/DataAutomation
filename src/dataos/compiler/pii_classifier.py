"""Sensitive Data & PII Classifier.

Source of truth: AI Prompt Library Section 28 "Sensitive Data and PII
Classifier": "Classify fields using schema metadata, deterministic
detectors, and organization policy. Do not expose raw sensitive values in
your response... If classification is uncertain and the field could be
highly sensitive, use the more restrictive handling until reviewed."

Deterministic code, not an LLM call - for the same reason the Security
Guard (Section 27) is: a classifier whose own output could expose the
values it is meant to protect, or whose judgment could be swayed by
manipulated content, defeats its purpose. "Do not expose raw sensitive
values in your response" is a guarantee only non-generative code can make
by construction here - classification reads column names and a bounded
value sample, but a `FieldClassification` never carries a raw value,
only counts, a category, and a masking recommendation.

Categories cover Section 28's list only where a column-name pattern
("schema metadata") or a value-shape regex (a "deterministic detector")
can honestly recognize it: personal identifiers, contact data,
authentication secrets, financial account data, payment data, government
identifiers, health data, precise location, and employee data.
"Confidential business data" and "minors' data" are not auto-classified -
both need domain policy or a derived fact (an actual age computation, an
internal-document registry) this codebase has no field for, so guessing
would violate the Global Constitution's "never guess" rule rather than
serve it.

`model_access` is a pure function of `classification`, never of
`confidence` - a LOW-confidence detection is exactly the "uncertain, could
be highly sensitive" case Section 28 says must get the *more* restrictive
handling, so confidence is never allowed to relax it toward ALLOW.
"""

from __future__ import annotations

import re
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

Classification = Literal[
    "personal_identifier",
    "contact_data",
    "authentication_secret",
    "financial_account_data",
    "payment_data",
    "government_identifier",
    "health_data",
    "precise_location",
    "employee_data",
    "public_non_sensitive",
]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
ModelAccess = Literal["ALLOW", "MASK", "DENY"]
EvidenceType = Literal["schema_metadata", "value_pattern", "schema_metadata+value_pattern"]

_MASK_PLACEHOLDER = "***MASKED***"

# Checked in this order; the first match wins, so higher-severity/less
# ambiguous categories are listed first (e.g. "ssn" before the broader
# "name" patterns).
_NAME_PATTERNS: list[tuple[Classification, re.Pattern]] = [
    (
        "authentication_secret",
        re.compile(r"(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|credential)", re.I),
    ),
    (
        "government_identifier",
        re.compile(r"(\bssn\b|social[_-]?security|passport|national[_-]?id|tax[_-]?id|\bein\b)", re.I),
    ),
    ("payment_data", re.compile(r"(credit[_-]?card|card[_-]?number|\bcvv\b|\bcvc\b|card[_-]?holder)", re.I)),
    (
        "financial_account_data",
        re.compile(r"(account[_-]?number|\biban\b|routing[_-]?number|bank[_-]?account|\bswift\b)", re.I),
    ),
    ("health_data", re.compile(r"(diagnosis|icd[_-]?code|medical|health[_-]?record|prescription)", re.I)),
    ("precise_location", re.compile(r"(latitude|longitude|\bgps\b|geo[_-]?location|precise[_-]?location)", re.I)),
    ("contact_data", re.compile(r"(e[-_]?mail|phone|mobile|street[_-]?address|zip[_-]?code|postal[_-]?code)", re.I)),
    ("employee_data", re.compile(r"(employee|\bstaff\b|hire[_-]?date|salary|payroll)", re.I)),
    (
        "personal_identifier",
        re.compile(
            r"(full[_-]?name|first[_-]?name|last[_-]?name|customer[_-]?name|date[_-]?of[_-]?birth|\bdob\b|birth[_-]?date)",
            re.I,
        ),
    ),
]

# Value-shape regexes - only meaningful over already-materialized string
# values, and only ever raise or supply a classification, never lower one.
_VALUE_PATTERNS: list[tuple[Classification, re.Pattern]] = [
    ("contact_data", re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")),
    ("government_identifier", re.compile(r"^\d{3}-\d{2}-\d{4}$")),  # US SSN shape
    ("payment_data", re.compile(r"^(?:\d[ -]?){13,19}$")),  # credit-card-shaped digit run
]

_MODEL_ACCESS: dict[Classification, ModelAccess] = {
    "authentication_secret": "DENY",
    "government_identifier": "DENY",
    "payment_data": "DENY",
    "financial_account_data": "MASK",
    "health_data": "MASK",
    "precise_location": "MASK",
    "contact_data": "MASK",
    "personal_identifier": "MASK",
    "employee_data": "MASK",
    "public_non_sensitive": "ALLOW",
}

_ACCESS_SEVERITY: dict[ModelAccess, int] = {"DENY": 2, "MASK": 1, "ALLOW": 0}

_HANDLING: dict[Classification, str] = {
    "authentication_secret": "never expose; use only through a secrets manager, never include in evidence or a prompt",
    "government_identifier": "never expose; redact entirely in any output or export",
    "payment_data": "never expose; tokenize/store only via a compliant payment processor",
    "financial_account_data": "mask before any model or export use",
    "health_data": "mask entirely; treat as protected health information pending policy review",
    "precise_location": "mask to a coarser granularity (e.g. region) before any model or export use",
    "contact_data": "mask before sending to a model",
    "personal_identifier": "mask before sending to a model; may be shown to an authorized operator",
    "employee_data": "mask before sending to a model; internal-use only pending policy review",
    "public_non_sensitive": "no masking required",
}


class FieldClassification(BaseModel):
    field: str
    classification: Classification
    confidence: Confidence
    evidence_type: EvidenceType
    model_access: ModelAccess
    handling: str
    policy_ref: str


class PIIClassificationReport(BaseModel):
    fields: list[FieldClassification] = Field(default_factory=list)
    blocking_flags: list[str] = Field(default_factory=list)


class PIIClassifier:
    def __init__(self, *, sample_size: int = 50) -> None:
        self._sample_size = sample_size

    def classify_dataframe(self, df: pl.DataFrame) -> PIIClassificationReport:
        fields = [self._classify_column(name, df[name]) for name in df.columns]

        deny_fields = [f.field for f in fields if f.model_access == "DENY"]
        blocking_flags = (
            [f"field(s) {deny_fields} classified with DENY model access - must never reach a model"]
            if deny_fields
            else []
        )

        return PIIClassificationReport(fields=fields, blocking_flags=blocking_flags)

    def mask_for_model(self, df: pl.DataFrame, report: PIIClassificationReport) -> pl.DataFrame:
        """Drops every DENY-classified column outright and replaces every
        MASK-classified column's values with a fixed placeholder - never a
        partial reveal (e.g. "last 4 digits"), since a nuanced masking
        rule this classifier cannot verify is safer to over-restrict than
        to get subtly wrong."""
        result = df
        for field_classification in report.fields:
            if field_classification.field not in result.columns:
                continue
            if field_classification.model_access == "DENY":
                result = result.drop(field_classification.field)
            elif field_classification.model_access == "MASK":
                result = result.with_columns(pl.lit(_MASK_PLACEHOLDER).alias(field_classification.field))
        return result

    def _classify_column(self, name: str, series: pl.Series) -> FieldClassification:
        name_match = next((cls for cls, pattern in _NAME_PATTERNS if pattern.search(name)), None)
        value_match = self._value_match(series)

        classification: Classification
        confidence: Confidence
        evidence_type: EvidenceType

        if name_match and value_match:
            evidence_type = "schema_metadata+value_pattern"
            if name_match == value_match:
                classification, confidence = name_match, "HIGH"
            else:
                # Conflicting signals: never resolve toward the more
                # permissive reading - pick whichever category demands
                # the stricter model_access.
                classification = max((name_match, value_match), key=lambda c: _ACCESS_SEVERITY[_MODEL_ACCESS[c]])
                confidence = "MEDIUM"
        elif name_match:
            classification, confidence, evidence_type = name_match, "MEDIUM", "schema_metadata"
        elif value_match:
            classification, confidence, evidence_type = value_match, "LOW", "value_pattern"
        else:
            classification, confidence, evidence_type = "public_non_sensitive", "MEDIUM", "schema_metadata"

        return FieldClassification(
            field=name,
            classification=classification,
            confidence=confidence,
            evidence_type=evidence_type,
            model_access=_MODEL_ACCESS[classification],
            handling=_HANDLING[classification],
            policy_ref=f"dataos.pii.{classification}",
        )

    def _value_match(self, series: pl.Series) -> Classification | None:
        if series.dtype != pl.Utf8:
            return None
        sample = [v for v in series.drop_nulls().head(self._sample_size).to_list() if v]
        if not sample:
            return None
        for classification, pattern in _VALUE_PATTERNS:
            matches = sum(1 for v in sample if pattern.match(v))
            if matches / len(sample) >= 0.5:
                return classification
        return None
