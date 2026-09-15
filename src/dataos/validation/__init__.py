"""Standalone Validation Engine.

Source of truth: Reliability-First Master Blueprint Section 10
"Validation and Verification Pyramid" and Section 4's reference
architecture, which shows the Validation Engine as its own component
downstream of the Operation Registry - these are cross-artifact checks
(conservation, reconciliation, dual computation) that are not any single
operation's job.

Checks are plain functions returning `dataos.evidence.models.ValidationResult`
- not a declarative "CheckSpec" DSL - since nothing in the platform yet
produces such specs (that is a future Planner-phase concern). A later
phase's Independent Verifier / Release Gate consumes a `ValidationReport`.
"""

from dataos.validation.engine import ValidationEngine, ValidationReport

__all__ = ["ValidationEngine", "ValidationReport"]
