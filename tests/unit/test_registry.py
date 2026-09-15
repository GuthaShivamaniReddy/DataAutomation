import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry import default_registry


def test_default_registry_has_all_phase1_operations():
    ops = default_registry.list_operations()
    for expected in ("select_filter@1.0", "cast@1.0", "deduplicate@1.0", "derive@1.0", "aggregate@1.0", "export@1.0"):
        assert expected in ops


def test_unregistered_operation_raises_no_safe_operation():
    with pytest.raises(PlatformError) as excinfo:
        default_registry.get("join")  # not implemented in Phase 1
    assert excinfo.value.code == ErrorCode.NO_SAFE_OPERATION


def test_get_by_id_resolves_highest_version():
    op = default_registry.get("aggregate")
    assert op.full_id() == "aggregate@1.0"
