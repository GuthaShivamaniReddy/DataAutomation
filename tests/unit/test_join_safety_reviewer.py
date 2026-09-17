import polars as pl

from dataos.compiler.join_safety_reviewer import JoinSafetyReviewer
from dataos.ingestion.profiling import profile_dataset


def _profiles(left_df: pl.DataFrame, right_df: pl.DataFrame):
    return profile_dataset(left_df), profile_dataset(right_df)


def test_clean_one_to_one_join_with_frames_is_approved():
    left = pl.DataFrame({"order_id": [1, 2, 3], "region": ["East", "West", "East"]})
    right = pl.DataFrame({"order_id": [1, 2, 3], "customer_name": ["Alice", "Bob", "Cara"]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="customers",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="left",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_one",
        left_frame=left,
        right_frame=right,
    )

    assert result.decision == "APPROVE"
    assert result.issues == []
    assert result.join_contract is not None
    assert result.join_contract.unmatched_policy.startswith("keep every left row")


def test_missing_join_key_column_is_rejected():
    left = pl.DataFrame({"order_id": [1, 2]})
    right = pl.DataFrame({"customer_id": [1, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="customers",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_one",
    )

    assert result.decision == "REJECT"
    assert any(i.code == "KEY_MISSING" for i in result.issues)
    assert result.join_contract is None


def test_key_dtype_mismatch_is_rejected():
    left = pl.DataFrame({"order_id": [1, 2]})
    right = pl.DataFrame({"order_id": ["1", "2"]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="customers",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_one",
    )

    assert result.decision == "REJECT"
    assert any(i.code == "KEY_DTYPE_MISMATCH" for i in result.issues)


def test_many_to_many_without_opt_in_is_rejected():
    left = pl.DataFrame({"id": [1, 1, 2]})
    right = pl.DataFrame({"id": [1, 1, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="a",
        right_name="b",
        left_keys=["id"],
        right_keys=["id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="many_to_many",
    )

    assert result.decision == "REJECT"
    assert any(i.code == "UNAPPROVED_MANY_TO_MANY" for i in result.issues)


def test_many_to_many_with_opt_in_is_approved():
    left = pl.DataFrame({"id": [1, 1, 2]})
    right = pl.DataFrame({"id": [1, 1, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="a",
        right_name="b",
        left_keys=["id"],
        right_keys=["id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="many_to_many",
        allow_many_to_many=True,
    )

    assert result.decision == "APPROVE"


def test_name_similarity_only_key_evidence_is_rejected():
    left = pl.DataFrame({"order_id": [1, 2]})
    right = pl.DataFrame({"order_id": [1, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="other_orders",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="inner",
        key_evidence="NAME_SIMILARITY_ONLY",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_one",
    )

    assert result.decision == "REJECT"
    assert any(i.code == "NAME_SIMILARITY_ONLY" for i in result.issues)


def test_undeclared_cardinality_requires_clarification():
    left = pl.DataFrame({"order_id": [1, 2]})
    right = pl.DataFrame({"order_id": [1, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="customers",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality=None,
    )

    assert result.decision == "CLARIFY"
    assert any(i.code == "CARDINALITY_UNDECLARED" for i in result.issues)


def test_ambiguous_key_candidates_require_clarification():
    left = pl.DataFrame({"order_id": [1, 2]})
    right = pl.DataFrame({"order_id": [1, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="customers",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_one",
        ambiguous_key_candidates=["customer_id", "client_id"],
    )

    assert result.decision == "CLARIFY"
    assert any(i.code == "AMBIGUOUS_KEY_CANDIDATES" for i in result.issues)


def test_duplicate_key_on_side_declared_unique_is_rejected_when_frame_supplied():
    left = pl.DataFrame({"order_id": [1, 1, 2], "region": ["East", "East", "West"]})
    right = pl.DataFrame({"order_id": [1, 2], "customer_name": ["Alice", "Bob"]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="customers",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="left",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_one",
        left_frame=left,
        right_frame=right,
    )

    assert result.decision == "REJECT"
    assert any(i.code == "DUPLICATE_KEY" for i in result.issues)


def test_uniqueness_unverified_without_a_frame_requires_clarification():
    # order_id has a null, so it is not a candidate key by profiling alone,
    # but there is no dataframe to prove/disprove duplicates.
    left = pl.DataFrame({"order_id": [1, None, 3]})
    right = pl.DataFrame({"order_id": [1, 2, 3]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="orders",
        right_name="customers",
        left_keys=["order_id"],
        right_keys=["order_id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_one",
    )

    assert result.decision == "CLARIFY"
    assert any(i.code == "UNIQUENESS_UNVERIFIED" for i in result.issues)


def test_high_multiplication_factor_warns_but_still_approves():
    left = pl.DataFrame({"id": [1, 2, 3]})
    right = pl.DataFrame({"id": [1, 1, 1, 2, 2, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="a",
        right_name="b",
        left_keys=["id"],
        right_keys=["id"],
        how="inner",
        key_evidence="GOVERNED_MAPPING",
        left_profile=left_profile,
        right_profile=right_profile,
        expected_cardinality="one_to_many",
        left_frame=left,
        right_frame=right,
        max_multiplication_factor=1.5,
    )

    assert result.decision == "APPROVE"
    assert any(i.code == "AMPLIFICATION_RISK" for i in result.issues)


def test_narrative_is_none_without_an_llm_client():
    left = pl.DataFrame({"id": [1, 2]})
    right = pl.DataFrame({"id": [1, 2]})
    left_profile, right_profile = _profiles(left, right)

    result = JoinSafetyReviewer().review(
        left_name="a", right_name="b", left_keys=["id"], right_keys=["id"], how="inner",
        key_evidence="GOVERNED_MAPPING", left_profile=left_profile, right_profile=right_profile,
        expected_cardinality="one_to_one",
    )
    assert result.narrative is None


def test_narrative_is_populated_without_changing_the_decision_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    left = pl.DataFrame({"id": [1, 2]})
    right = pl.DataFrame({"id": [1, 2]})
    left_profile, right_profile = _profiles(left, right)

    reviewer = JoinSafetyReviewer(DeterministicLLMClient())
    result = reviewer.review(
        left_name="a", right_name="b", left_keys=["id"], right_keys=["id"], how="inner",
        key_evidence="GOVERNED_MAPPING", left_profile=left_profile, right_profile=right_profile,
        expected_cardinality="one_to_one",
    )

    assert result.narrative is not None
    assert result.decision == "APPROVE"
