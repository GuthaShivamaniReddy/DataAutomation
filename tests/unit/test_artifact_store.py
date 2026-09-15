from dataos.workflow.artifact_store import ArtifactStore


def test_save_and_load_round_trip(orders_basic_df, tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.save("run_1", "s1", orders_basic_df)

    assert ref.row_count == orders_basic_df.height
    reloaded = store.load(ref)
    assert reloaded.equals(orders_basic_df)


def test_load_by_ids_matches_save(orders_basic_df, tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    store.save("run_1", "s1", orders_basic_df)

    reloaded = store.load_by_ids("run_1", "s1")
    assert reloaded.equals(orders_basic_df)


def test_checksum_is_stable_for_identical_content(orders_basic_df, tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    ref_a = store.save("run_1", "s1", orders_basic_df)
    ref_b = store.save("run_2", "s1", orders_basic_df)
    assert ref_a.checksum == ref_b.checksum


def test_source_node_id_with_colon_is_path_safe(orders_basic_df, tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    store.save("run_1", "source:orders", orders_basic_df)
    reloaded = store.load_by_ids("run_1", "source:orders")
    assert reloaded.equals(orders_basic_df)
