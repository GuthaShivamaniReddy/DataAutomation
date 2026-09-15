from dataos.compiler.semantic_resolver import SemanticResolver


def test_resolve_governed_metric(sample_semantic_dictionary):
    resolver = SemanticResolver(sample_semantic_dictionary)
    resolved = resolver.resolve_metric("orders")
    assert resolved.status == "GOVERNED"
    assert resolved.metric.semantic_id == "order_count"


def test_resolve_ambiguous_metric(sample_semantic_dictionary):
    resolver = SemanticResolver(sample_semantic_dictionary)
    resolved = resolver.resolve_metric("revenue")
    assert resolved.status == "AMBIGUOUS"
    assert set(resolved.candidate_ids) == {"recognized_revenue", "gross_revenue"}
    assert resolved.metric is None


def test_resolve_ungoverned_metric(sample_semantic_dictionary):
    resolver = SemanticResolver(sample_semantic_dictionary)
    resolved = resolver.resolve_metric("churn rate")
    assert resolved.status == "UNGOVERNED"
    assert resolved.metric is None
    assert resolved.candidate_ids == []
