from dataos.compiler.requirement_compiler import RequirementCompiler
from dataos.contracts.requirement_contract import DefinitionStatus, RequirementStatus
from dataos.llm.deterministic import DeterministicLLMClient


def _compiler(sample_semantic_dictionary) -> RequirementCompiler:
    return RequirementCompiler(DeterministicLLMClient(), sample_semantic_dictionary)


def test_governed_metric_with_sources_and_no_timezone_gap_stays_draft(sample_semantic_dictionary):
    compiler = _compiler(sample_semantic_dictionary)
    output = compiler.compile(user_request="show order count", sources=["orders"])

    assert output.contract.status == RequirementStatus.DRAFT
    assert output.contract.clarifications == []
    assert output.contract.metrics[0].definition_status == DefinitionStatus.GOVERNED
    assert output.contract.metrics[0].formula == "count(orders.order_id)"
    assert output.envelope.status == "OK"


def test_ambiguous_revenue_blocks(sample_semantic_dictionary):
    # Golden scenario (Blueprint Appendix C / Prompt Library Section 39): "Ambiguous
    # revenue" - dataset/dictionary has multiple revenue-like definitions with no single
    # governed answer -> must CLARIFY, never silently pick one.
    compiler = _compiler(sample_semantic_dictionary)
    output = compiler.compile(user_request="show revenue by region", sources=["orders"])

    assert output.contract.status == RequirementStatus.BLOCKED
    assert output.contract.metrics[0].definition_status == DefinitionStatus.AMBIGUOUS
    assert any("revenue" in c.issue for c in output.contract.clarifications)
    assert output.envelope.status == "NEEDS_CLARIFICATION"


def test_no_sources_blocks(sample_semantic_dictionary):
    compiler = _compiler(sample_semantic_dictionary)
    output = compiler.compile(user_request="show order count", sources=[])
    assert output.contract.status == RequirementStatus.BLOCKED


def test_material_term_with_no_governed_definition_blocks(sample_semantic_dictionary):
    compiler = _compiler(sample_semantic_dictionary)
    output = compiler.compile(user_request="show growth", sources=["orders"])
    assert output.contract.status == RequirementStatus.BLOCKED
    assert output.contract.metrics[0].definition_status == DefinitionStatus.MISSING


def test_date_bucket_without_timezone_blocks(sample_semantic_dictionary):
    # Golden scenario: "Timezone boundary" - requirement must resolve reporting
    # timezone before daily/monthly aggregation.
    compiler = _compiler(sample_semantic_dictionary)
    output = compiler.compile(user_request="show order count by month", sources=["orders"])
    assert output.contract.status == RequirementStatus.BLOCKED
    assert any("timezone" in c.question.lower() for c in output.contract.clarifications)


def test_compiler_never_sets_approved_itself(sample_semantic_dictionary):
    compiler = _compiler(sample_semantic_dictionary)
    output = compiler.compile(user_request="show order count", sources=["orders"])
    # Even a fully clean, unblocked contract stays DRAFT - approval is a separate
    # human/policy action (Blueprint Section 18), never something the compiler grants itself.
    assert output.contract.status != RequirementStatus.APPROVED
    assert output.contract.ready_for_planning is False
