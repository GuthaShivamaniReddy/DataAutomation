"""Requirement Compiler orchestration.

Source of truth: AI Prompt Library "Agent pipeline" ordering:
Requirement Compiler -> Ambiguity Gate -> Semantic Resolver, and Section
38 "Prompt chaining patterns" / "One-time analysis" chain.

This class wires together the LLM extraction step (`RawExtraction`), the
deterministic SemanticResolver, and the deterministic AmbiguityGate into
one `RequirementContract`. It never marks a contract APPROVED itself -
that stays a human/policy action (Blueprint Section 18 UX flow: "User
approves when policy requires it"). The contract's own Phase-0 model
validator (dataos.contracts.requirement_contract) is the single,
already-tested source of truth for the final BLOCKED / ready_for_planning
determination.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from dataos.compiler.ambiguity_gate import AmbiguityGate
from dataos.compiler.constitution import GLOBAL_CONSTITUTION, AgentEnvelope
from dataos.compiler.extraction import ExtractedMetric, RawExtraction
from dataos.compiler.prompts import REQUIREMENT_COMPILER_SYSTEM_PROMPT
from dataos.compiler.semantic_resolver import ResolvedTerm, SemanticResolver
from dataos.contracts.requirement_contract import (
    DefinitionStatus,
    Metric,
    RequirementContract,
    RequirementStatus,
    Source,
    TimeSpec,
)
from dataos.ingestion.profiling import DatasetProfile
from dataos.llm.client import LLMClient
from dataos.semantics.dictionary import SemanticDictionary

REQUIREMENT_COMPILER_PROMPT_VERSION = "1.0"


class CompilerOutput(BaseModel):
    contract: RequirementContract
    envelope: AgentEnvelope


class RequirementCompiler:
    def __init__(self, llm_client: LLMClient, semantic_dictionary: SemanticDictionary) -> None:
        self._llm_client = llm_client
        self._resolver = SemanticResolver(semantic_dictionary)
        self._gate = AmbiguityGate()

    def compile(
        self,
        *,
        user_request: str,
        sources: list[str],
        profile: DatasetProfile | None = None,
    ) -> CompilerOutput:
        """`sources` is caller-supplied (the datasets the user has already
        uploaded/selected), never guessed from prose - Blueprint Appendix
        A: "sources: Which exact snapshots/systems?" is a UI-level
        selection, not an NLU inference."""

        extraction = self._llm_client.complete_structured(
            system_prompt=f"{GLOBAL_CONSTITUTION}\n\n{REQUIREMENT_COMPILER_SYSTEM_PROMPT}",
            user_prompt=user_request,
            response_model=RawExtraction,
        )

        resolved_metrics = [self._resolver.resolve_metric(m.term) for m in extraction.metrics]

        gate_result = self._gate.evaluate(
            extraction=extraction,
            resolved_metrics=resolved_metrics,
            sources=sources,
            profile=profile,
        )

        metrics = [
            _build_metric(extracted, resolved)
            for extracted, resolved in zip(extraction.metrics, resolved_metrics)
        ]

        contract = RequirementContract(
            objective=extraction.objective,
            sources=[Source(name=s) for s in sources],
            metrics=metrics,
            dimensions=extraction.dimensions,
            filters=extraction.filters,
            group_by=extraction.group_by,
            time=TimeSpec(range=extraction.time_range, timezone=extraction.timezone),
            clarifications=gate_result.clarifications,
            status=RequirementStatus.DRAFT,
        )

        envelope = AgentEnvelope(
            run_id=str(uuid.uuid4()),
            agent="requirement_compiler",
            agent_prompt_version=REQUIREMENT_COMPILER_PROMPT_VERSION,
            model_id=getattr(self._llm_client, "model_id", type(self._llm_client).__name__),
            status="NEEDS_CLARIFICATION" if contract.status == RequirementStatus.BLOCKED else "OK",
            result={"objective": contract.objective},
            assumptions=list(contract.assumptions),
            next_action=(
                "resolve clarifications before planning"
                if contract.status == RequirementStatus.BLOCKED
                else "await human approval before planning"
            ),
        )

        return CompilerOutput(contract=contract, envelope=envelope)


def _build_metric(extracted: ExtractedMetric, resolved: ResolvedTerm) -> Metric:
    if resolved.status == "GOVERNED":
        governed = resolved.metric
        assert governed is not None
        return Metric(
            name=resolved.term,
            definition=governed.formula,
            formula=governed.formula,
            source_fields=governed.source_fields,
            definition_status=DefinitionStatus.GOVERNED,
        )
    if resolved.status == "AMBIGUOUS":
        return Metric(name=resolved.term, definition_status=DefinitionStatus.AMBIGUOUS)
    # UNGOVERNED
    if extracted.formula:
        return Metric(
            name=resolved.term,
            formula=extracted.formula,
            definition_status=DefinitionStatus.EXPLICIT,
        )
    return Metric(name=resolved.term, definition_status=DefinitionStatus.MISSING)
