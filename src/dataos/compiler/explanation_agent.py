"""Report & Explanation Agent.

Source of truth: AI Prompt Library Section 18 "Report & Explanation
Agent": "Write the user-facing answer using only RELEASED result objects
and evidence supplied to you... Never recompute a number. Never introduce
a metric not present in the released evidence. Cite each material
conclusion to its evidence ID internally... Do not hide validation
warnings." Also "Recommended model behavior": "Never allow an Explanation
Agent to alter numbers. It may only cite values from a verified result
object."

This is the one agent in the whole pipeline whose actual job is prose
generation, so - unlike the deterministic gates elsewhere in this package
- it genuinely calls an LLM. The non-negotiable rule above is still
enforced deterministically, never left to prompt discipline alone: every
finding's `evidence_refs` must resolve to evidence this call was actually
given (a metric name or step id present in the rendered context); a
finding citing anything else fails MODEL_OUTPUT_INVALID rather than being
trusted downstream, and every caller-supplied warning is force-appended
to `limitations` so a model can never silently drop one.
"""

from __future__ import annotations

import json
import uuid
from typing import Literal

from pydantic import BaseModel

from dataos.compiler.constitution import GLOBAL_CONSTITUTION, AgentEnvelope
from dataos.compiler.prompts import REPORT_EXPLANATION_SYSTEM_PROMPT
from dataos.compiler.raw_explanation import RawExplanation
from dataos.errors import ErrorCode, PlatformError
from dataos.llm.client import LLMClient

EXPLANATION_AGENT_PROMPT_VERSION = "1.0"

Audience = Literal["analyst", "operator", "executive", "customer", "api"]


class ExplanationOutput(BaseModel):
    explanation: RawExplanation
    envelope: AgentEnvelope


class ExplanationAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def explain(
        self,
        *,
        run_id: str,
        evidence_context: dict,
        known_evidence_refs: frozenset[str],
        forced_limitations: list[str] | None = None,
        audience: Audience = "analyst",
    ) -> ExplanationOutput:
        user_prompt = json.dumps({**evidence_context, "audience": audience})

        raw = self._llm_client.complete_structured(
            system_prompt=f"{GLOBAL_CONSTITUTION}\n\n{REPORT_EXPLANATION_SYSTEM_PROMPT}",
            user_prompt=user_prompt,
            response_model=RawExplanation,
        )

        cited_refs = {ref for finding in raw.findings for ref in finding.evidence_refs}
        unknown_refs = sorted(cited_refs - known_evidence_refs)
        if unknown_refs:
            raise PlatformError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                "explanation cites evidence_refs that were never supplied to this call",
                evidence={"unknown_evidence_refs": unknown_refs, "known_evidence_refs": sorted(known_evidence_refs)},
            )

        limitations = list(raw.limitations)
        for warning in forced_limitations or []:
            if warning not in limitations:
                limitations.append(warning)
        explanation = raw.model_copy(update={"limitations": limitations})

        envelope = AgentEnvelope(
            run_id=str(uuid.uuid4()),
            agent="explanation_agent",
            agent_prompt_version=EXPLANATION_AGENT_PROMPT_VERSION,
            model_id=getattr(self._llm_client, "model_id", type(self._llm_client).__name__),
            status="OK",
            result={"summary": explanation.summary, "workflow_run_id": run_id},
            next_action="deliver to audience",
        )

        return ExplanationOutput(explanation=explanation, envelope=envelope)
