"""Shared helper for the optional LLM narrative call used by
`ValidationRuleGenerator`, `ReconciliationAgent`, `IndependentVerifier`,
`ConfidenceScorer`, `ReleaseGate`, and `SchemaDriftMonitor`.

A thin wrapper around `RawNarrative` - not a new abstraction over what any
of those components decide. See `raw_narrative.py`'s own docstring for
the one rule this helper exists to make trivially satisfiable at every
call site: the caller's actual decision is always computed before this is
ever invoked, and this only ever returns prose to attach alongside it.
"""

from __future__ import annotations

import json

from dataos.compiler.constitution import GLOBAL_CONSTITUTION
from dataos.compiler.raw_narrative import RawNarrative
from dataos.llm.client import LLMClient


def narrate(llm_client: LLMClient, *, system_prompt: str, items: list[dict]) -> dict[str, str]:
    """`items` is `[{"ref": ..., "facts": {...}}]` - already-computed,
    non-raw-data facts safe to hand to a model. Returns `{ref: narrative}`
    only for refs the model actually answered; callers must tolerate a
    missing ref (fall back to no narrative for that item) rather than
    assume full coverage."""
    raw = llm_client.complete_structured(
        system_prompt=f"{GLOBAL_CONSTITUTION}\n\n{system_prompt}",
        user_prompt=json.dumps({"items": items}),
        response_model=RawNarrative,
    )
    return {item.ref: item.narrative for item in raw.items}
