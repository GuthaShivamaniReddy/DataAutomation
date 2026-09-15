"""Anthropic-backed LLMClient.

Not exercised by any test in this repository - no ANTHROPIC_API_KEY is
configured in this environment. The `anthropic` package is an optional
dependency (`pip install -e ".[llm]"`) and is imported lazily inside
__init__ so importing this module, or the rest of the package, never
requires it to be installed.

Structured output is forced via tool-use with `tool_choice` pinned to a
single tool whose input_schema is the response_model's JSON schema, at
temperature=0 - AI Prompt Library "Recommended model behavior": "Use
structured outputs / JSON Schema wherever the model provider supports
it. Use low randomness for requirement interpretation, planning, and
verification (typically temperature 0 to 0.2)."

Manual verification once a key exists:

    import os
    from dataos.llm.anthropic_client import AnthropicLLMClient
    from dataos.compiler.extraction import RawExtraction

    client = AnthropicLLMClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = client.complete_structured(
        system_prompt="...", user_prompt="show revenue by month",
        response_model=RawExtraction,
    )
    print(result)
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from dataos.errors import ErrorCode, PlatformError
from dataos.llm.client import LLMClient

T = TypeVar("T", bound=BaseModel)

_TOOL_NAME = "emit_result"


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5") -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                "the 'anthropic' package is not installed; install with `pip install -e \".[llm]\"` "
                "to use AnthropicLLMClient",
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.model_id = model

    def complete_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        schema = response_model.model_json_schema()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            system=system_prompt,
            tools=[{"name": _TOOL_NAME, "description": f"Emit a {response_model.__name__}", "input_schema": schema}],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": user_prompt}],
        )

        tool_use_block = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
        if tool_use_block is None:
            raise PlatformError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                f"model did not return a tool_use block for forced tool '{_TOOL_NAME}'",
            )

        tool_input = getattr(tool_use_block, "input", None)
        try:
            return response_model.model_validate(tool_input)
        except ValidationError as exc:
            raise PlatformError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                f"model output failed validation against {response_model.__name__}: {exc}",
                evidence={"raw_output": tool_input},
            ) from exc
