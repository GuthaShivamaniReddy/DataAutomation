"""LLM client interface.

Source of truth: AI Prompt Library, "How to Use This Prompt Library" /
"Recommended model behavior" and Section 38 "Runtime context discipline":
"Use structured outputs / JSON Schema wherever the model provider
supports it... Use JSON schema validation on every agent response.
Reject malformed outputs rather than attempting to guess what the agent
meant."

Every agent in the compiler pipeline that needs a model talks to this
interface, never to a specific provider's SDK directly. That keeps the
provider swappable and, more importantly, keeps every call point forced
through the same "validate or reject" discipline - a malformed or
schema-violating model response can never silently flow downstream as if
it were trustworthy structured data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    @abstractmethod
    def complete_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        """Request a structured completion and return a validated instance of `response_model`.

        Implementations must raise PlatformError(ErrorCode.MODEL_OUTPUT_INVALID, ...)
        rather than returning partially-valid or coerced data when the
        underlying model's response does not satisfy `response_model`.
        """
