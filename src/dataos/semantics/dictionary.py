"""The versioned semantic dictionary.

Blueprint Section 6: "The semantic layer is the main defense against the
AI invent­ing business meaning." Nothing in this file talks to an LLM -
it is a plain, inspectable, versioned store that the SemanticResolver
looks up against. An LLM call can propose a term; only this store (and a
human's approval to add to it) can make a definition governed.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from dataos.semantics.models import SemanticCalendar, SemanticDimension, SemanticEntity, SemanticMetric


class SemanticDictionary(BaseModel):
    metrics: dict[str, SemanticMetric] = Field(default_factory=dict)
    dimensions: dict[str, SemanticDimension] = Field(default_factory=dict)
    entities: dict[str, SemanticEntity] = Field(default_factory=dict)
    calendar: SemanticCalendar | None = None

    def add_metric(self, metric: SemanticMetric) -> None:
        """Blueprint 6.1: a custom metric must never silently overwrite a
        governed definition."""
        if metric.semantic_id in self.metrics:
            raise ValueError(
                f"semantic_id '{metric.semantic_id}' already governed at version "
                f"{self.metrics[metric.semantic_id].version} - use a new semantic_id "
                f"or bump the version explicitly"
            )
        self.metrics[metric.semantic_id] = metric

    def find_metric_candidates(self, term: str) -> list[SemanticMetric]:
        return [m for m in self.metrics.values() if m.matches(term)]

    def find_dimension_candidates(self, term: str) -> list[SemanticDimension]:
        return [d for d in self.dimensions.values() if d.matches(term)]

    @classmethod
    def load_from_json(cls, path: str | Path) -> "SemanticDictionary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save_to_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
