from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    """One normalized ``(source, response, human score)`` tuple."""

    uid: str
    source: str
    response: str
    human_score: float | None
    raw_score: float | None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def prompt_record(self, *, include_score: bool = True) -> dict[str, Any]:
        result = {
            "Source": self.source,
            "Response": self.response,
        }
        if include_score:
            if self.human_score is None:
                raise ValueError("Human feedback is required unless explicitly disabled")
            result["Human Score"] = self.human_score
        return result


@dataclass
class TaskInstruction:
    task_type: str
    task_description: str
    relationship: str
    criteria: dict[str, str]

    def full_task_description(self) -> str:
        description = self.task_description.strip()
        relationship = self.relationship.strip()
        if not relationship or relationship.casefold() in description.casefold():
            return description
        if not relationship.casefold().startswith("the relationship between"):
            relationship = (
                "The relationship between ‘Source’ and ‘Response’ is "
                + relationship
            )
        if not relationship.endswith((".", "!", "?")):
            relationship += "."
        return f"{description} {relationship}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskInstruction":
        return cls(
            task_type=str(value["task_type"]),
            task_description=str(value["task_description"]),
            relationship=str(value["relationship"]),
            criteria={str(k): str(v) for k, v in value["criteria"].items()},
        )


@dataclass
class CriterionResult:
    explanation: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"explanation": self.explanation, "score": self.score}


@dataclass
class EvaluationResult:
    uid: str
    source: str
    response: str
    human_score: float | None
    criterion_results: dict[str, CriterionResult]
    overall_explanation: str
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source": self.source,
            "response": self.response,
            "human_score": self.human_score,
            "criteria": {
                name: result.to_dict()
                for name, result in self.criterion_results.items()
            },
            "overall": {
                "explanation": self.overall_explanation,
                "score": self.overall_score,
            },
        }
