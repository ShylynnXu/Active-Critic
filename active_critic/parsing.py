from __future__ import annotations

import json
import math
import re
from typing import Any


class OutputFormatError(ValueError):
    pass


def get_case_insensitive(
    value: dict[str, Any], key: str, default: Any = None
) -> Any:
    normalized = key.casefold()
    matches = [item for candidate, item in value.items() if str(candidate).casefold() == normalized]
    if len(matches) > 1:
        raise OutputFormatError(f"Duplicate field name: {key!r}")
    return matches[0] if matches else default


def parse_json_value(text: str) -> Any:
    """Parse the whole output; never discard prose or salvage a nested object."""
    def unique_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise OutputFormatError(f"Duplicate JSON field: {key!r}")
            value[key] = item
        return value

    def reject_constant(value):
        raise OutputFormatError(f"Nonfinite JSON constant: {value}")

    if not isinstance(text, str):
        raise OutputFormatError("Expected JSON text")
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.I | re.S)
    if fenced:
        cleaned = fenced.group(1)
    try:
        return json.loads(cleaned, object_pairs_hook=unique_fields, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise OutputFormatError("The entire model response must be one valid JSON value") from exc


def parse_json_object(text: str) -> dict[str, Any]:
    value = parse_json_value(text)
    if not isinstance(value, dict):
        raise OutputFormatError("Expected a JSON object")
    return value


def clamp_score(value: Any) -> float:
    """Validate a scalar score without silently extracting or clipping a number."""
    if isinstance(value, bool):
        raise OutputFormatError("A boolean is not a score")
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", value)
        if not match:
            raise OutputFormatError(f"Cannot parse score from {value!r}")
        value = match.group(1)
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise OutputFormatError(f"Invalid score: {value!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise OutputFormatError(f"Score outside [0, 100]: {score}")
    return score


def parse_criteria(value: Any, minimum: int = 4, maximum: int | None = 10) -> dict[str, str]:
    if isinstance(value, str):
        value = parse_json_value(value)
    if isinstance(value, dict) and "criteria" in {str(k).lower() for k in value}:
        value = get_case_insensitive(value, "criteria")
    if not isinstance(value, dict):
        raise OutputFormatError("Criteria must be a JSON object")
    criteria = {}
    normalized_names = set()
    for name, description in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(description, str) or not description.strip():
            raise OutputFormatError("Every criterion needs a nonempty name and string definition")
        normalized = "".join(c for c in name.casefold() if c.isalnum())
        if not normalized or normalized in normalized_names:
            raise OutputFormatError(f"Duplicate or invalid criterion name: {name!r}")
        normalized_names.add(normalized)
        criteria[name.strip()] = description.strip()
    if len(criteria) < minimum or (maximum is not None and len(criteria) > maximum):
        raise OutputFormatError(
            f"Expected {minimum}-{maximum if maximum is not None else 'unbounded'} criteria, received {len(criteria)}"
        )
    return criteria
