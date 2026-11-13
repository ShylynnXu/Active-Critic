from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schemas import Example


@dataclass(frozen=True)
class DatasetConfig:
    filename: str
    score_key: str
    score_max: float


# Normalize human ratings by the maximum value of each benchmark's scale.
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "summeval": DatasetConfig("summeval_normalization.json", "overall", 5.0),
    "topicalchat": DatasetConfig("topical_chat_normalization.json", "overall", 5.0),
    "sfres": DatasetConfig("sfres_normalization.json", "overall", 6.0),
    "openmeva": DatasetConfig("openmeva_normalization.json", "overall", 5.0),
}

def _normalize_text_group(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _split_unquoted(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {"'", '"'}:
            quote = None if quote == character else character if quote is None else quote
        elif character == "," and quote is None:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return parts


def _normalize_sfres_group(value: str) -> str:
    """Canonicalize a dialogue MR so reordered slots remain one source unit."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    match = re.fullmatch(r"\s*([^()]+)\((.*)\)\s*", normalized, flags=re.DOTALL)
    if not match:
        return _normalize_text_group(normalized)
    intent = _normalize_text_group(match.group(1))
    slots: list[tuple[str, str]] = []
    for raw_slot in _split_unquoted(match.group(2)):
        key, separator, raw_value = raw_slot.partition("=")
        if not separator:
            return _normalize_text_group(normalized)
        slot_name = re.sub(r"\s+", "", key).casefold()
        slot_value = _normalize_text_group(raw_value.strip().strip("'\""))
        slots.append((slot_name, slot_value))
    canonical_slots = ",".join(
        f"{key}={slot_value}" for key, slot_value in sorted(slots)
    )
    return f"{intent}({canonical_slots})"


def canonical_dataset_name(name: str) -> str:
    normalized = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "summeval": "summeval",
        "topicalchat": "topicalchat",
        "sfres": "sfres",
        "openmeva": "openmeva",
        "roc": "openmeva",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {name!r}; choose from {sorted(DATASET_CONFIGS)}"
        ) from exc


def _raw_human_rating(value, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError("A boolean human label is not a rating")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid human rating: {value!r}") from exc
    if not math.isfinite(score) or not 0 <= score <= maximum:
        raise ValueError(f"Human rating must be finite and in [0, {maximum}]: {score}")
    return score


def load_dataset(name: str, dataset_dir: str | Path) -> list[Example]:
    """Load a benchmark and normalize its overall label to [0, 100]."""

    canonical = canonical_dataset_name(name)
    config = DATASET_CONFIGS[canonical]
    path = Path(dataset_dir) / config.filename
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset must be a non-empty JSON array: {path}")

    examples: list[Example] = []
    for index, row in enumerate(rows):
        try:
            raw_score = _raw_human_rating(row["scores"][config.score_key], config.score_max)
            source, response = row["source"], row["response"]
            for field, value in (("source", source), ("response", response)):
                if not isinstance(value, str):
                    raise ValueError(f"{field} must be text")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row {index} in {path}: {exc}") from exc
        score = raw_score * 100.0 / config.score_max
        natural_id = row.get("id", row.get("doc_id", row.get("source_id", index)))
        if canonical == "summeval":
            group_id = row.get("id", source)
        elif canonical == "openmeva":
            group_id = row.get("doc_id", source)
        elif canonical == "sfres":
            group_id = _normalize_sfres_group(source)
        else:
            source_id = row.get("source_id")
            group_id = source_id or _normalize_text_group(source)
        uid = f"{canonical}:{natural_id}:{index}"
        examples.append(
            Example(
                uid=uid,
                source=source,
                response=response,
                human_score=score,
                raw_score=raw_score,
                metadata={
                    "group_id": str(group_id),
                    "dataset": canonical,
                    "system_id": str(row.get("source_id", "")) if canonical == "summeval" else str(row.get("system_id", "")),
                    "dimension_scores": dict(row["scores"]),
                },
            )
        )
    return examples


def _grouped_indices(examples: list[Example]) -> list[list[int]]:
    """Join shared source IDs AND canonical source texts, even if IDs disagree."""
    parents = list(range(len(examples)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners: dict[tuple[str, str], int] = {}
    for i, example in enumerate(examples):
        keys = [("source", _normalize_text_group(example.source))]
        if "group_id" in example.metadata:
            keys.append(("id", str(example.metadata["group_id"])))
        for key in keys:
            if key in owners:
                parents[root(i)] = root(owners[key])
            else:
                owners[key] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(examples)):
        groups[root(i)].append(i)
    return list(groups.values())


def audit_split(train: Iterable[Example], test: Iterable[Example]) -> dict:
    train, test = list(train), list(test)
    if not train or not test:
        raise ValueError("Both training and test partitions must be non-empty")
    train_ids, test_ids = {x.uid for x in train}, {x.uid for x in test}
    if len(train_ids) != len(train) or len(test_ids) != len(test):
        raise ValueError("Example UIDs must be unique")
    sources = lambda xs: {_normalize_text_group(x.source) for x in xs}
    pairs = lambda xs: {(_normalize_text_group(x.source), _normalize_text_group(x.response)) for x in xs}
    groups = lambda xs: {str(x.metadata.get("group_id", _normalize_text_group(x.source))) for x in xs}
    result = {
        "train_size": len(train), "test_size": len(test),
        "train_groups": len(_grouped_indices(train)), "test_groups": len(_grouped_indices(test)),
        "overlapping_ids": len(train_ids & test_ids),
        "overlapping_sources": len(sources(train) & sources(test)),
        "overlapping_group_ids": len(groups(train) & groups(test)),
        "duplicate_pairs_across_split": len(pairs(train) & pairs(test)),
        "duplicate_pairs_within_train": len(train) - len(pairs(train)),
        "duplicate_pairs_within_test": len(test) - len(pairs(test)),
    }
    if any(result[k] for k in ("overlapping_ids", "overlapping_sources", "overlapping_group_ids", "duplicate_pairs_across_split")):
        raise ValueError(f"Train/test leakage detected: {result}")
    return result


def split_examples(
    examples: Iterable[Example], train_ratio: float = 0.25, seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    """Keep source groups intact while honoring the 25% group and row budgets."""
    items = list(examples)
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must lie strictly between 0 and 1")
    groups = _grouped_indices(items)
    if len(groups) < 2:
        raise ValueError("A disjoint split requires at least two source groups")
    random.Random(seed).shuffle(groups)
    # Group counts are rounded to the nearest integer. For unequal-sized groups,
    # select the same number of whole groups with the nearest feasible row count.
    # This uses only source-group sizes, never human scores or response quality.
    count = min(len(groups) - 1, max(1, int(math.floor(len(groups) * train_ratio + 0.5))))
    target_rows = int(math.ceil(len(items) * train_ratio))
    chosen = _balanced_group_subset(groups, count, target_rows)
    indices = {i for position in chosen for i in groups[position]}
    train = [x for i, x in enumerate(items) if i in indices]
    test = [x for i, x in enumerate(items) if i not in indices]
    audit_split(train, test)
    return train, test


def _balanced_group_subset(groups: list[list[int]], count: int, target_rows: int) -> list[int]:
    """Cardinality-constrained subset sum using row-count bitsets.

    Equal-sized groups use the seeded prefix. Unequal groups retain that seeded
    ordering as the tie-break and match the requested row budget when feasible.
    """
    if len({len(group) for group in groups}) == 1:
        return list(range(count))
    reachable = [1] + [0] * count
    history = [tuple(reachable)]
    for position, group in enumerate(groups):
        for number in range(min(count, position + 1), 0, -1):
            reachable[number] |= reachable[number - 1] << len(group)
        history.append(tuple(reachable))
    possible = reachable[count]
    row_count = min(
        (n for n in range(possible.bit_length()) if possible & (1 << n)),
        key=lambda n: (abs(n - target_rows), n > target_rows),
    )
    chosen = []
    for position in range(len(groups) - 1, -1, -1):
        if count == 0:
            break
        if history[position][count] & (1 << row_count):
            continue
        chosen.append(position)
        row_count -= len(groups[position])
        count -= 1
    if count != 0 or row_count != 0:
        raise AssertionError("Could not reconstruct the source-group split")
    return sorted(chosen)


def labeled_subset(
    training: Iterable[Example], budget: str, total_size: int, seed: int,
) -> list[Example]:
    """Nested label budgets sampled solely from the already fixed training split."""
    items = list(training)
    if budget == "all":
        count = len(items)
    elif budget.endswith("%"):
        fraction = float(budget[:-1]) / 100
        if not 0 < fraction <= 0.25:
            raise ValueError("Percentage budget must be in (0, 25%]")
        count = len(items) if fraction == 0.25 else int(math.ceil(total_size * fraction))
    else:
        count = int(budget)
    if count < 2 or count > len(items):
        raise ValueError(f"Requested {count} labels but the training pool has {len(items)}")
    random.Random(seed).shuffle(items)
    return items[:count]


def preference_examples(examples: Iterable[Example], supervision: str) -> list[Example]:
    """Section V-B OO/FO: fluency >=4.5, ALL other three dimensions <3."""
    from dataclasses import replace

    if supervision not in {"OO", "FO"}:
        raise ValueError("supervision must be OO or FO")
    selected = []
    for example in examples:
        scores = example.metadata.get("dimension_scores", {})
        required = ("fluency", "coherence", "consistency", "relevance")
        if not all(k in scores for k in required):
            raise ValueError("OO/FO requires SummEval dimension labels")
        validated = {}
        for name in required:
            try:
                validated[name] = _raw_human_rating(scores[name], 5.0)
            except ValueError as exc:
                raise ValueError(f"Invalid OO/FO {name} rating for {example.uid}: {exc}") from exc
        scores = validated
        if scores["fluency"] >= 4.5 and all(scores[k] < 3 for k in required[1:]):
            selected.append(
                replace(example, human_score=float(scores["fluency"]) * 20,
                        raw_score=float(scores["fluency"]))
                if supervision == "FO" else example
            )
    return selected
